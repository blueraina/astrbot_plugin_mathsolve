"""Local, per-group PDF usage images and restart-safe daily delivery."""
import asyncio
from datetime import datetime, timedelta
import hashlib
import json
import logging
from pathlib import Path
import re
import time
import uuid

try:
    from .pdf_task_stats import BEIJING_TZ, task_scope
    from .resource_cleanup import owned
except ImportError:
    from pdf_task_stats import BEIJING_TZ, task_scope
    from resource_cleanup import owned

logger = logging.getLogger('astrbot')


def report_html(data):
    template = Path(__file__).with_name('usage_report.html').read_text(encoding='utf-8')
    # No untrusted markup: all labels are assigned with textContent in the template.
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return template.replace('/*__REPORT_DATA__*/null', payload)


class UsageReportMixin:
    def _usage_register_group(self, event, scope):
        from astrbot.core.platform.message_session import MessageSession
        from astrbot.core.platform.message_type import MessageType
        platform, kind, group = json.loads(scope)
        if kind == 'group':
            # Per-user UMO isolation must not change the actual group destination.
            target = str(MessageSession(platform, MessageType.GROUP_MESSAGE, group))
            self._pdf_get_task_stats().register_group(scope, target, group)

    def _start_usage_report_scheduler(self):
        self._pdf_get_task_stats()
        self._usage_render_lock = asyncio.Lock()
        self._usage_manual_last = {}
        self._usage_scheduler_task = asyncio.create_task(self._usage_scheduler_loop())

    async def _stop_usage_report_scheduler(self):
        task = getattr(self, '_usage_scheduler_task', None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._usage_scheduler_task = None

    async def _usage_render_report(self, data):
        async with self._usage_render_lock, self._render_sema:
            await self._ensure_browser()
            if self._browser is None:
                raise RuntimeError('统计图浏览器暂不可用')
            root = Path(self.DATA_DIR) / 'usage_reports'
            root.mkdir(parents=True, exist_ok=True)
            # Only this feature's disposable PNGs; never touch existing PDFs or caches.
            files = sorted(root.glob('report-*.png'), key=lambda p: p.stat().st_mtime, reverse=True)
            for index, old in enumerate(files):
                if index >= 100 or time.time()-old.stat().st_mtime > 172800:
                    old.unlink(missing_ok=True)
            path = root / ('report-' + uuid.uuid4().hex + '.png')
            async with owned(
                lambda: self._browser.new_context(viewport={'width': 1280, 'height': 1100}, device_scale_factor=1.5),
                lambda context: context.close(),
            ) as browser_context:
                await browser_context.route('**/*', lambda route: route.abort())
                page = await browser_context.new_page()
                page.set_default_timeout(30000)
                await page.set_content(report_html(data), wait_until='load')
                await page.wait_for_function('window.reportReady === true')
                await page.evaluate('document.fonts.ready')
                await page.locator('#report').screenshot(path=str(path), timeout=30000)
            return str(path)

    async def _usage_handle_manual(self, event):
        scope = task_scope(event)
        _, kind, group = json.loads(scope)
        if kind != 'group':
            return '请在群聊中发送 /pdf统计。'
        if self._usage_render_lock.locked():
            return '统计图正在生成，请稍后再试。'
        last = self._usage_manual_last.get(scope)
        if last is not None and time.monotonic()-last < 30:
            return '统计图刚刚生成过，请等待 30 秒后再试。'
        self._usage_manual_last[scope] = time.monotonic()
        try:
            now = datetime.now(BEIJING_TZ)
            data = self._pdf_get_task_stats().report(scope, now.date(), now)
            data.update(group_label='群 ' + group, sample=False)
            path = await asyncio.wait_for(self._usage_render_report(data), timeout=60)
            from astrbot.api.message_components import Image
            await event.send(self._build_msg_chain_from_components([Image.fromFileSystem(path)]))
            return ''
        except BaseException:
            self._usage_manual_last.pop(scope, None)
            raise

    async def _usage_scheduler_tick(self, now=None):
        if not bool(self._cfg('pdf_usage_report_enabled', True)):
            return
        now = now or datetime.now(BEIJING_TZ)
        send_time = str(self._cfg('pdf_usage_report_time', '00:00') or '00:00').strip()
        if not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', send_time):
            logger.warning('[mathsolve] PDF 统计日报发送时间无效，应为 HH:MM')
            return
        if now.strftime('%H:%M') < send_time:
            return
        day = now.date()-timedelta(days=1)
        configured = self._cfg('pdf_usage_report_groups', []) or []
        allowed = {str(v).strip() for v in configured} if isinstance(configured, list) else set(re.split(r'[,，\s]+', str(configured)))
        stats = self._pdf_get_task_stats()
        for scope, target, group in stats.delivery_groups():
            if allowed and group not in allowed:
                continue
            data = stats.report(scope, day, now)
            # Do not send fabricated pre-installation reports or inactive-group spam.
            if not any(v is not None for v in data['pdf']):
                continue
            if bool(self._cfg('pdf_usage_report_skip_empty', True)) and not (data['pdf_total'] or data['deep_total'] or data['models']):
                continue
            if not stats.claim_delivery(scope, day.isoformat(), now):
                continue
            try:
                data.update(group_label='群 ' + group, sample=False)
                path = await asyncio.wait_for(self._usage_render_report(data), timeout=90)
                from astrbot.api.message_components import Image
                message = self._build_msg_chain_from_components([Image.fromFileSystem(path)])
                sent = await asyncio.wait_for(self.context.send_message(target, message), timeout=60)
                if sent is not True:
                    raise RuntimeError('未找到可发送的平台')
                stats.mark_delivered(scope, day.isoformat())
                logger.info('[mathsolve] PDF 统计日报已发送: day=%s scope=%s', day, hashlib.sha256(scope.encode()).hexdigest()[:8])
            except Exception as exc:
                logger.warning('[mathsolve] PDF 统计日报发送失败，10 分钟后重试: %s', type(exc).__name__)

    async def _usage_scheduler_loop(self):
        while True:
            try:
                await self._usage_scheduler_tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning('[mathsolve] PDF 统计日报暂不可用: %s', type(exc).__name__)
            await asyncio.sleep(30)
