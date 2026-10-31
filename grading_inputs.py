"""Read only this turn's explicit quotes/forwards; never use agent prose/history."""
import asyncio
import json
import re

try:
    from .grading_document import GradingError
except ImportError:
    from grading_document import GradingError


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def kind(value):
    value_type = field(value, 'type', '')
    return str(getattr(value_type, 'value', value_type) or type(value).__name__).lower()


def placeholder(text):
    return bool(re.fullmatch(r'\s*\[(?:转发消息|合并转发|forward message|图片|image)\]\s*', str(text or ''), re.I))


def has_submission_text(text):
    text = re.sub(r'\[At:[^]]+\]|<at\b[^>]*>.*?</at>', '', str(text or ''), flags=re.I).strip()
    text = re.sub(r'^/?批改(?:\s+|$)', '', text).strip()
    if not text or placeholder(text):
        return False
    request_only = r'(?:请|麻烦|帮我|帮忙|给我|你|能否|能不能|可以|帮|我|把|这份|这道|这个|这些|题目|作答|答案|试卷|作业|图片|转发|聊天记录|批改|批阅|检查|看看|一下|下|评分|打分|并|和|的|进行|吧|吗|呀|谢谢|重新|再|[\s，。！？,.!?])*'
    return re.fullmatch(request_only, text) is None


async def read_references(event):
    """Return texts and image components from complete explicitly referenced chains.

    Failed/oversized nested references abort the whole submission, even if some
    earlier nodes were readable. API calls are restricted to IDs in this turn.
    """
    texts, images, active, fetched = [], [], set(), set()
    visited = 0
    pdf_reference = False
    client = None

    async def lookup(action, ident, depth):
        nonlocal client
        if not ident:
            raise GradingError('引用或转发缺少消息 ID，请重新转发题目与作答')
        key = (action, str(ident))
        if key in active:
            raise GradingError('转发记录存在循环引用，请直接发送题目与作答')
        if key in fetched:
            return
        if len(fetched) >= 32:
            raise GradingError('引用或转发记录过多，请减少单次提交量；本次未保存')
        if client is None:
            from astrbot.core.utils.quoted_message.onebot_client import OneBotClient
            client = OneBotClient(event)
        active.add(key);fetched.add(key)
        try:
            raw = await getattr(client, action)(str(ident))
            if not isinstance(raw, dict) or raw.get('status') == 'failed' or raw.get('retcode', 0) not in (0, None):
                raise GradingError('无法读取引用或转发记录，请重新转发，或直接发送原图；本次未保存')
            data = raw.get('data') if isinstance(raw.get('data'), dict) else raw
            if action == 'get_forward_msg':
                nodes = data.get('messages') or data.get('message') or data.get('nodes') or data.get('nodeList')
                if not isinstance(nodes, list) or not nodes:
                    raise GradingError('转发记录为空或已失效，请重新转发；本次未保存')
                for node in nodes:
                    if not isinstance(node, dict):
                        raise GradingError('转发记录格式无法读取，请直接发送原图；本次未保存')
                    body = node.get('data', node) if node.get('type') == 'node' else node
                    await visit(content(body.get('message') or body.get('content')), depth + 1)
            else:
                await visit(content(data.get('message') or data.get('messages') or data.get('raw_message')), depth + 1)
        finally:
            active.remove(key)

    def content(value):
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (ValueError, TypeError):
                decoded = None
            if isinstance(decoded, list):
                return decoded
            # A CQ reference must not be mistaken for the actual question.
            if '[CQ:' in value or placeholder(value):
                raise GradingError('引用记录只有占位内容，请重新转发或直接发送原图；本次未保存')
            return [{'type': 'text', 'data': {'text': value}}]
        if not isinstance(value, list) or not value:
            raise GradingError('引用记录没有可读取的内容，请重新发送题目与作答；本次未保存')
        return value

    async def visit(parts, depth=0):
        nonlocal visited, pdf_reference
        if depth > 8:
            raise GradingError('引用嵌套过深，请直接发送题目与作答；本次未保存')
        for part in parts:
            visited += 1
            if visited > 1000:
                raise GradingError('转发记录过长，请减少单次提交量；本次未保存')
            name = kind(part)
            data = field(part, 'data', None) if isinstance(part, dict) else part
            data = data if data is not None else part
            if name == 'dict' and isinstance(data, dict) and ('message' in data or 'content' in data):
                name = 'node'
            if name in ('plain', 'text'):
                text = str(field(data, 'text', '') or '')
                if placeholder(text) and re.search(r'转发|forward', text, re.I):
                    raise GradingError('转发记录只有占位内容，请重新转发或直接发送原图；本次未保存')
                if text and not placeholder(text):
                    texts.append(text)
            elif name in ('at', 'atall'):
                continue
            elif name == 'image':
                images.append(part)
                if len(images) > 64:
                    raise GradingError('单次引用图片超过 64 张，请分批提交；本次未保存')
            elif name in ('reply', 'quote'):
                chain = field(data, 'chain', None) or field(data, 'content', None)
                if chain and not all(kind(x) in ('plain', 'text') and placeholder(field(field(x, 'data', x), 'text', '')) for x in chain):
                    await visit(content(chain), depth + 1)
                else:
                    await lookup('get_msg', field(data, 'id', None), depth)
            elif name in ('forward', 'forward_msg'):
                await lookup('get_forward_msg', field(data, 'id', None) or field(data, 'message_id', None), depth)
            elif name == 'node':
                body = field(data, 'content', None) or field(data, 'message', None)
                if body:
                    await visit(content(body), depth + 1)
                else:
                    await lookup('get_msg', field(data, 'id', None), depth)
            elif name == 'nodes':
                nodes = field(data, 'nodes', None) or field(data, 'content', None)
                await visit(content(nodes), depth + 1)
            elif name == 'json':
                card = field(data, 'data', '')
                try:
                    card = json.loads(card) if isinstance(card, str) else card
                    detail = card['meta']['detail']
                    ident = detail.get('resid') or detail.get('uniseq')
                except (ValueError, TypeError, KeyError):
                    ident = None
                if not ident:
                    raise GradingError('无法读取引用的消息卡片，请直接发送题目与作答')
                await lookup('get_forward_msg', ident, depth)
            elif name == 'file' and str(field(data, 'name', '') or field(data, 'file_name', '')).lower().endswith('.pdf'):
                pdf_reference = True  # The caller resolves known /pdf thread context.
            else:
                raise GradingError('引用中包含无法读取的内容，请直接发送题目文字或原图；本次未保存')
            if sum(map(len, texts)) > 60000:
                raise GradingError('引用文字过长，请减少单次提交量；本次未保存')

    refs = [p for p in getattr(getattr(event, 'message_obj', None), 'message', []) or []
            if kind(p) in ('reply', 'quote', 'forward', 'forward_msg', 'node', 'nodes', 'json')]
    try:
        async with asyncio.timeout(30):
            await visit(refs)
    except asyncio.CancelledError:
        raise
    except GradingError:
        raise
    except Exception as exc:
        raise GradingError('引用或转发读取失败，请重新转发或直接发送原图；本次未保存') from exc
    if refs and not texts and not images and not pdf_reference:
        raise GradingError('引用记录没有题目或作答内容；本次未保存')
    return '\n'.join(texts).strip(), images, bool(refs)
