"""Request-local telemetry; no prompts, image URLs, API keys or answers are saved."""
from contextvars import ContextVar
from functools import wraps
import logging

try:
    from .pdf_task_stats import task_scope
except ImportError:
    from pdf_task_stats import task_scope

_current_job = ContextVar('mathsolve_usage_job', default=None)
_log = logging.getLogger('astrbot')


def record_model_call(provider):
    job = _current_job.get()
    if job is not None:
        try:
            stats, scope, token = job
            stats.record_call(scope, token, provider)
        except Exception as exc:
            _log.warning('[mathsolve] 模型调用统计暂不可用: %s', type(exc).__name__)


def tracked_generation(kind):
    def decorate(fn):
        @wraps(fn)
        async def run(self, problem_text, event, *args, **kwargs):
            # Check before counting. Reuse the same admission ticket throughout the job.
            if kind == 'spdf':
                kwargs['_admission'] = self._spdf_require_access(event, kwargs.get('_admission'))
            token = kwargs.pop('_usage_token', None)
            context_token = None
            try:
                stats = self._pdf_get_task_stats()
                scope = task_scope(event)
                self._usage_register_group(event, scope)
                # /pdf commands already recorded admission; tools are separate from the curve.
                token = token or stats.record_job(scope, 'pdf_tool' if kind == 'pdf' else kind)
                context_token = _current_job.set((stats, scope, token))
            except Exception as exc:
                _log.warning('[mathsolve] PDF 调用统计暂不可用: %s', type(exc).__name__)
            try:
                return await fn(self, problem_text, event, *args, **kwargs)
            finally:
                if context_token is not None:
                    _current_job.reset(context_token)
        return run
    return decorate
