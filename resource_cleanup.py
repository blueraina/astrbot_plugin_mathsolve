"""Bounded, cancellation-safe cleanup for this plugin's own resources only."""
import asyncio
from contextlib import asynccontextmanager
import logging
import os
import signal
import subprocess
from pathlib import Path

logger = logging.getLogger('astrbot')


async def cleanup(*actions, timeout=8):
    """Finish independent cleanup actions before propagating caller cancellation."""
    async def finish():
        for action in actions:
            try:
                await asyncio.wait_for(action(), timeout=timeout)
            except (Exception, asyncio.CancelledError) as exc:
                # Cleanup must not mask the original render error or skip later steps.
                logger.warning('[mathsolve cleanup] %s', type(exc).__name__)
    task = asyncio.create_task(finish())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


async def acquire(factory, close):
    """Do not lose an asynchronously created resource if its caller is cancelled."""
    task = asyncio.create_task(factory())
    try:
        return await asyncio.shield(task)
    except BaseException:
        async def dispose_created():
            resource = await task
            await close(resource)
        await cleanup(dispose_created, timeout=35)
        raise


@asynccontextmanager
async def owned(factory, close):
    resource = await acquire(factory, close)
    try:
        yield resource
    finally:
        await cleanup(lambda: close(resource))


async def _stop_process(process):
    if process.returncode is None:
        try:
            if os.name == 'posix':
                # Only the private session created by run_process, never a global kill.
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
    await process.wait()


async def run_process(args, *, cwd=None, env=None, timeout=60):
    """Async subprocess with timeout/cancellation teardown and wait/reap."""
    process = await acquire(
        lambda: asyncio.create_subprocess_exec(*args, cwd=cwd, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=(os.name == 'posix')),
        _stop_process,
    )
    reader = asyncio.create_task(process.communicate())
    async def drain():
        try:
            await asyncio.wait_for(asyncio.shield(reader), timeout=5)
        finally:
            if not reader.done():
                reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
    try:
        stdout, _ = await asyncio.wait_for(asyncio.shield(reader), timeout=max(1, timeout))
        return subprocess.CompletedProcess(args, process.returncode, stdout, None)
    finally:
        await cleanup(lambda: _stop_process(process), drain)


BACKGROUND_TASKS = ('_session_cleaner_task', '_cache_cleaner_task',
    '_daily_report_task', '_usage_scheduler_task', '_playwright_install_task', '_grading_index_task')


async def release_instance_resources(plugin):
    """Stop only this instance's background tasks; do not cancel user PDF jobs."""
    tasks = {getattr(plugin, name, None) for name in BACKGROUND_TASKS}
    tasks.discard(None);tasks.discard(asyncio.current_task())
    for task in tasks:
        if not task.done():task.cancel()
    async def join():
        if tasks:await asyncio.gather(*tasks, return_exceptions=True)
    browser, pw = getattr(plugin, '_browser', None), getattr(plugin, '_pw', None)
    async def close_browser():
        if browser is not None:await browser.close()
    async def stop_driver():
        if pw is not None:await pw.stop()
    try:
        await cleanup(join, close_browser, stop_driver, timeout=12)
    finally:
        for name in BACKGROUND_TASKS:setattr(plugin,name,None)
        plugin._browser=None;plugin._pw=None


async def retire_legacy_instances(current):
    """One-version handoff: old releases did not implement background teardown.

    Match exact coroutine source + data directory. Never scan/kill processes or
    touch another plugin. Deployment requires zero in-flight PDF/render jobs.
    """
    memory_path = Path(__file__).with_name('memory.py').resolve()
    owners = {}
    for task in asyncio.all_tasks():
        coroutine=task.get_coro();code=getattr(coroutine,'cr_code',None)
        frame=getattr(coroutine,'cr_frame',None)
        if code is None or frame is None or Path(code.co_filename).resolve()!=memory_path:continue
        if code.co_name not in ('_clean_expired_sessions_loop','_clean_cache_files_loop'):continue
        owner=frame.f_locals.get('self')
        if owner is None or owner is current or type(owner).__name__!=type(current).__name__:continue
        if getattr(owner,'DATA_DIR',None)!=getattr(current,'DATA_DIR',None):continue
        owners[id(owner)]=owner
    for owner in owners.values():
        # Do not interrupt a rendering operation belonging to the old instance.
        semaphore=getattr(owner,'_render_sema',None)
        limit=int(owner._cfg('render_concurrency',2) or 2)
        if semaphore is not None and semaphore._value < limit:
            logger.warning('[mathsolve cleanup] legacy instance busy; retained for manual review')
            continue
        # Preserve in-memory device preferences and follow-up context during handoff.
        old_globals=owner._ensure_browser.__func__.__globals__
        new_globals=current._ensure_browser.__func__.__globals__
        for key in ('USER_PREFERENCES','MATH_SESSION_STATE'):
            old,new=old_globals.get(key),new_globals.get(key)
            if isinstance(old,dict) and isinstance(new,dict) and old is not new:
                for k,v in old.items():new.setdefault(k,v)
        await release_instance_resources(owner)
    if owners:logger.info('[mathsolve cleanup] legacy handoff checked: %d',len(owners))
