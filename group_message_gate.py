"""Persistent, per-platform/group receive switch owned by mathsolve.

AstrBot's ordinary message handlers run AFTER command/permission checks and
media preprocessing. Guard the first pipeline stage instead, while retaining
PipelineScheduler.execute's event registration and temporary-file cleanup.
The runtime wrapper is reversible; no AstrBot core files are modified.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json


GROUP_GATE_KEY = "group_receive_switch_v1"
CONTROL_COMMANDS = frozenset(("elmute", "elstart"))


def group_key(event) -> str | None:
    """Do not use the user-isolated UMO: all members share one group switch."""
    if event.is_private_chat():
        return None
    group_id = str(event.get_group_id() or "").strip()
    platform_id = str(event.get_platform_id() or "").strip()
    if not group_id or not platform_id:
        return None
    return json.dumps([platform_id, str(event.get_self_id() or ""), group_id],
                      ensure_ascii=False, separators=(",", ":"))


def control_command(event, config) -> str | None:
    """Recognize only the current message, never quoted text or command substrings."""
    for comp in event.get_messages() or []:
        if comp.__class__.__name__ == "At":
            if str(getattr(comp, "qq", "")) not in (str(event.get_self_id()), "all"):
                return None  # A command addressed to another bot is not ours.
    text = str(event.message_str or "").strip()
    candidates = [text]
    for prefix in config.get("wake_prefix", ["/"]):
        if prefix and text.startswith(prefix):
            candidates.append(text[len(prefix):].strip())
    for candidate in candidates:
        if candidate.startswith("/"):
            candidate = candidate[1:]
        if candidate in CONTROL_COMMANDS:
            return candidate
    return None


def is_bot_admin(event, config) -> bool:
    # Before WakingCheckStage the event.role has not yet been populated.
    # Apply the same configured bot-admin IDs and API-key restriction.
    if event.get_extra("_api_key_allow_admin_role") is False:
        return False
    return str(event.get_sender_id()) in {str(uid) for uid in config.get("admins_id", [])}


class GroupMessageGate:
    def __init__(self, read, write):
        self._read = read
        self._write = write
        self._lock = asyncio.Lock()
        self._record = {"version": 1, "muted_groups": []}
        self._muted: set[str] = set()
        self._installed = False
        self._scheduler = self._previous = self._wrapper = None

    async def load(self):
        async with self._lock:
            record = await self._read(GROUP_GATE_KEY, None)
            if record is None:
                record = {"version": 1, "muted_groups": []}
            if not isinstance(record, dict) or record.get("version") != 1:
                raise ValueError("群消息开关存储格式无效，未覆盖原数据")
            muted = record.get("muted_groups")
            if not isinstance(muted, list) or not all(isinstance(k, str) for k in muted):
                raise ValueError("群消息开关列表无效，未覆盖原数据")
            self._record = dict(record)
            self._muted = set(muted)

    def is_muted(self, event) -> bool:
        key = group_key(event)
        return key is not None and key in self._muted

    async def set_muted(self, event, muted: bool) -> bool:
        key = group_key(event)
        if key is None:
            raise ValueError("请在需要设置的群聊中使用此指令。")
        async with self._lock:
            changed = (key in self._muted) != muted
            if not changed:
                return False
            updated = set(self._muted)
            (updated.add if muted else updated.discard)(key)
            record = {**self._record, "muted_groups": sorted(updated)}
            # Commit before acknowledging; failed writes leave the old policy intact.
            await self._write(GROUP_GATE_KEY, record)
            self._record = record
            self._muted = updated
            return True

    def should_block(self, event, config) -> bool:
        if not self.is_muted(event):
            return False
        return not (control_command(event, config) and is_bot_admin(event, config))

    def install(self, scheduler_cls=None):
        if scheduler_cls is None:
            from astrbot.core.pipeline.scheduler import PipelineScheduler
            scheduler_cls = PipelineScheduler
        if self._installed:
            return
        previous = scheduler_cls._process_stages
        signature = inspect.signature(previous)
        if "from_stage" not in signature.parameters:
            raise RuntimeError("当前 AstrBot 消息流水线不兼容群静默开关")
        prior_gate = getattr(previous, "_mathsolve_group_gate", None)
        if prior_gate is not None:
            prior_gate.uninstall()
            previous = scheduler_cls._process_stages

        @functools.wraps(previous)
        async def guarded(scheduler, event, from_stage=0):
            # Re-entrant downstream stages of an accepted/in-flight request
            # are left alone. Only NEW group events are subject to this switch.
            if self._installed and from_stage == 0:
                if self.should_block(event, scheduler.ctx.astrbot_config):
                    event.stop_event()
                    return
            return await previous(scheduler, event, from_stage)

        guarded._mathsolve_group_gate = self
        self._scheduler, self._previous, self._wrapper = scheduler_cls, previous, guarded
        self._installed = True
        scheduler_cls._process_stages = guarded

    def uninstall(self):
        self._installed = False
        # Never undo a wrapper installed by some other plugin after ours.
        # If wrapped by another owner, our now-inactive wrapper just delegates.
        if self._scheduler is not None and self._scheduler._process_stages is self._wrapper:
            self._scheduler._process_stages = self._previous
