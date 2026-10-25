"""Restore /md text boundaries before AstrBot's standard command filter."""
import re

from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.core.star.filter.custom_filter import CustomFilter


class MarkdownMixedMessageFilter(CustomFilter):
    def filter(self, event, cfg):
        # This filter only normalizes; the following CommandFilter still owns
        # command matching, permissions, aliases and normal wake requirements.
        if not getattr(event, 'is_at_or_wake_command', False):
            return True
        parts = getattr(getattr(event, 'message_obj', None), 'message', None)
        if not isinstance(parts, (list, tuple)) or len(parts) > 512:
            return True
        flat, spaced = [], []
        has_image = False
        for part in parts:
            if isinstance(part, Plain):
                flat.append(part.text)
                spaced.append(part.text)
            elif isinstance(part, Image):
                has_image = True
                spaced.append('\n')
            elif isinstance(part, Reply):
                # Quoted text/images are data, never another command source.
                continue
            elif isinstance(part, At) and str(part.qq) == str(event.get_self_id()):
                continue
            else:
                return True
        if not has_image:
            return True
        old = str(event.get_message_str() or '').strip()
        collapsed = ''.join(flat).strip()
        restored = ''.join(spaced).strip()
        # WakingCheckStage may already have removed the configured prefix.
        # Only modify the current message, not text rewritten by another plugin.
        if old != collapsed:
            for prefix in cfg.get('wake_prefix', []):
                if (prefix and collapsed.startswith(prefix)
                        and collapsed[len(prefix):].strip() == old):
                    if not restored.startswith(prefix):
                        return True
                    restored = restored[len(prefix):].strip()
                    break
            else:
                return True
        if re.fullmatch(r'md(?:\s+[\s\S]*)?', restored) and restored != old:
            event.message_str = restored
        return True
