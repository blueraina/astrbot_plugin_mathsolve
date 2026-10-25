# -*- coding: utf-8 -*-
"""Font selection helpers for XeLaTeX PDF output."""

import re
from typing import List, Sequence


_JAPANESE_SCRIPT_RE = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f]"
)

_SERIF_FONT_CANDIDATES = {
    "jp": (
        "Noto Serif CJK JP",
        "Noto Serif JP",
        "Source Han Serif JP",
        "Harano Aji Mincho",
        "IPAexMincho",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Source Han Sans JP",
    ),
    "sc": (
        "Noto Serif CJK SC",
        "Noto Serif SC",
        "Source Han Serif SC",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "FandolSong-Regular.otf",
    ),
}

_SANS_FONT_CANDIDATES = {
    "jp": (
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Source Han Sans JP",
        "Harano Aji Gothic",
        "IPAexGothic",
    ),
    "sc": (
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "FandolHei-Regular.otf",
    ),
}


def contains_japanese_script(*texts: str) -> bool:
    """Return True when text contains Japanese kana or related symbols."""
    return any(
        _JAPANESE_SCRIPT_RE.search(str(text or "")) is not None
        for text in texts
    )


def _ordered_font_candidates(
    groups: dict,
    prefer_japanese: bool,
) -> Sequence[str]:
    primary = "jp" if prefer_japanese else "sc"
    secondary = "sc" if prefer_japanese else "jp"
    return tuple(groups[primary]) + tuple(groups[secondary])


def _xelatex_font_switch(command: str, candidates: Sequence[str]) -> str:
    """Build nested font-existence checks without breaking CTEX fallback."""
    fallback = ""
    for font_name in reversed(tuple(candidates)):
        fallback = (
            rf"\IfFontExistsTF{{{font_name}}}"
            rf"{{{command}{{{font_name}}}}}"
            rf"{{{fallback}}}"
        )
    return fallback


def build_pdf_cjk_font_preamble(*texts: str) -> List[str]:
    """Build a Japanese-aware CJK font preamble for XeLaTeX.

    Noto/Source Han/Harano Aji fonts are selected only when installed. If none
    are available, the final empty branch keeps CTEX's existing platform
    default instead of making compilation fail.
    """
    prefer_japanese = contains_japanese_script(*texts)
    secondary = "sc" if prefer_japanese else "jp"
    serif_candidates = _ordered_font_candidates(
        _SERIF_FONT_CANDIDATES,
        prefer_japanese,
    )
    sans_candidates = _ordered_font_candidates(
        _SANS_FONT_CANDIDATES,
        prefer_japanese,
    )
    serif_fallback_candidates = _SERIF_FONT_CANDIDATES[secondary]
    sans_fallback_candidates = _SANS_FONT_CANDIDATES[secondary]
    return [
        r"\usepackage{fontspec}",
        r"\usepackage{xeCJK}",
        r"\xeCJKsetup{AutoFallBack=true}",
        _xelatex_font_switch(r"\setCJKmainfont", serif_candidates),
        _xelatex_font_switch(
            r"\setCJKfallbackfamilyfont{\CJKrmdefault}",
            serif_fallback_candidates,
        ),
        _xelatex_font_switch(r"\setCJKsansfont", sans_candidates),
        _xelatex_font_switch(
            r"\setCJKfallbackfamilyfont{\CJKsfdefault}",
            sans_fallback_candidates,
        ),
    ]
