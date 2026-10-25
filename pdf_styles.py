"""Snapshot each user's fixed/random PDF theme once, not per compile or box."""
from contextvars import ContextVar
from functools import wraps
import inspect
import secrets
try:
    from .pdf_theme_preferences import choose_style
    from .pdf_theme_designs import new_box_preamble
except ImportError:
    from pdf_theme_preferences import choose_style
    from pdf_theme_designs import new_box_preamble


# Preserve the old default pool for users without a personal preference.
# Retain legacy F/C definitions for already-selected/in-flight documents only.
STYLE_IDS = ('original', 'a')
_current_style = ContextVar('mathsolve_pdf_box_style', default='original')


def current_style():
    return _current_style.get()


def randomized_pdf_style(func):
    """Snapshot the user's preference once; isolate concurrent jobs and repairs."""
    signature = inspect.signature(func)
    @wraps(func)
    async def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        plugin, event = bound.arguments.get("self"), bound.arguments.get("event")
        if plugin is not None and hasattr(plugin, "_pdf_theme_choice"):
            # A storage error must not silently render a fixed-theme request randomly.
            choice = await plugin._pdf_theme_choice(event)
            style = choose_style(choice)
        else:
            style = secrets.choice(STYLE_IDS)
        token = _current_style.set(style)
        try:
            return await func(*args, **kwargs)
        finally:
            _current_style.reset(token)
    return wrapped


_ORIGINAL = (
    r'\tcbset{enhanced, boxrule=0.8pt, arc=2.5mm}',
    r'\newtcolorbox{problemBox}[1][]{colback=blue!3!white, colframe=blue!65!black, title=题目, fonttitle=\bfseries, #1}',
    r'\newtcolorbox{theoremBox}[2][]{colback=teal!3!white, colframe=teal!65!black, title=定理：#2, fonttitle=\bfseries, #1}',
    r'\newtcolorbox{lemmaBox}[2][]{colback=orange!5!white, colframe=orange!80!black, title=引理：#2, fonttitle=\bfseries, #1}',
    r'\newtcolorbox{formulaBox}[2][]{colback=violet!3!white, colframe=violet!65!black, title=公式：#2, fonttitle=\bfseries, #1}',
)

_PALETTES = {
    'a': ('385B78', '637888'),
    'f': ('3E665F', '71847C'),
    'c': ('74618F', '887990'),
}
_PASTEL_TINTS = {
    'problemBox': ('msProblemTint', 'EEF5FC'),
    'theoremBox': ('msTheoremTint', 'EFF8F2'),
    'lemmaBox': ('msLemmaTint', 'FFF5EA'),
    'formulaBox': ('msFormulaTint', 'F4F0FA'),
}
_TITLE_TINTS = {
    'problemBox': ('msProblemTitleTint', 'DDE9F6'),
    'theoremBox': ('msTheoremTitleTint', 'DDEEE3'),
    'lemmaBox': ('msLemmaTitleTint', 'FCE8D4'),
    'formulaBox': ('msFormulaTitleTint', 'E8DEF3'),
}
_SHAPES = {
    'a': (
        r'arc=2mm,boxrule=1.05pt,colframe=#1!65!white,colbacktitle=white,coltitle=#1,'
        r'titlerule=.65pt,toptitle=1.6mm,bottomtitle=1.6mm,fonttitle=\rmfamily\bfseries'
    ),
    'f': (
        r'arc=.8mm,boxrule=.55pt,colframe=#1!65!white,'
        r'borderline={.25pt}{-1.3pt}{#1!35!white},coltitle=#1,'
        r'attach boxed title to top center={yshift=-2mm},'
        r'boxed title style={frame hidden,colback=white,boxrule=0pt,'
        r'left=3mm,right=3mm,top=.8mm,bottom=.8mm},'
        r'left=3.8mm,right=3.8mm,top=5mm,bottom=3mm,before skip=5mm,after skip=3mm'
    ),
    'c': (
        r'arc=2.6mm,boxrule=.85pt,colframe=#1!55!white,coltitle=#1,'
        r'borderline={.4pt}{-1.6pt}{#1!40!white},'
        r'attach boxed title to top left={xshift=3.8mm,yshift=-2mm},'
        r'boxed title style={colback=#1!9!white,colframe=#1!55!white,'
        r'arc=1.4mm,boxrule=.75pt,left=2.4mm,right=2.4mm,top=1mm,bottom=1mm},'
        r'top=4.1mm,before skip=4.5mm'
    ),
}


def box_preamble(style=None):
    """Keep legacy environment signatures and optional title/options overrides."""
    style = current_style() if style is None else style
    if style == 'original':
        return list(_ORIGINAL)
    new = new_box_preamble(style)
    if new is not None:
        return [_ORIGINAL[0], *new]
    if style not in _PALETTES:
        raise ValueError('Unknown PDF box style: ' + str(style))
    primary, secondary = _PALETTES[style]
    lines = [
        # Leave raw warning tcolorboxes unaffected by the named answer themes.
        _ORIGINAL[0],
        rf'% mathsolve-pdf-box-style: {style}',
        r'\definecolor{msBoxInk}{HTML}{202B36}',
        rf'\definecolor{{msBoxPrimary}}{{HTML}}{{{primary}}}',
        rf'\definecolor{{msBoxSecondary}}{{HTML}}{{{secondary}}}',
        r'\tcbset{msAnswerBase/.style={enhanced,breakable,colback=white,coltext=msBoxInk,'
        r'boxsep=0pt,left=3.4mm,right=3.4mm,top=2.5mm,bottom=2.5mm,'
        r'before skip=3mm,after skip=2mm,fonttitle=\sffamily\bfseries,boxrule=.65pt}}',
        r'\tcbset{msAnswerShape/.style={' + _SHAPES[style] + '}}',
    ]
    if style in ('a', 'f'):
        lines.extend(rf'\definecolor{{{name}}}{{HTML}}{{{hex_color}}}'
                     for name, hex_color in _PASTEL_TINTS.values())
    if style == 'a':
        # Distinct soft pastel headers; body fills remain very pale and unchanged.
        lines.extend(rf'\definecolor{{{name}}}{{HTML}}{{{hex_color}}}'
                     for name, hex_color in _TITLE_TINTS.values())
    for env, title, color, args in (
        ('problemBox', '题目', 'msBoxPrimary', '[1][]'),
        ('theoremBox', '定理：#2', 'msBoxSecondary', '[2][]'),
        ('lemmaBox', '引理：#2', 'msBoxSecondary', '[2][]'),
        ('formulaBox', '公式：#2', 'msBoxPrimary', '[2][]'),
    ):
        # A has pastel headers; F keeps its white floating title unchanged.
        # Keep #1 last so existing caller-provided options still take priority.
        tint_options = ''
        if style == 'a':
            tint = _PASTEL_TINTS[env][0]
            title_tint = _TITLE_TINTS[env][0]
            tint_options = ',colbacktitle=' + title_tint + ',colback=' + tint + '!25!white'
        elif style == 'f':
            tint = _PASTEL_TINTS[env][0]
            tint_options = ',colback=' + tint + '!40!white'
        lines.append(r'\newtcolorbox{' + env + '}' + args
                     + '{msAnswerBase,msAnswerShape=' + color
                     + ',title={' + title + '}' + tint_options + ',#1}')
    return lines
