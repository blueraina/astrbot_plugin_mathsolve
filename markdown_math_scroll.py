"""Exporter-owned C scroll containers for real TeX, with no active HTML."""
import re

OUTER_STYLE = (
    'display:block;position:relative;box-sizing:border-box;width:100%;max-width:100%;'
    'min-width:0;overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch;'
    'padding:12px 0;margin:12px 0;'
)
INNER_STYLE = 'display:inline-block;min-width:100%;box-sizing:border-box;white-space:nowrap;padding:8px;'
PREFIX = '<span style="'+OUTER_STYLE+'"><span style="'+INNER_STYLE+'">$\\displaystyle\\begin{gathered}'
SUFFIX = r'\end{gathered}$</span></span>'
SCROLL_RE = re.compile(re.escape(PREFIX)+r'(?P<tex>[\s\S]*?)'+re.escape(SUFFIX))
MATH_RE = re.compile(r'(?<!\\)\$\$(?P<display>[\s\S]*?)(?<!\\)\$\$'
                     r'|(?<!\\)\$(?P<inline>(?:\\.|[^$\n])+?)(?<!\\)\$')
MAX_SCROLL_FORMULAS = 512


def compact_tex(source):
    """Remove TeX comments before flattening lines; retain explicit TeX \\ rows."""
    lines = []
    for line in source.strip().splitlines():
        comment = None
        for match in re.finditer('%', line):
            before = match.start()-1
            while before >= 0 and line[before] == '\\':
                before -= 1
            if (match.start()-1-before) % 2 == 0:
                comment = match.start()
                break
        # A TeX comment consumes its newline, whereas an ordinary newline is a
        # space. Keep that distinction for commands and text split across lines.
        lines.append(line[:comment] if comment is not None else line+' ')
    tex = ''.join(lines).strip()
    if not tex or '\x00' in tex:
        raise ValueError('Empty or invalid scroll formula')
    if re.search(r'\\(?:href|url|includegraphics|require|html\w*|cssId|class|style)\b', tex):
        raise ValueError('Active markup or external resources are not allowed in formulas')
    # Raw tag delimiters cannot survive into an HTML container. TeX relational
    # operators preserve the mathematical meaning without HTML interpretation.
    tex = tex.replace('<', r'\lt ').replace('>', r'\gt ')
    # Inline math does not support \tag. Preserve simple visible labels as text.
    tex = re.sub(r'\\tag(\*)?\{([^{}\\\n]{1,80})\}',
                 lambda m: r'\qquad\text{'+(m[2] if m[1] else '('+m[2]+')')+'}', tex)
    if re.search(r'\\(?:tag|label)\b', tex):
        raise ValueError('Put complex equation labels in prose outside the formula')
    # A raw $ would terminate the fixed inline-math wrapper. Escaped dollars are
    # literal TeX; nested math inside \text must instead use standard TeX syntax.
    if re.search(r'(?<!\\)\$', tex):
        raise ValueError('Nested dollar delimiter in scroll formula')
    return tex


def render_scroll_formula(tex):
    return PREFIX+compact_tex(tex)+SUFFIX


def long_inline_formula(tex):
    """Conservative width proxy, not a pixel measurement or algebra rewrite."""
    if re.search(r'\\begin\{(?:[bBpPvV]?matrix|smallmatrix|cases|aligned|array|gathered)\}', tex):
        return True
    visible = re.sub(r'\\[A-Za-z]+\s*', 'x', tex)
    visible = re.sub(r'[\s{}^_]', '', visible)
    return len(visible) >= 48


def scroll_formulas_for_model(reference):
    """References carry TeX rather than our layout, preventing nested wrappers."""
    return SCROLL_RE.sub(lambda m: '$$'+compact_tex(m['tex'])+'$$', reference)
