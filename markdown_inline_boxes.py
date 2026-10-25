"""Fixed fog-blue text boxes; no model-authored CSS or active HTML."""
from bisect import bisect_right
import html
import re

STYLE_A = (
    'display:inline-block;max-width:100%;box-sizing:border-box;vertical-align:baseline;'
    'white-space:normal;overflow-wrap:anywhere;font-family:inherit;font-size:0.95em;'
    'font-weight:500;line-height:1.55;margin:2px 0;color:#35536f;'
    'background-color:#eef4fb;border:1px solid #cbdced;border-radius:7px;padding:2px 8px;'
)
MAX_BOXES = 512
MAX_BOX_CHARS = 60
# Only the exact exporter-owned span and text/entities, never nested HTML.
BOX_RE = re.compile(
    r'<span style="'+re.escape(STYLE_A)+r'">'
    r'(?P<text>(?:[^<>&\r\n]|&(?:amp|lt|gt|quot|apos|#[0-9]{1,7}|#x[0-9A-Fa-f]{1,6});)*)</span>'
)


def plain_text(text):
    # Numeric entities keep Markdown punctuation literal inside inline HTML.
    # No URLs, style values or attributes can be supplied by the model.
    escaped = html.escape(text, quote=True)
    escaped = re.sub(r'[\\`*_\[\]!|~]', lambda m: '&#'+str(ord(m[0]))+';', escaped)
    # A mis-styled paragraph must not turn into Markdown/HTML when unboxed.
    escaped = re.sub(r'(?m)^(\s*)([#>+\-]|\d+[.)])',
                     lambda m: m[1]+''.join('&#'+str(ord(c))+';' for c in m[2]), escaped)
    # Keep dialogue turns on separate lines when removing an erroneous code span.
    escaped = re.sub(r'(?<=\S)\n(?=[^\n])', '  \n', escaped)
    return escaped


def render_box(text):
    return '<span style="'+STYLE_A+'">'+plain_text(text)+'</span>'


def is_literal_code(text):
    """Keep recognizable code/math/paths literal, not a decorative text box."""
    return bool(
        re.search(r'[$\\<>`]|&(?:lt|gt);|(?:^|\s)--[a-z]', text)
        or re.match(r'^(?:(?:https?|file)://|[/~][\w.])', text)
        or re.search(r'\w\([^)]*\)|\w\s*(?:==|!=|:=|=(?!=))|[{}]', text)
        or re.search(r'(?m)^\s*(?:def|class|import|from|return|print|const|let|var|function|'
                     r'pip|python[\d.]*|npm|git|sudo|curl|ssh|SELECT|INSERT)\b', text)
        or re.fullmatch(r'[A-Za-z_]\w*(?:[./:]\w+)+', text)
        or re.fullmatch(r'[A-Za-z_]\w*_\w+', text)
        or (len(text) > MAX_BOX_CHARS and re.fullmatch(r'[A-Za-z0-9_.-]+', text))
        or ('\n' in text and all(re.fullmatch(r'[A-Za-z_]\w*\d\w*', line)
                                 for line in text.splitlines()))
    )


def is_box_label(text):
    """Conservative local-label filter; not a general sentence classifier."""
    text = text.strip()
    if (not text or len(text) > MAX_BOX_CHARS or '\n' in text or '\r' in text
            or is_literal_code(text)
            or len(re.findall(r'[\u3400-\u9fff]', text)) > 16
            or len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)) > 8):
        return False
    # Courtesy abbreviations are part of the user's original salutation labels.
    without_abbreviations = re.sub(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.', '', text, flags=re.I)
    if re.search(r'[.!?;:。！？；：…]', without_abbreviations):
        return False
    if re.match(r"(?i)^(?:I|we|you|he|she|they|it|this|that|there)\b\s+"
                r"(?:am|is|are|was|were|can|could|will|would|should|must|may|might|"
                r"have|has|had|do|does|did|need|want|hope|think|look|feel|agree|recommend)\b", text):
        return False
    if re.match(r'(?i)^(?:please|let us|let\x27s|thank you for)\b', text):
        return False
    if re.match(r'^(?:我|你|他|她|它|这|那|请|让|如果|因为|所以|因此|首先|然后|最后)', text):
        return False
    if re.search(r'可以|应该|必须|需要|意味着|表示|说明|都为|都是', text):
        return False
    return True


def inline_text(source):
    match = re.fullmatch(r'(?P<fence>`+)(?P<text>[\s\S]*?)(?P=fence)', source)
    if not match:
        return None
    text = match['text']
    if not text.strip():
        return None
    if text.startswith(' ') and text.endswith(' ') and text.strip():
        text = text[1:-1]  # CommonMark code-span padding convention.
    return None if is_literal_code(text) else text


def split_inline_code(text):
    """Yield (is_code, source), with matching backtick runs, in bounded time."""
    runs = list(re.finditer(r'`+', text))
    by_length = {}
    ends = {}
    for run in runs:
        by_length.setdefault(len(run[0]), []).append(run.start())
        ends[run.start()] = run.end()
    tokens = re.compile(
        r'(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$|(?<!\\)\$(?:\\.|[^$\n])+(?<!\\)\$'
        r'|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|(?P<ticks>`+)'
    )
    cursor = search_from = 0
    while (run := tokens.search(text, search_from)) is not None:
        search_from = run.end()
        if run['ticks'] is None:
            continue
        before = run.start() - 1
        while before >= 0 and text[before] == '\\':
            before -= 1
        if (run.start() - 1 - before) % 2:
            continue
        peers = by_length[len(run[0])]
        next_index = bisect_right(peers, run.start())
        if next_index == len(peers):
            continue
        closing_end = ends[peers[next_index]]
        yield False, text[cursor:run.start()]
        yield True, text[run.start():closing_end]
        cursor = search_from = closing_end
    yield False, text[cursor:]


def boxes_for_model(reference):
    # Keep historical text legible, without copying exporter styles into prompts.
    def replace(match):
        text = html.unescape(match['text'])
        if not is_box_label(text):
            return plain_text(text)
        length = max((len(m[0]) for m in re.finditer(r'`+', text)), default=0) + 1
        fence = '`' * length
        return fence+' '+text+' '+fence
    return BOX_RE.sub(replace, reference)
