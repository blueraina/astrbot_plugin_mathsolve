"""Conservative structural scope for decorative labels, never ordinary prose."""
import html
import re
from bisect import bisect_right

try:
    from .markdown_inline_boxes import BOX_RE, inline_text, split_inline_code, plain_text, is_box_label
except ImportError:
    from markdown_inline_boxes import BOX_RE, inline_text, split_inline_code, plain_text, is_box_label

REFERENCE = re.compile(r'称呼|落款|词汇|术语|缩写|关键词|短语|固定表达|salutation|sign.off|vocab|glossary|\bterms?\b|abbrev|key.?words?|phrases?', re.I)
DOCUMENT = re.compile(r'完整|全文|整封|正文|范文|邮件示例|示例邮件|邮件模板|对话|段落|主题行|subject.?line|full email|complete|sample email|email example|dialogue|conversation', re.I)
COLUMN = re.compile(r'^(?:推荐|常用|英文|中文|英语|商务|专业|正式|日常|邮件|英文邮件|参考|核心|关键|English|Chinese|recommended|common|business|formal|informal|email|key|\s)*'
                    r'(?:称呼|落款|词汇|术语|缩写|关键词|短语|固定表达|表达|单词|salutations?|sign.offs?|vocabulary|terms?|abbreviations?|keywords?|phrases?|expressions?)\s*$', re.I)
LOCAL = re.compile(r'^(?:称呼|术语|缩写|词汇|关键词|固定短语|salutation|term|abbreviation|keyword)\s*[:：]?\s*$', re.I)
SALUTATION = re.compile(r'^(?:Dear\b|Hi\b|Hello\b|Good (?:morning|afternoon|evening)\b|To whom it may concern\b|(?:Best|Kind|Warm) regards\b|Sincerely\b|Respectfully\b)', re.I)


def cells(line):
    """Pipe table cell ranges, ignoring escaped pipes and code-span pipes."""
    masked = list(line)
    offset = 0
    for code, piece in split_inline_code(line):
        if code:
            masked[offset:offset+len(piece)] = ' ' * len(piece)
        offset += len(piece)
    boundaries = [m.start() for m in re.finditer(r'(?<!\\)\|', ''.join(masked))]
    if not boundaries:
        return []
    ranges = list(zip([-1]+boundaries, boundaries+[len(line)]))
    result = [(a+1, b) for a, b in ranges]
    if not line[:boundaries[0]].strip():
        result.pop(0)
    if not line[boundaries[-1]+1:].strip():
        result.pop()
    return result


def leading(prefix, classified=False):
    """Only a primary label or its alternatives, not a term inside explanation."""
    prefix = BOX_RE.sub(lambda m: '' if is_box_label(html.unescape(m['text'])) else plain_text(html.unescape(m['text'])), prefix)
    prefix = ''.join('' if code and inline_text(piece) is not None and is_box_label(inline_text(piece))
                     else piece for code, piece in split_inline_code(prefix))
    prefix = re.sub(r'<br[ \t]*/?>|&lt;br[ \t]*/?&gt;', ' / ', prefix, flags=re.I)
    prefix = re.sub(r'\*{1,2}|__', '', prefix).strip()
    if classified and len(prefix) <= 40 and re.fullmatch(r'[^.!?。！？\n]+[:：]', prefix):
        return True
    return not re.sub(r'\bor\b|\band\b|[/、,，;；\s]', '', prefix, flags=re.I)


class BoxContext:
    def __init__(self):
        self.headings = []
        self.email = False

    def scope(self, source):
        lines = source.splitlines(keepends=True)
        starts, rules = [], []
        offset = 0
        table = None
        for index, line in enumerate(lines):
            starts.append(offset)
            offset += len(line)
            rule = None
            heading = re.match(r'^ {0,3}(#{1,6})\s+(.+)', line)
            if heading:
                level = len(heading[1])
                self.headings = [(n, t) for n, t in self.headings if n < level]
                self.headings.append((level, heading[2]))
                self.email = False
                table = None
            denied = self.email or any(DOCUMENT.search(t) for _, t in self.headings)
            reference = any(REFERENCE.search(t) for _, t in self.headings)
            # Complete unquoted letters need the same protection as blockquotes.
            if re.match(r'^\s*(?:\*\*)?(?:Subject|主题)\s*[:：]', line, re.I):
                self.email = denied = True
            spans = cells(line)
            next_line = lines[index+1] if index+1 < len(lines) else ''
            next_cells = cells(next_line)
            if spans and len(spans) == len(next_cells) and all(
                    re.fullmatch(r'\s*:?-{3,}:?\s*', next_line[a:b]) for a, b in next_cells):
                table = [bool(COLUMN.fullmatch(re.sub(r'[*_`]', '', line[a:b]).strip())) for a, b in spans]
            elif table is not None and not spans:
                table = None
            elif not heading and not denied and not re.match(r'^\s*>', line):
                if table is not None and len(table) == len(spans):
                    rule = ('table', [(a, b) for (a, b), allowed in zip(spans, table) if allowed])
                else:
                    item = re.match(r'^\s*(?:[-+*]|\d+[.)])\s+', line)
                    rule = ('list', item.end(), reference) if item else ('local', reference)
            rules.append(rule)

        def allowed(position, label):
            row = bisect_right(starts, position)-1
            if row < 0 or not rules[row]:
                return False
            column = position-starts[row]
            line, rule = lines[row], rules[row]
            if rule[0] == 'table':
                return any(a <= column < b and leading(line[a:column]) for a, b in rule[1])
            if rule[0] == 'list':
                prefix = line[rule[1]:column]
                return bool((rule[2] or LOCAL.fullmatch(prefix.strip()) or SALUTATION.match(label))
                            and leading(prefix, classified=True))
            return bool(LOCAL.fullmatch(line[:column].strip()) or (rule[1] and leading(line[:column])))

        result, offset = [], 0
        for code, piece in split_inline_code(source):
            if code:
                label = inline_text(piece)
                result.append(piece if label is None or allowed(offset, label) else plain_text(label))
            else:
                def replace(match):
                    label = html.unescape(match['text'])
                    # Leave invalid canonical content for the existing validator.
                    if re.search(r'[$\\<>]', label) or allowed(offset+match.start(), label):
                        return match[0]
                    return plain_text(label)
                result.append(BOX_RE.sub(replace, piece))
            offset += len(piece)
        return ''.join(result)
