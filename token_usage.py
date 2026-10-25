"""SPDF-local API-reported usage. Never retain prompts, responses or credentials."""
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from io import BytesIO
import json
import logging
from pathlib import Path
import threading
from xml.sax.saxutils import escape

_current = ContextVar('mathsolve_spdf_tokens', default=None)
_render_lock = threading.Lock()
_log = logging.getLogger('astrbot')
WARNING_KEY = 'md2img_spdf_token_warning'


def field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def number(value):
    if type(value) is int and 0 <= value <= 10**12:
        return value
    # A few compatible gateways serialize integer counters as strings.
    if isinstance(value, str) and value.isdigit() and len(value) <= 13:
        n = int(value)
        return n if n <= 10**12 else None
    return None


@dataclass(frozen=True)
class Usage:
    input: object = None
    output: object = None
    total: object = None
    note: str = ''


def extract_usage(response):
    raw = field(response, 'raw_completion')
    usage = field(raw, 'usage')
    if usage is None:
        choices = field(raw, 'choices', []) or []
        if choices:
            usage = field(choices[0], 'usage')
    # Prefer the actual API fields to adapter defaults (missing usage can be
    # converted to an all-zero TokenUsage by AstrBot).
    if usage is not None:
        inp = number(field(usage, 'prompt_tokens', field(usage, 'input_tokens')))
        out = number(field(usage, 'completion_tokens', field(usage, 'output_tokens')))
        total = number(field(usage, 'total_tokens'))
        # Native Anthropic input_tokens excludes cache creation/read tokens.
        if field(raw, 'type') == 'message' and inp is not None:
            for key in ('cache_creation_input_tokens', 'cache_read_input_tokens'):
                value = field(usage, key)
                if value is not None:
                    value = number(value)
                    if value is None:
                        inp = None; break
                    inp += value
        if inp is not None or out is not None or total is not None:
            note = ''
            if total is None and inp is not None and out is not None:
                total = inp + out
            elif total is not None and inp is not None and out is not None and total != inp+out:
                note = '接口总计与输入/输出分项不一致，合计保留接口值'
            # Cached/reasoning details are breakdowns, never added again.
            return Usage(inp, out, total, note)
        return Usage(note='接口未提供可用的 token 计数')
    # Native Gemini: prefer raw totals over normalized adapter cache arithmetic.
    native = field(raw, 'usage_metadata')
    if native is not None:
        inp = number(field(native, 'prompt_token_count'))
        total = number(field(native, 'total_token_count'))
        # Native totalTokenCount is prompt + thoughts + candidates. Cached and
        # tool-use breakdowns are not added to promptTokenCount again.
        out = total-inp if total is not None and inp is not None and total >= inp else None
        return Usage(inp, out, total)
    normalized = field(response, 'usage')
    other = number(field(normalized, 'input_other'))
    cached = number(field(normalized, 'input_cached'))
    out = number(field(normalized, 'output'))
    inp = other+cached if other is not None and cached is not None else None
    # An all-zero adapter placeholder is not proof of a free invocation.
    if not any(x is not None and x > 0 for x in (inp, out)):
        return Usage()
    return Usage(inp, out, inp+out if inp is not None and out is not None else None)


class Ledger:
    def __init__(self):
        self.calls = []

    def begin(self, model):
        call = {'model': str(model or '未知模型'), 'usage': Usage(), 'state': 'pending'}
        self.calls.append(call)
        return call

    def snapshot(self):
        rows = {}
        for call in self.calls:
            name = call['model']
            row = rows.setdefault(name, {'model': name, 'calls': 0, 'failed': 0, 'missing': 0,
                                         'sums': dict(input=0, output=0, total=0),
                                         'known': dict(input=0, output=0, total=0), 'notes': []})
            row['calls'] += 1
            row['failed'] += call['state'] in ('failed', 'cancelled')
            u = call['usage']
            row['missing'] += any(getattr(u, key) is None for key in row['sums'])
            for key in row['sums']:
                value = getattr(u, key)
                if value is not None:
                    row['known'][key] += 1; row['sums'][key] += value
            if u.note and u.note not in row['notes']: row['notes'].append(u.note)
        rows = sorted(rows.values(), key=lambda row: (-row['sums']['total'], row['model']))
        total = {'model': '总计', 'calls': sum(r['calls'] for r in rows),
                 'failed': sum(r['failed'] for r in rows), 'missing': sum(r['missing'] for r in rows),
                 'sums': {k: sum(r['sums'][k] for r in rows) for k in ('input', 'output', 'total')},
                 'known': {k: sum(r['known'][k] for r in rows) for k in ('input', 'output', 'total')}}
        complete = bool(rows) and not total['missing']
        total['model'] = '总计' if complete else '已知用量合计'
        return {'version': 1, 'scope': 'single_spdf', 'complete': complete, 'rows': rows, 'total': total}


def tracked_spdf_tokens(fn):
    @wraps(fn)
    async def run(*args, **kwargs):
        token = _current.set(Ledger())
        try:
            return await fn(*args, **kwargs)
        finally:
            _current.reset(token)
    return run


async def generate_with_token_usage(owner, **kwargs):
    ledger = _current.get()
    if ledger is None:
        return await owner.context.llm_generate(**kwargs)
    provider_id = kwargs.get('chat_provider_id', '')
    model = provider_id
    try:
        provider = owner.context.get_provider_by_id(provider_id)
        model = provider.get_model() or provider_id
    except Exception:
        pass
    call = ledger.begin(model)
    try:
        response = await owner.context.llm_generate(**kwargs)
    except asyncio.CancelledError:
        call['state'] = 'cancelled'
        raise
    except Exception:
        call['state'] = 'failed'
        raise
    call['state'] = 'returned'
    try:
        actual = field(field(response, 'raw_completion'), 'model')
        if isinstance(actual, str) and actual.strip(): call['model'] = actual.strip()
        call['usage'] = extract_usage(response)
    except Exception as exc:
        # Telemetry must not turn a successful answer into a failed answer.
        _log.warning('[mathsolve] token usage extraction failed: %s', type(exc).__name__)
    return response


def cell(row, key):
    known = row['known'][key]
    if not known: return '未返回'
    return f"{row['sums'][key]:,}" + ('*' if known < row['calls'] else '')


def _usage_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font = 'DeepthinkUsageCJK'
    if font not in pdfmetrics.getRegisteredFontNames():
        fonts = ('/usr/local/share/fonts/windows/msyh.ttc',
                 '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                 '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
        for path in fonts:
            if Path(path).is_file():
                try:
                    pdfmetrics.registerFont(TTFont(font, path)); break
                except Exception: continue
        else:
            raise RuntimeError('No supported embedded Chinese font for token report')
    return font


def render_report(snapshot, pagesize):
    """Return a small footer overlay and its top edge; never paginate usage."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Paragraph
    font = _usage_font()
    width, height = pagesize
    margin = min(44, width * 0.075)
    usable = width - 2 * margin
    columns = 3 if len(snapshot['rows']) > 8 else 2
    gap = 12
    column_width = (usable - gap * (columns - 1)) / columns
    total = snapshot['total']
    heading = (f"Deepthink 模型用量 | {total['model']} {cell(total, 'total')} tokens"
               f"（输入 {cell(total, 'input')} / 输出 {cell(total, 'output')}）"
               f" | {total['calls']} 次调用")
    rows = [(f"{row['model']}：{cell(row, 'total')}"
             f"（输入 {cell(row, 'input')} / 输出 {cell(row, 'output')}；{row['calls']} 次）")
            for row in snapshot['rows']]
    notes = []
    if not snapshot['complete']:
        notes.append(f"{total['missing']} 次用量未返回或不完整；* 为已知部分，未知不按 0 计。")
    if total['failed']:
        notes.append(f"{total['failed']} 次失败或取消。")
    for row in snapshot['rows']:
        for note in row['notes']:
            if note not in notes:
                notes.append(note)
    for font_size in (7, 6.5, 6):
        style = ParagraphStyle('usage-footer', fontName=font, fontSize=font_size,
                               leading=font_size * 1.3, textColor=colors.HexColor('#606873'),
                               wordWrap='CJK', splitLongWords=True, spaceAfter=0)
        def paragraph(text, available):
            p = Paragraph(escape(str(text)), style)
            _, h = p.wrap(available, height)
            return p, h
        head, head_h = paragraph(heading, usable)
        note, note_h = paragraph(' '.join(notes), usable)
        # Removing the boilerplate must not move the existing usage rows.
        note_h = max(note_h, style.leading)
        groups = []
        for start in range(0, len(rows), columns):
            group = [paragraph(text, column_width) for text in rows[start:start + columns]]
            groups.append((group, max(h for _, h in group)))
        block_height = head_h + 3 + sum(h + 1 for _, h in groups) + 2 + note_h
        # Keep genuinely exceptional model lists bounded and legible. On failure
        # the caller returns the original answer with an explicit warning.
        if block_height <= min(240, height * 0.30):
            break
    else:
        raise ValueError('Token usage is too large for a legible last-page footer')
    bottom = 12
    top = bottom + block_height
    out = BytesIO()
    canvas = Canvas(out, pagesize=pagesize)
    y = top - head_h
    head.drawOn(canvas, margin, y)
    y -= 3
    for group, row_h in groups:
        for index, (p, h) in enumerate(group):
            p.drawOn(canvas, margin + index * (column_width + gap), y - h)
        y -= row_h + 1
    note.drawOn(canvas, margin, bottom)
    canvas.showPage()
    canvas.save()
    return out.getvalue(), top


def _reserve_footer(page, pdf_bytes, footer_top):
    """Leave existing pages alone; only move/scale last-page ink if necessary."""
    import pymupdf
    from pypdf import Transformation
    from pypdf.generic import ArrayObject, FloatObject, NameObject, RectangleObject
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    with pymupdf.open(stream=pdf_bytes, filetype='pdf') as doc:
        boxes = [box for kind, box, *_ in doc[-1].get_bboxlog() if kind != 'ignore-text']
    if not boxes:
        return
    ink_top = max(0, min(box[1] for box in boxes))
    ink_bottom = min(height, max(box[3] for box in boxes))
    allowed_bottom = height - footer_top - 7
    shift = max(0, ink_bottom - allowed_bottom)
    if shift == 0:
        return
    if shift <= max(0, ink_top - 12):
        transform = Transformation().translate(0, shift)
    else:
        scale = min(1.0, (allowed_bottom - 12) / max(1, ink_bottom - ink_top))
        transform = Transformation((scale, 0, 0, scale, width * (1-scale)/2,
                                    height * (1-scale) - 12 + scale * ink_top))
    page.add_transformation(transform)
    # Content transforms alone do not move hyperlinks. Keep their hit regions
    # aligned without rasterizing the mathematics or removing PDF navigation.
    for ref in page.get('/Annots', []):
        annotation = ref.get_object()
        if '/Rect' in annotation:
            rect = annotation['/Rect']
            a = transform.apply_on((float(rect[0]), float(rect[1])))
            b = transform.apply_on((float(rect[2]), float(rect[3])))
            annotation[NameObject('/Rect')] = RectangleObject((*a, *b))
        if '/QuadPoints' in annotation:
            values = annotation['/QuadPoints']
            points = [transform.apply_on((float(values[i]), float(values[i+1])))
                      for i in range(0, len(values), 2)]
            annotation[NameObject('/QuadPoints')] = ArrayObject([FloatObject(v) for point in points for v in point])


def append_report(pdf_bytes, snapshot):
    """Compatibility entry point: stamp usage on the final page, never append."""
    from pypdf import PdfReader, PdfWriter
    original = PdfReader(BytesIO(pdf_bytes))
    if not original.pages:
        raise ValueError('Cannot add a footer to an empty PDF')
    writer = PdfWriter()
    writer.clone_document_from_reader(original)
    last = writer.pages[-1]
    # Native SPDF XeLaTeX output has this geometry. Do not misplace a footer on
    # unsupported rotated/cropped input; preserve it via the caller's fallback.
    if last.rotation or tuple(last.cropbox) != tuple(last.mediabox) or tuple(last.mediabox.lower_left) != (0, 0):
        raise ValueError('Unsupported SPDF page geometry')
    with _render_lock:
        overlay, footer_top = render_report(snapshot, (float(last.mediabox.width), float(last.mediabox.height)))
    _reserve_footer(last, pdf_bytes, footer_top)
    last.merge_page(PdfReader(BytesIO(overlay)).pages[0])
    metadata = {str(k):str(v) for k,v in (original.metadata or {}).items() if v is not None}
    metadata['/DeepthinkTokenUsage'] = json.dumps(snapshot, ensure_ascii=False, separators=(',',':'))
    writer.add_metadata(metadata)
    out = BytesIO(); writer.write(out)
    result = out.getvalue()
    assert len(PdfReader(BytesIO(result)).pages) == len(original.pages)
    return result


async def append_current_report(pdf_bytes, event=None):
    ledger = _current.get()
    if ledger is None: raise RuntimeError('Missing request-local SPDF token ledger')
    try:
        if event is not None: event.set_extra(WARNING_KEY, '')
    except Exception: pass
    try:
        return await asyncio.to_thread(append_report, pdf_bytes, ledger.snapshot())
    except Exception as exc:
        # Never discard a valid mathematical answer because telemetry rendering
        # failed. Both command/tool callers surface the explicit warning.
        _log.warning('[mathsolve] SPDF token footer failed: %s', type(exc).__name__)
        warning = '⚠️ 解答 PDF 已生成，但本次 token 用量页脚生成失败，文件中未附用量统计。'
        try:
            if event is not None: event.set_extra(WARNING_KEY, warning)
        except Exception: pass
        return pdf_bytes


def report_warning(event):
    try: return str(event.get_extra(WARNING_KEY, '') or '')
    except Exception: return ''
