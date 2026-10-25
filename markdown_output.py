"""Portable Markdown document output, independent of PDF/image rendering."""
import asyncio
from copy import copy
import html
import json
import logging
import os
from pathlib import Path
import re
import uuid
import unicodedata
from datetime import datetime
from urllib.parse import urlsplit

try:
    from .markdown_svg import MAX_DIAGRAMS, SvgFormatError, svg_markdown_image, decode_svg_data, sanitize_svg
    from .markdown_plot import function_plot_svg
    from .markdown_inline_boxes import BOX_RE, MAX_BOXES, render_box, inline_text, split_inline_code, boxes_for_model, is_box_label, plain_text
    from .markdown_math_scroll import SCROLL_RE, MATH_RE, MAX_SCROLL_FORMULAS, render_scroll_formula, long_inline_formula, scroll_formulas_for_model
    from .markdown_box_context import BoxContext
except ImportError:
    from markdown_svg import MAX_DIAGRAMS, SvgFormatError, svg_markdown_image, decode_svg_data, sanitize_svg
    from markdown_plot import function_plot_svg
    from markdown_inline_boxes import BOX_RE, MAX_BOXES, render_box, inline_text, split_inline_code, boxes_for_model, is_box_label, plain_text
    from markdown_math_scroll import SCROLL_RE, MATH_RE, MAX_SCROLL_FORMULAS, render_scroll_formula, long_inline_formula, scroll_formulas_for_model
    from markdown_box_context import BoxContext

logger = logging.getLogger('astrbot')
MAX_DOCUMENT_BYTES = 1024 * 1024
MD_TOOL_NAME = 'md2img_solve_math_md'
# Accept only our canonical HTML image wrapper, never arbitrary HTML attrs.
# Retain legacy Markdown images for previously generated/quoted documents.
SVG_IMAGE_RE = re.compile(
    r'!\[(?:\\.|[^\]\\\n])*\]\(\s*data:image/svg\+xml;base64,(?P<markdown>[A-Za-z0-9+/=]+)\s*\)'
    r'|<img src="data:image/svg\+xml;base64,(?P<html>[A-Za-z0-9+/=]+)" alt="[^"<>\r\n]*" />'
)


def markdown_document_requested(event):
    """Only this user's current instruction grants MD file delivery permission."""
    text = unicodedata.normalize('NFKC', str(getattr(event, 'message_str', '') or ''))
    text = re.sub(r'(?m)^\s*>[^\n]*$|```[\s\S]*?```|~~~[\s\S]*?~~~', '', text).strip()
    text = text.replace('`', '')
    label = r'(?:markdown|(?<![a-zA-Z])\.?md(?![a-zA-Z0-9_]))'
    if re.match(r'^/?md(?:\s|$)', text):
        return True
    if (re.search(r'(?:不要|不用|别|无需|不需要|禁止)\s*(?:用|发|生成|输出)?\s*'+label, text, re.I)
            or re.search(r'(?:不要|别|不用|无需|不需要)\s*(?:任何)?(?:文件|文档|附件)', text)
            or re.search(r'^(?:请问|我想知道)?\s*(?:如何|怎么|为什么|为何).{0,24}'+label, text, re.I)):
        return False
    return bool(re.search(
        r'(?:用|以|按)\s*'+label+r'\s*(?:的)?\s*(?:文档|文件|格式|形式)?\s*(?:来|给我|帮我|重新)?\s*(?:解答|回答|回复|作答|写|输出|发|整理|导出|证明|评分|批改|画|绘制|作图|$)'
        r'|(?:生成|导出|转成|转为|做成|整理成|保存为|输出|发|给|要)\s*(?:给我|我|成|为)?\s*(?:一份|一个|个|份)?\s*'+label+r'(?:\s*(?:文档|文件|格式|形式|版|解答)|\s*$)'
        r'|'+label+r'\s*(?:文档|文件|格式|形式|版)\s*(?:给我|发给我|发我|输出|解答|回答|回复)',
        text, re.I))


def constrain_markdown_tool(req, event, enabled=True):
    """Route an explicit MD request without mutating the global tool registry."""
    requested = markdown_document_requested(event)
    toolset = getattr(req, 'func_tool', None)
    if toolset is not None and isinstance(getattr(toolset, 'tools', None), list):
        rejected = ({'md2img_render_markdown', 'md2img_solve_math_pdf', 'md2img_solve_math_spdf'}
                    if requested else {MD_TOOL_NAME})
        if not enabled:
            rejected.add(MD_TOOL_NAME)
        local = copy(toolset)
        local.tools = [t for t in toolset.tools if getattr(t, 'name', '') not in rejected]
        req.func_tool = local
    if requested:
        marker = '[本轮 Markdown 文档输出]'
        prompt = str(getattr(req, 'system_prompt', '') or '')
        if marker not in prompt:
            req.system_prompt = prompt+'\n'+marker+(
                '用户要求的是独立 .md 文档，不是图片或 PDF。调用 md2img_solve_math_md；'
                '消息自带和引用图片由工具自动读取，不要编造图片 URL。'
                '不要同时调用图片/PDF工具。工具返回 send_ok=true 表示文件已发出，'
                '只需简短确认，不重复发送解答正文或再次调用工具。失败时如实说明，不擅自改用 PDF。\n'
                if enabled else 'Markdown 文档功能已关闭，请告知用户；不要擅自改发图片或 PDF。\n')
    return requested


class MarkdownFormatError(ValueError):
    pass


def normalize_markdown_document(raw):
    """Keep Markdown/math/code as source, not a fenced or HTML document.

    This does not attempt to control the reader's QQ version or preview mode.
    Static SVG is validated and embedded as image data, never executed/fetched.
    No images are fetched or linked to local/server-only files by this exporter.
    """
    text = str(raw or '').lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n').strip()
    if not text or '\x00' in text or len(text.encode('utf-8')) > MAX_DOCUMENT_BYTES:
        raise MarkdownFormatError('Empty, invalid or oversized Markdown document')
    text = re.sub(r'(?is)^<think>.*?</think>\s*', '', text).strip()
    for _ in range(2):
        wrapper = re.fullmatch(r'(?is)<(?:md|markdown)>\s*(.*?)\s*</(?:md|markdown)>', text)
        if wrapper:
            text = wrapper.group(1).strip()
            continue
        wrapper = re.fullmatch(r'(?is)(`{3,}|~{3,})(?:markdown|md)\s*\n(.*?)\n\1\s*', text)
        if wrapper:
            text = wrapper.group(2).strip()
            continue
        break
    if text.startswith(('{', '[')):
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            pass
        else:
            if isinstance(parsed, (dict, list)):
                raise MarkdownFormatError('Expected Markdown, received a JSON envelope')
    if re.fullmatch(r'(?is)(`{3,}|~{3,})(?:latex|tex)\s*\n.*?\n\1\s*', text):
        raise MarkdownFormatError('The whole answer is a LaTeX code block')

    diagrams = []
    boxes = []
    formulas = []
    box_context = BoxContext()

    def embed_formula(tex):
        if len(formulas) >= MAX_SCROLL_FORMULAS:
            raise MarkdownFormatError('Too many scroll formulas')
        try:
            rendered = render_scroll_formula(tex)
        except ValueError as exc:
            raise MarkdownFormatError(str(exc)) from exc
        token = '\x00MATHSCROLL'+str(len(formulas))+'\x00'
        formulas.append(rendered)
        return token

    def embed_box(text):
        if re.search(r'[$\\<>]', text):
            raise MarkdownFormatError('Invalid inline text box')
        if not is_box_label(text):
            return plain_text(text)
        if len(boxes) >= MAX_BOXES:
            raise MarkdownFormatError('Too many inline text boxes')
        token = '\x00BOX'+str(len(boxes))+'\x00'
        boxes.append(render_box(text))
        return token

    def embed_svg(source):
        if len(diagrams) >= MAX_DIAGRAMS:
            raise MarkdownFormatError('At most 12 SVG diagrams per Markdown document')
        try:
            image = svg_markdown_image(source, len(diagrams)+1)
        except SvgFormatError as exc:
            raise MarkdownFormatError(str(exc)) from exc
        token = '\x00SVG'+str(len(diagrams))+'\x00'
        diagrams.append(image)
        return '\n\n'+token+'\n\n'

    def embed_encoded(match):
        try:
            return embed_svg(decode_svg_data(match['markdown'] or match['html']))
        except SvgFormatError as exc:
            raise MarkdownFormatError(str(exc)) from exc

    def embed_plot(source):
        try:
            return embed_svg(function_plot_svg(source))
        except SvgFormatError as exc:
            raise MarkdownFormatError(str(exc)) from exc

    def prose(value):
        # Revalidate only the exact exporter-owned wrapper, not arbitrary styles.
        value = SCROLL_RE.sub(lambda m: embed_formula(m['tex']), value)
        value = BOX_RE.sub(lambda m: embed_box(html.unescape(m['text'])), value)
        # Only data-image SVG produced by this pipeline can bypass the normal
        # image restriction, and it is revalidated even on a subsequent pass.
        value = SVG_IMAGE_RE.sub(embed_encoded, value)
        value = re.sub(r'<svg\b[^>]*>[\s\S]*?</svg\s*>', lambda m: embed_svg(m[0]), value)
        if re.search(r'\\(?:documentclass|usepackage)\b|\\(?:begin|end)\{document\}', value):
            raise MarkdownFormatError('Expected Markdown, received a LaTeX document')
        if re.search(r'<\s*(?:script|style|svg|iframe|div|table|img|html|body|canvas)\b', value, re.I):
            raise MarkdownFormatError('Only complete validated static SVG is supported; other HTML is not allowed')
        if re.search(r'!\[[^\]\n]*\]\s*(?:\(|\[)', value):
            raise MarkdownFormatError('External or local image dependency')
        if re.search(r'(?m)^\s*>?\s*\[(?:[a-z-]+-card|![A-Z]+)\]', value):
            raise MarkdownFormatError('Custom card/callout extension')
        if re.search(r'(?:\]\s*\(\s*|^\s*\[[^\]]+\]:\s*|<)\s*(?:javascript|data|file|vbscript):', value, re.I | re.M):
            raise MarkdownFormatError('Nonportable or active link')
        # Only normalize math outside code. Dollar delimiters have the widest
        # mobile Markdown math support; never rewrite the equation itself.
        value = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: '$$'+m[1].strip()+'$$', value)
        value = re.sub(r'\\\(([\s\S]*?)\\\)', lambda m: '$'+m[1].strip()+'$', value)
        if len(re.findall(r'(?<!\\)\$\$', value)) % 2:
            raise MarkdownFormatError('Unclosed display math delimiter')
        value = MATH_RE.sub(lambda m: embed_formula(m['display'] if m['display'] is not None else m['inline'])
                            if m['display'] is not None or long_inline_formula(m['inline']) else m[0], value)
        # Preserve math while escaping remaining raw tags outside math/code.
        parts = re.split(r'(\$\$[\s\S]*?\$\$|(?<!\\)\$(?:\\.|[^$\n])+?(?<!\\)\$)', value)
        for index in range(0, len(parts), 2):
            # Only attribute-free line breaks are allowed. Recover the exact
            # legacy escaped form too; literal examples inside code stay literal.
            parts[index] = re.sub(r'(?:<br[ \t]*/?>|&lt;br[ \t]*/?&gt;)', '\x00BR\x00', parts[index], flags=re.I)
            parts[index] = re.sub(r'</?[A-Za-z][^>\n]*>', lambda m: html.escape(m[0]), parts[index])
            parts[index] = parts[index].replace('\x00BR\x00', '<br />')
        return ''.join(parts)

    output, plain, code = [], [], []
    fence = None
    diagram_fence = ''
    def flush():
        if plain:
            for is_code, piece in split_inline_code(box_context.scope(''.join(plain))):
                if is_code:
                    label = inline_text(piece)
                    output.append(embed_box(label) if label is not None else piece)
                else:
                    output.append(prose(piece))
            plain.clear()
    for line in text.splitlines(keepends=True):
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*?)(?:\n)?$', line)
        if fence:
            code.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                source = ''.join(code[1:-1])
                output.append(embed_svg(source) if diagram_fence == 'svg' else
                              embed_plot(source) if diagram_fence == 'functionplot' else ''.join(code))
                code.clear(); fence = None; diagram_fence = ''
        elif marker:
            flush()
            if marker[2].strip().lower() == 'mermaid':
                raise MarkdownFormatError('Diagram extension requires a separate renderer')
            fence = marker[1]
            diagram_fence = marker[2].strip().lower()
            code.append(line)
        else:
            plain.append(line)
    if fence:
        raise MarkdownFormatError('Unclosed code fence')
    flush()
    result = ''.join(output).strip()
    for index, image in enumerate(diagrams):
        result = re.sub(r'\n*\x00SVG'+str(index)+r'\x00\n*', lambda m: '\n\n'+image+'\n\n', result)
    result = re.sub(r'\x00BOX(\d+)\x00', lambda m: boxes[int(m[1])], result)
    result = re.sub(r'\x00MATHSCROLL(\d+)\x00', lambda m: formulas[int(m[1])], result)
    result = result.strip()
    if not result:
        raise MarkdownFormatError('Empty Markdown after removing transport wrappers')
    if len(result.encode('utf-8')) > MAX_DOCUMENT_BYTES:
        raise MarkdownFormatError('Oversized Markdown document')
    return result+'\n'


def _svg_reference_for_model(reference):
    """Quoted documents stay interpretable: give the model SVG, not Base64."""
    count = 0
    def replace(match):
        nonlocal count
        count += 1
        if count > MAX_DIAGRAMS:
            return '[其余历史配图略；请根据本轮题意绘图]'
        try:
            safe = sanitize_svg(decode_svg_data(match['markdown'] or match['html']))
        except SvgFormatError:
            return '[历史配图无法解析；请根据本轮题意重新绘图]'
        return '\n\n```svg\n'+safe+'\n```\n\n'
    return SVG_IMAGE_RE.sub(replace, reference)


def build_markdown_prompt(problem_text, has_image=False, reference='', compact=True):
    reference = _svg_reference_for_model(boxes_for_model(scroll_formulas_for_model(reference))) if reference else ''
    return (
        '请完成用户要求，输出一份适合手机 QQ 原生预览的 Markdown 解答文档。\n'
        '格式约束：\n'
        '1. 只输出 Markdown 正文，不要输出文件路径、<md> 标签或 LaTeX 文档，不要用 JSON 或 HTML 包裹整篇文档。'
        '不要将整篇包在 ```markdown 代码块中。\n'
        '2. 使用 #、##、### 标题、普通列表、引用和必要的标准 Markdown 表格；'
        '表格的竖线须正确转义。单元格内确需换行可用无属性的 <br />，不要写成 &lt;br&gt; 或用反引号包住；'
        '不要使用插件私有 box/callout 标记、CSS、Mermaid 或脚注扩展。\n'
        '小框只用于独立的称呼/落款清单、词汇/术语/缩写表中的主词条，例如 `Dear Professor [Last Name],`、`To whom it may concern`；'
        '确实需要局部突出时才用单反引号，导出器会排成雾蓝圆角小框。默认使用普通正文，不为装饰额外标记句子。'
        '完整句子、普通回答、对话台词、邮件正文、长段落均不加框、不包反引号；长例句用普通列表或引用展示，'
        '这些区域内的称呼、落款、缩写也不套小框，即使只有一两个词；主题行、标题、表格说明列和普通正文同样不套框。'
        '不要因为短或是英语就包反引号；需要一般强调时用普通加粗。除无属性换行标签外，不要自行写 HTML、CSS 或配色。'
        '程序代码使用正常代码块，数学公式仍用数学定界符，不要为做小框把公式放进反引号。\n'
        '3. 数学行内公式用 $...$，行间公式用独立行的 $$ 包围；支持标准 LaTeX 数学语法，'
        '矩阵用 bmatrix，分段函数用 cases，对齐推导用 aligned；不要自定义宏或使用 TikZ/宏包。'
        '长公式请独立放在 $$ 中；多步计算可用 aligned 按逻辑分行，但不必为手机宽度强行拆坏数学结构。'
        '导出器会自动为独立公式添加 C 方案限宽滚动容器，超宽时只滑动公式；短行内公式照常。'
        '仍只写标准数学定界符，不要自行写 span、HTML、CSS 或滚动容器，不要缩小字号或把公式改成图片、表格、代码块。'
        '公式编号写在公式外的正文，不用 tag 或 label 命令。'
        '除下面定义的绘图块外，必要的程序示例才使用代码块，并确保闭合。\n'
        '4. 文件须独立可读，不要嵌入本地路径、临时 URL、外链图片、其他 HTML 或资源附件。'
        '若图形有助于理解用户的问题，主动在相关解释附近绘制静态 SVG，例如几何构型、函数图像、积分区域、向量关系或流程示意；'
        '用户明确要求画图时应配图，用户要求不要图则不画。不需要图的简单运算、纯文字问答不要为装饰强行加图。\n'
        '函数曲线和积分区域图优先使用 ```functionplot 代码块，块内只写 JSON，让导出器按公式计算坐标并生成内嵌 SVG，'
        '不要手算大量像素坐标。格式示例（只展示字段，不要套用示例公式；必须根据当前题目填写）：'
        '{"title":"函数与区域", "x_range":[0,2], "curves":[{"expr":"x**2","label":"y=x²"},{"expr":"2*x","label":"y=2x"}], '
        '"shade_between":[0,1], "shade_range":[0,2], "points":[[0,0,"O"],[2,4,"(2,4)"]]}。'
        '其中 x_range 必填；curves 为 1~4 个函数；y_range 可选 [最小值,最大值]，缺省自动取范围；'
        'shade_between 是需要着色的两条函数的从 0 开始的编号，shade_range 为着色的 x 区间，必须在 x_range 内；'
        '不需要着色就不写这两项。points 为可选的关键点 [x,y,标签]，仅填写已求出的准确点。'
        'expr 支持 x、pi、e、+ - * / **、sin/cos/tan/asin/acos/atan/exp/log/ln/sqrt/abs；'
        '乘法显式写 *，不要写 Python 代码、属性、导入或未定义变量，其他参数先代入数值。'
        '标签用普通文字/Unicode；不连续点左右应分开作图，数值采样图用于辅助理解而不是代替证明。'
        '无法用函数表达的几何构型、流程图等，再使用下面的 svg 方式。\n'
        'SVG 规范：每幅图单独使用 ```svg 代码块，里面给出完整 <svg xmlns="http://www.w3.org/2000/svg" '
        'width="640" height="400" viewBox="0 0 640 400">...</svg> 源码；导出器会校验并转成 Markdown 内嵌 SVG 图片，'
        '手机看到图形而不是代码。不要自行生成 Base64、data URI、图片链接或调用其他绘图工具。'
        '尺寸按内容调整，不要裁切标签；常用宽 640、高 320~520，字号至少 18，浅色填充、清楚的线条，单份最多 12 幅。\n'
        '只使用 svg/g/defs/title/desc/path/rect/circle/ellipse/line/polyline/polygon/text/tspan/marker/clipPath；'
        '纯色填充，使用简单属性，不用 style 标签、脚本、事件属性、foreignObject、image、use、动画、滤镜、渐变或外部资源。'
        '箭头可用 marker，引用仅限本图 url(#id)，不要在 marker/clipPath 中再引用其他图元。'
        'XML 标签、属性引号须闭合，文字中的 &、< 写成 &amp;、&lt;。'
        '图内文字仅用普通文字、Unicode 或 tspan 上下标，不要写 $...$、LaTeX 命令、HTML；复杂公式放在图外 Markdown。\n'
        '绘图须与题意和解答一致：先确定坐标/比例和关键点，再绘制边界、曲线和区域；'
        '坐标映射使用 X=原点X+横轴比例×x、Y=原点Y-纵轴比例×y，注意 SVG 的 y 坐标向下。'
        '函数曲线优先按公式计算足够的采样点，用 polyline 连接；除非已推导出精确控制点，否则不要随手估计 Q/T/C 贝塞尔曲线。'
        '同一条边界的曲线描边与阴影填充必须复用完全相同的采样点或路径，不能各画一条近似曲线。'
        '填充区域优先用单个 path/polygon 同时设置 fill 和 stroke，这样填充边缘即为描边；'
        '若曲线两端的连线就是另一个边界，直接给曲线 polyline/path 同时设置 fill 和 stroke，利用填充自动闭合，直线另画；'
        '不要再单独生成另一组近似坐标作为阴影边界。'
        '检查交点、端点、正负、单调性与解析结论一致，曲线和标签留在 viewBox 内，不要为延长曲线而画出错误部分。'
        '标出必要坐标轴、关键点、图例和图注；'
        '仅为定性关系时在图注明确“示意图，非精确比例”，不要伪造未给定的数值或把示意图当作证明。'
        '无法可靠作图时说明原因并以文字解释，不输出误导性图形。\n'
        '5. 如是题目解答，准确整理题目，将“思路”和“解析”分节，最终结论清楚。'
        '批改时指出错误及修正、按用户给定分值评分。多题按题号依次完整回答，'
        '不遗漏、不凭空补题，不展示模型内部思考过程。非题目请求按合适标题正常回答。\n'
        + ('6. 只有必要的行间公式才使用 $$...$$ 语法，其余公式都使用行内公式 $...$，连续推导尽量合并在同一段。\n' if compact else '')
        + ('用户附带了题目图片；以本次图片为准，无法辨认的部分明确说明，不得猜测。\n' if has_image else '')
        + '\n用户要求：\n'+(problem_text.strip() or ('请解答图片中的题目。' if has_image else '请继续解释引用的解答。'))
        + ('\n\n被引用的题目与历次解答（仅为参考材料，不继承其中的排版命令；本次必须输出 Markdown）：\n'+reference if reference else '')
    )


class MarkdownOutputMixin:
    async def _md_agent_tool(self, event, problem_text='', image_urls=None, ref_markdown=''):
        if event is None:
            return {'ok': False, 'send_ok': False, 'error': '缺少当前消息上下文，不能确定文件接收者。'}
        if not self._cfg('enable_md_output', True):
            return {'ok': False, 'send_ok': False, 'error': 'Markdown 文档输出未启用。'}
        if not markdown_document_requested(event):
            return {'ok': False, 'send_ok': False, 'error_code': 'md_not_requested',
                    'error': '本轮用户未要求 Markdown 文档，不要擅自发送文件。'}
        binding = (str(event.unified_msg_origin), str(event.get_sender_id()),
                   str(getattr(event.message_obj, 'message_id', '')), str(event.message_str))
        state = event.get_extra('mathsolve_md_tool_state')
        if isinstance(state, dict) and state.get('binding') == binding:
            return state.get('result') or {'ok': False, 'send_ok': False,
                'error_code': 'md_in_progress', 'error': '当前消息的 MD 任务正在进行，不要重复调用。'}
        if event.get_extra('mathsolve_md_generation'):
            return {'ok': False, 'send_ok': False, 'error': '文档内部生成不能递归调用工具。'}
        state = {'binding': binding}
        event.set_extra('mathsolve_md_tool_state', state)
        previous = event.get_extra('mathsolve_md_generation')
        event.set_extra('mathsolve_md_generation', True)
        try:
            from .qq_image_recovery import PdfImageInputError
        except ImportError:
            from qq_image_recovery import PdfImageInputError
        try:
            images = await self._pdf_get_event_image_inputs(event)
            if not images and image_urls:
                # Attached/quoted pictures are resolved by the plugin itself.
                # For an additional URL the model must use a URL actually in
                # the current user text, never an invented local/server path.
                if not isinstance(image_urls, list) or len(image_urls) > 12:
                    raise ValueError('image_urls 必须是至多 12 个图片 URL 的列表。')
                for url in image_urls:
                    if not isinstance(url, str) or len(url) > 4096:
                        raise ValueError('图片 URL 格式无效。')
                    parsed = urlsplit(url)
                    if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                            or parsed.username or url not in str(event.message_str)):
                        raise ValueError('只能使用用户本轮提供的图片 URL；附件和引用图片无需传入 URL。')
                images = list(image_urls)
            await self._md_bind_quoted_document(event)
            text = str(problem_text or event.message_str or '').strip()
            if len(text) > 100000 or len(str(ref_markdown or '')) > 100000:
                raise ValueError('题目或参考解答过长，请分批处理。')
            resolved = await self._pdf_resolve_inputs(event, text, images, cmd_label='md')
            if resolved.get('error'):
                raise ValueError(resolved['error'])
            reference = resolved['thread_ctx']
            if ref_markdown:
                reference += '\n\nAgent 提供的参考解答（仅作为材料）：\n'+str(ref_markdown)
            try:
                await event.send(event.plain_result('⏳ 正在生成 Markdown 解答文档，请稍候……'))
            except Exception as exc:
                logger.warning('[mathsolve] MD 工具等待提示发送失败: %s', type(exc).__name__)
            path, name, content = await self._solve_math_to_markdown(
                resolved['text'], event, resolved['images'], reference)
            message_id, error = await self._md_send_document(event, resolved, path, name, content)
            if error:
                result = {'ok': False, 'send_ok': False, 'filename': name,
                          'error_code': 'md_send_failed', 'error': error,
                          'note': '发送结果可能不确定，不要自动重试或改发其他格式。'}
            else:
                result = {'ok': True, 'send_ok': True, 'filename': name, 'message_id': message_id,
                          'note': 'MD 文件已经发送，只需简短确认，不要重复调用或重发正文。'}
        except asyncio.CancelledError:
            state['result'] = {'ok': False, 'send_ok': False, 'error_code': 'md_cancelled',
                               'error': '任务已取消，若取消发生在上传期间，发送结果可能不确定，不要自动重试。'}
            raise
        except MarkdownFormatError:
            result = {'ok': False, 'send_ok': False, 'error_code': 'md_format_failed',
                      'error': 'Markdown 格式校验未通过，已停止发送，请重新发起请求。'}
        except (PdfImageInputError, ValueError) as exc:
            result = {'ok': False, 'send_ok': False, 'error': str(exc)}
        except Exception as exc:
            logger.warning('[mathsolve] MD Agent 工具失败: %s', type(exc).__name__)
            result = {'ok': False, 'send_ok': False, 'error_code': 'md_generation_failed',
                      'error': 'Markdown 生成失败，请检查 /pdf 使用的模型与候补设置，或查看服务器日志。'}
        finally:
            event.set_extra('mathsolve_md_generation', previous)
        state['result'] = result
        return result

    async def _md_send_document(self, event, resolved, path, name, content):
        import astrbot.api.message_components as Comp
        prefix = []
        if event.get_sender_id() and not self._is_private_chat(event):
            prefix.append(Comp.At(qq=event.get_sender_id()))
        message_id, error = await self._send_pdf_file_component(event, Comp, prefix, path, name, label='Markdown')
        if not error:
            await self._pdf_register_thread_result(skey=resolved['skey'], uid=resolved['uid'],
                thread_id=resolved['thread_id'], problem_text=resolved['text'], image_refs=resolved['images'],
                tex_src=content, sent_msg_id=message_id or 'md-file:'+name)
        return message_id, error

    async def _md_bind_quoted_document(self, event):
        """QQ file sends may return no message ID. Match only our exact filename
        in this turn's explicit quote and this user's own stored thread.
        """
        try:
            from .shared import MATH_SESSION_STATE, _get_session_key, _get_sender_uid
        except ImportError:
            from shared import MATH_SESSION_STATE, _get_session_key, _get_sender_uid
        from astrbot.api.message_components import Reply, File
        reply_id = self._extract_reply_msg_id(event)
        if not reply_id:
            return
        names, quoted_text = set(), []
        for part in event.message_obj.message:
            if not isinstance(part, Reply):
                continue
            quoted_text.append(str(part.message_str or '')[:20000])
            for child in (part.chain or [])[:100]:
                if isinstance(child, File) and child.name:
                    names.add(child.name)
        uid = _get_sender_uid(event)
        async with self._state_lock:
            state = MATH_SESSION_STATE.get(_get_session_key(event), {})
            mapping = state.get('pdf_msg_to_thread', {})
            threads = state.get('pdf_threads', {})
            if not isinstance(mapping, dict) or not isinstance(threads, dict):
                return
            matches = set()
            for key, thread_id in list(mapping.items()):
                if not str(key).startswith('md-file:'):
                    continue
                name = key[len('md-file:'):]
                if name not in names and not any(name in text for text in quoted_text):
                    continue
                thread = threads.get(thread_id, {})
                if str(thread.get('owner_uid')) == str(uid):
                    matches.add(thread_id)
            if len(matches) == 1:
                mapping[str(reply_id)] = matches.pop()
                while len(mapping) > 200:
                    mapping.pop(next(iter(mapping)))

    async def _solve_math_to_markdown(self, problem_text, event, image_urls=None, reference=''):
        try:
            from .shared import _get_sender_display_name, _sanitize_filename
        except ImportError:
            from shared import _get_sender_display_name, _sanitize_filename
        primary = str(self._cfg('pdf_provider_id', '') or '').strip()
        try:
            current = await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
        except Exception:
            current = ''
        routed = str(self._cfg('pdf_gpt_provider_id', '') or '').strip() if self._pdf_requests_gpt_model(problem_text) else ''
        images = await self._pdf_snapshot_image_urls(image_urls, event=event) if image_urls else []
        chain = self._pdf_build_provider_chain(primary, current, routed_provider_id=routed,
            routed_timeout_sec=self._pdf_parse_timeout_value(self._cfg('pdf_gpt_provider_timeout_sec', None), self._pdf_default_model_timeout_sec()),
            routed_compact_prompt_rules=self._pdf_compact_enabled_for_slot('pdf_gpt_provider_enable_compact_prompt_rules') if routed else None)
        prompt = lambda spec: build_markdown_prompt(problem_text, bool(images), reference,
                                                     bool(spec.get('compact_prompt_rules', True)))
        raw, provider, _ = await self._pdf_llm_generate_with_fallback(
            provider_chain=chain, prompt=prompt, image_urls=images, purpose='md-initial')
        try:
            content = normalize_markdown_document(raw)
        except MarkdownFormatError as exc:
            # One bounded format repair, not a PDF compilation/guard loop.
            reason = str(exc)
            raw, provider, _ = await self._pdf_llm_generate_with_fallback(
                provider_chain=self._pdf_prioritize_provider_chain(chain, provider),
                prompt=lambda spec: prompt(spec)+'\n\n上一版格式校验未通过：'+reason+
                    '\n请完整重写为符合以上约束的 Markdown，不要改变数学含义。若是 SVG 错误，请修正图的 XML/属性/坐标，'
                    '仍用 svg 代码块给出；函数图使用 functionplot 时修正 JSON 字段、公式或区间。不要改成外链或省略用户需要的配图。上一版如下：\n'+raw[:100000],
                image_urls=images, purpose='md-format-repair')
            content = normalize_markdown_document(raw)
        model = re.sub(r'[\x00-\x1f\x7f]', '', await self._pdf_model_label_for_filename(provider))[:24]
        user = re.sub(r'[\x00-\x1f\x7f]', '', _get_sender_display_name(event))[:24]
        stamp = datetime.now().strftime('%Y%m%d%H%M%S')
        # Bound names even if a provider/user supplied an extremely long label.
        name = _sanitize_filename(f'MD-{model}-{user}-{stamp}-{uuid.uuid4().hex[:8]}.md')
        # Separate bucket: new Markdown files must not evict existing PDFs from
        # their max-file limit. Clean only this command's own cache on generation.
        directory = Path(self.PDF_CACHE_DIR)/'_md_documents'
        directory.mkdir(parents=True, exist_ok=True)
        if self._cfg('cache_cleanup_enabled', True):
            try:
                self._clean_one_cache_dir(str(directory),
                    ttl_sec=self._cache_int_cfg('pdf_cache_ttl_sec', 604800),
                    max_files=self._cache_int_cfg('pdf_cache_max_files', 300),
                    protect_recent_sec=self._cache_int_cfg('cache_cleanup_protect_recent_sec', 300))
            except Exception as exc:
                logger.warning('[mathsolve] Markdown 缓存清理暂不可用: %s', type(exc).__name__)
        path = directory/name
        temporary = directory/(name+'.tmp-'+uuid.uuid4().hex)
        try:
            with temporary.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(content)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return str(path), name, content

    async def _md_command(self, event):
        # Separate event flag prevents status/error messages being re-rendered as
        # images and prevents math-coach/tools contaminating this document call.
        event.set_extra('mathsolve_md_generation', True)
        if not self._cfg('enable_md_output', True):
            yield event.plain_result('Markdown 文档输出未启用')
            return
        match = re.fullmatch(r'\s*/?md(?:\s+([\s\S]*))?\s*', event.message_str or '', re.I)
        if not match:
            yield event.plain_result('用法：/md 题目文字，或发送题目图片并附上 /md')
            return
        arg = (match.group(1) or '').strip()
        try:
            from .qq_image_recovery import PdfImageInputError
        except ImportError:
            from qq_image_recovery import PdfImageInputError
        try:
            images = await self._pdf_get_event_image_inputs(event)
            await self._md_bind_quoted_document(event)
            resolved = await self._pdf_resolve_inputs(event, arg, images, cmd_label='md')
        except PdfImageInputError as exc:
            yield event.plain_result('❌ Markdown 生成已暂停：'+str(exc))
            return
        if resolved.get('error'):
            yield event.plain_result(resolved['error'])
            return
        for notice in resolved.get('notices') or []:
            yield event.plain_result(notice)
        yield event.plain_result('⏳ 正在生成 Markdown 解答文档，请稍候……')
        try:
            path, name, content = await self._solve_math_to_markdown(
                resolved['text'], event, resolved['images'], resolved['thread_ctx'])
        except PdfImageInputError as exc:
            yield event.plain_result('❌ Markdown 生成已暂停：'+str(exc))
            return
        except MarkdownFormatError:
            yield event.plain_result('❌ Markdown 格式校验未通过，已停止发送，以免打开后显示源码或不完整内容。请重试。')
            return
        except Exception as exc:
            logger.warning('[mathsolve] Markdown 生成失败: %s', type(exc).__name__)
            yield event.plain_result('❌ Markdown 生成失败，请检查 /pdf 使用的模型及候补模型是否可用，或联系管理员检查日志。')
            return
        message_id, error = await self._md_send_document(event, resolved, path, name, content)
        if error:
            yield event.plain_result(error)
            return

