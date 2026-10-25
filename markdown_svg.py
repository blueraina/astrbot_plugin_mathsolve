"""Bounded static SVG -> self-contained Markdown image; never executes/fetches."""
import base64
import binascii
import html
import math
import re
import xml.etree.ElementTree as ET

SVG_NS = 'http://www.w3.org/2000/svg'
MAX_SVG_BYTES = 128 * 1024
MAX_SVG_NODES = 1600
MAX_SVG_DEPTH = 24
MAX_DIAGRAMS = 12
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
COMMON = set('id fill fill-opacity fill-rule stroke stroke-width stroke-opacity stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit opacity transform clip-path vector-effect'.split())
TEXT = set('font-family font-size font-weight font-style text-anchor dominant-baseline alignment-baseline letter-spacing'.split())
SHAPES = {
    'svg': set('viewBox width height preserveAspectRatio version role aria-label'.split()),
    'g': set(), 'defs': set(), 'title': set(), 'desc': set(),
    'path': {'d', 'marker-start', 'marker-mid', 'marker-end'},
    'rect': set('x y width height rx ry'.split()),
    'circle': set('cx cy r'.split()), 'ellipse': set('cx cy rx ry'.split()),
    'line': set('x1 y1 x2 y2 marker-start marker-mid marker-end'.split()),
    'polyline': set('points marker-start marker-mid marker-end'.split()),
    'polygon': {'points'},
    'text': set('x y dx dy rotate textLength lengthAdjust'.split()),
    'tspan': set('x y dx dy rotate textLength lengthAdjust baseline-shift'.split()),
    'marker': set('viewBox refX refY markerWidth markerHeight markerUnits orient preserveAspectRatio'.split()),
    'clipPath': {'clipPathUnits'},
}
NUMERIC = set('x y x1 y1 x2 y2 cx cy r rx ry width height dx dy rotate font-size letter-spacing stroke-width stroke-dashoffset stroke-miterlimit opacity fill-opacity stroke-opacity refX refY markerWidth markerHeight textLength'.split())
ENUMS = {
    'fill-rule': {'nonzero', 'evenodd'}, 'stroke-linecap': {'butt', 'round', 'square'},
    'stroke-linejoin': {'miter', 'round', 'bevel'}, 'text-anchor': {'start', 'middle', 'end'},
    'vector-effect': {'none', 'non-scaling-stroke'}, 'font-style': {'normal', 'italic', 'oblique'},
    'font-weight': {'normal', 'bold', 'bolder', 'lighter', *map(str, range(100, 1000, 100))},
    'markerUnits': {'strokeWidth', 'userSpaceOnUse'}, 'clipPathUnits': {'userSpaceOnUse', 'objectBoundingBox'},
    'lengthAdjust': {'spacing', 'spacingAndGlyphs'},
    'dominant-baseline': {'auto', 'middle', 'central', 'hanging', 'alphabetic', 'text-before-edge', 'text-after-edge'},
    'alignment-baseline': {'auto', 'baseline', 'middle', 'central', 'hanging', 'alphabetic', 'before-edge', 'after-edge', 'text-before-edge', 'text-after-edge'},
}


class SvgFormatError(ValueError):
    pass


def _numbers(value, *, unit=False):
    if unit:
        value = re.sub(r'(?<=\d)px\b', '', value)
    if not value.strip() or re.sub(NUMBER, '', value).strip(' ,\t\r\n'):
        raise SvgFormatError('SVG numeric attribute is invalid')
    numbers = [float(n) for n in re.findall(NUMBER, value)]
    if len(numbers) > 20000 or any(not math.isfinite(n) or abs(n) > 1000000 for n in numbers):
        raise SvgFormatError('SVG numeric attribute exceeds limits')
    return numbers


def _path(value):
    if re.sub(NUMBER+r'|[MmLlHhVvCcSsQqTtAaZz]|[\s,]', '', value):
        raise SvgFormatError('Invalid SVG path data')
    tokens = re.findall(NUMBER+r'|[A-Za-z]', value)
    if not tokens or tokens[0] not in ('M', 'm') or len(tokens) > 20000:
        raise SvgFormatError('Invalid or oversized SVG path')
    counts = {'M': 2, 'L': 2, 'H': 1, 'V': 1, 'C': 6, 'S': 4, 'Q': 4, 'T': 2, 'A': 7, 'Z': 0}
    index = 0
    while index < len(tokens):
        command = tokens[index].upper()
        if command not in counts:
            raise SvgFormatError('SVG path lacks a command')
        index += 1
        start = index
        while index < len(tokens) and not re.fullmatch('[A-Za-z]', tokens[index]):
            index += 1
        values = _numbers(' '.join(tokens[start:index])) if start != index else []
        count = counts[command]
        if (not count and values) or (count and (not values or len(values) % count)):
            raise SvgFormatError('SVG path command has wrong coordinate count')
        if command == 'A':
            for i in range(0, len(values), 7):
                if values[i] < 0 or values[i+1] < 0 or values[i+3] not in (0, 1) or values[i+4] not in (0, 1):
                    raise SvgFormatError('Invalid SVG arc parameters')


def _transform(value):
    matches = list(re.finditer(r'(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^()]*)\)', value))
    remainder = re.sub(r'(matrix|translate|scale|rotate|skewX|skewY)\s*\([^()]*\)', '', value)
    if not matches or len(matches) > 16 or remainder.strip(' ,\t\r\n'):
        raise SvgFormatError('Invalid SVG transform')
    counts = {'matrix': {6}, 'translate': {1, 2}, 'scale': {1, 2}, 'rotate': {1, 3}, 'skewX': {1}, 'skewY': {1}}
    for match in matches:
        if len(_numbers(match[2])) not in counts[match[1]]:
            raise SvgFormatError('Invalid SVG transform arguments')


def _local_name(name):
    if name.startswith('{'):
        namespace, name = name[1:].split('}', 1)
        if namespace != SVG_NS:
            raise SvgFormatError('Only SVG namespace is supported')
    return name


def sanitize_svg(source):
    """Reject, rather than silently remove, unsafe/unknown diagram content."""
    if not isinstance(source, str) or len(source.encode('utf-8')) > MAX_SVG_BYTES:
        raise SvgFormatError('SVG must be at most 128 KiB')
    source = source.strip()
    source = re.sub(r'^<\?xml\s+[^?]*\?>\s*', '', source, count=1)
    source = re.sub(r'<!--[\s\S]*?-->', '', source)
    if re.search(r'<!|<\?|\x00', source):
        raise SvgFormatError('SVG declarations, entities and processing instructions are not allowed')
    try:
        root = ET.fromstring(source)
    except (ET.ParseError, ValueError, RecursionError) as exc:
        raise SvgFormatError('SVG XML is malformed; close tags and escape & as &amp;') from exc
    if _local_name(root.tag) != 'svg':
        raise SvgFormatError('Diagram root must be svg')
    ids, refs, visible = {}, [], False
    stack = [(root, 0, False)]
    nodes = 0
    while stack:
        node, depth, definition = stack.pop()
        nodes += 1
        if nodes > MAX_SVG_NODES or depth > MAX_SVG_DEPTH:
            raise SvgFormatError('SVG structure exceeds limits')
        tag = _local_name(node.tag)
        if tag not in SHAPES or (tag == 'svg' and node is not root):
            raise SvgFormatError('SVG supports static shapes/text only; no script, foreignObject, image, use, animation or external resources')
        node.tag = tag
        definition = definition or tag in ('defs', 'clipPath', 'marker')
        if not definition and tag in ('path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'text'):
            visible = True
        allowed = COMMON | SHAPES[tag] | (TEXT if tag in ('svg', 'g', 'text', 'tspan') else set())
        # A small presentation-property subset is converted to real attributes.
        # No stylesheet, CSS URL, custom property or selector is ever retained.
        style = node.attrib.pop('style', '')
        for entry in style.split(';'):
            if not entry.strip():
                continue
            key, separator, value = entry.partition(':')
            key, value = key.strip(), value.strip()
            if not separator or key not in (COMMON | TEXT) - {'id', 'transform', 'clip-path'} or key not in allowed:
                raise SvgFormatError('SVG style contains unsupported properties; use simple presentation attributes')
            node.set(key, value)
        for key, value in list(node.attrib.items()):
            if key not in allowed or len(value) > MAX_SVG_BYTES:
                raise SvgFormatError('SVG contains unsupported attributes')
            if key == 'id':
                if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]{0,79}', value) or value in ids:
                    raise SvgFormatError('SVG ids must be valid and unique')
                ids[value] = tag
            elif key in ('fill', 'stroke', 'clip-path', 'marker-start', 'marker-mid', 'marker-end'):
                ref = re.fullmatch(r'url\(#([A-Za-z_][A-Za-z0-9_.-]{0,79})\)', value)
                if ref:
                    expected = 'clipPath' if key == 'clip-path' else 'marker' if key.startswith('marker-') else None
                    if expected is None or definition:
                        raise SvgFormatError('Use solid fill/stroke colors, not paint servers')
                    refs.append((ref[1], expected))
                elif key not in ('fill', 'stroke'):
                    if value != 'none':
                        raise SvgFormatError('SVG references must target a local id')
                elif not (re.fullmatch(r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?', value)
                          or re.fullmatch(r'[A-Za-z]{1,24}', value)
                          or re.fullmatch(r'rgba?\(\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+(?:\s*,\s*[\d.]+)?\s*\)', value)):
                    raise SvgFormatError('SVG paint must be a solid color')
            elif key == 'd':
                _path(value)
            elif key == 'points':
                values = _numbers(value)
                if len(values) < 4 or len(values) % 2:
                    raise SvgFormatError('SVG points must be coordinate pairs')
            elif key == 'transform':
                _transform(value)
            elif key == 'viewBox':
                values = _numbers(value)
                if len(values) != 4 or not all(0 < n <= 4096 for n in values[2:]):
                    raise SvgFormatError('SVG viewBox must have four numbers and positive dimensions at most 4096')
            elif key in NUMERIC or key == 'stroke-dasharray':
                if key == 'stroke-dasharray' and value == 'none':
                    continue
                if node is root and key in ('width', 'height') and value == '100%':
                    continue
                values = _numbers(value, unit=True)
                if key not in ('x', 'y', 'dx', 'dy', 'rotate', 'stroke-dasharray') and len(values) != 1:
                    raise SvgFormatError('SVG scalar attribute needs one number')
                if key in ('width', 'height', 'r', 'rx', 'ry', 'stroke-width', 'font-size', 'markerWidth', 'markerHeight') and any(n < 0 for n in values):
                    raise SvgFormatError('SVG dimensions cannot be negative')
                if key in ('opacity', 'fill-opacity', 'stroke-opacity') and any(not 0 <= n <= 1 for n in values):
                    raise SvgFormatError('SVG opacity must be between zero and one')
            elif key in ENUMS:
                if value not in ENUMS[key]:
                    raise SvgFormatError('SVG presentation value is unsupported')
            elif key == 'orient':
                if value not in ('auto', 'auto-start-reverse'):
                    _numbers(value)
            elif key == 'baseline-shift':
                if value not in ('sub', 'super', 'baseline'):
                    _numbers(value, unit=True)
            elif key == 'font-family':
                if not re.fullmatch(r"[\w\s,'\"-]{1,160}", value):
                    raise SvgFormatError('SVG font-family must contain only local font names')
            elif key == 'preserveAspectRatio':
                if not re.fullmatch(r'(?:none|x(?:Min|Mid|Max)Y(?:Min|Mid|Max)(?:\s+(?:meet|slice))?)', value):
                    raise SvgFormatError('Invalid SVG aspect ratio')
            elif key == 'version' and value not in ('1.0', '1.1', '2.0'):
                raise SvgFormatError('Invalid SVG version')
            elif key == 'role' and value != 'img':
                raise SvgFormatError('Invalid SVG role')
        if tag in ('text', 'tspan') and re.search(r'\$|\\[A-Za-z]+', ''.join(node.itertext())):
            raise SvgFormatError('SVG text cannot render LaTeX; use plain/Unicode labels and put formulas in Markdown')
        stack.extend((child, depth+1, definition) for child in reversed(list(node)))
    if not visible:
        raise SvgFormatError('SVG has no visible shapes or labels')
    if any(ids.get(name) != expected for name, expected in refs):
        raise SvgFormatError('SVG marker/clip reference is missing or has the wrong type')
    if 'viewBox' not in root.attrib:
        if not all(k in root.attrib and root.get(k) != '100%' for k in ('width', 'height')):
            raise SvgFormatError('SVG needs a viewBox or explicit width and height')
        sizes = [_numbers(root.get(k), unit=True) for k in ('width', 'height')]
        if any(len(n) != 1 or not 0 < n[0] <= 4096 for n in sizes):
            raise SvgFormatError('Invalid SVG width or height')
        root.set('viewBox', '0 0 '+str(sizes[0][0])+' '+str(sizes[1][0]))
    box = _numbers(root.get('viewBox'))
    for key, fallback in (('width', box[2]), ('height', box[3])):
        value = root.get(key)
        sizes = [fallback] if value in (None, '100%') else _numbers(value, unit=True)
        if len(sizes) != 1 or not 0 < sizes[0] <= 4096:
            raise SvgFormatError('Invalid SVG width or height')
        root.set(key, format(sizes[0], 'g'))
    root.set('xmlns', SVG_NS)
    result = ET.tostring(root, encoding='unicode', short_empty_elements=True)
    if len(result.encode('utf-8')) > MAX_SVG_BYTES:
        raise SvgFormatError('Serialized SVG exceeds 128 KiB')
    return result


def svg_markdown_image(source, number=1):
    safe = sanitize_svg(source)
    title = next((n.text for n in ET.fromstring(safe).iter() if n.tag == '{'+SVG_NS+'}title' and n.text), '')
    title = re.sub(r'\s+', ' ', title).strip()[:120] or f'图 {number}'
    # Keep image alt text from creating Markdown/HTML structure.
    title = re.sub(r'[\\\[\]<>`]', '', title)
    payload = base64.b64encode(safe.encode('utf-8')).decode('ascii')
    # QQ's Markdown image syntax can expose long data URIs as literal text.
    # The same SVG payload renders through this minimal, client-tested tag.
    return f'<img src="data:image/svg+xml;base64,{payload}" alt="{html.escape(title, quote=True)}" />'


def decode_svg_data(payload):
    if len(payload) > MAX_SVG_BYTES * 2:
        raise SvgFormatError('Encoded SVG exceeds limits')
    try:
        return base64.b64decode(re.sub(r'\s+', '', payload), validate=True).decode('utf-8')
    except (ValueError, UnicodeError, binascii.Error) as exc:
        raise SvgFormatError('Invalid UTF-8 Base64 SVG') from exc
