"""Small declarative function plotter: numeric AST interpreter, never eval/exec."""
import ast
import html
import json
import math

try:
    from .markdown_svg import SvgFormatError
except ImportError:
    from markdown_svg import SvgFormatError

FUNCTIONS = {'sin': (math.sin, {1}), 'cos': (math.cos, {1}), 'tan': (math.tan, {1}),
             'asin': (math.asin, {1}), 'acos': (math.acos, {1}), 'atan': (math.atan, {1}),
             'exp': (math.exp, {1}), 'log': (math.log, {1, 2}), 'ln': (math.log, {1}),
             'sqrt': (math.sqrt, {1}), 'abs': (abs, {1})}
COLORS = ('#3978b7', '#d47a45', '#689b7d', '#8a70b5')


def compile_expression(expression):
    """Only constants, x, arithmetic and explicitly listed numeric functions."""
    if not isinstance(expression, str) or len(expression) > 240:
        raise SvgFormatError('Function expression must be at most 240 characters')
    try:
        tree = ast.parse(expression.replace('^', '**').replace('π', 'pi'), mode='eval')
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise SvgFormatError('Invalid function expression; use explicit * and **') from exc
    count = 0
    def validate(node, depth=0):
        nonlocal count
        count += 1
        if count > 80 or depth > 12:
            raise SvgFormatError('Function expression is too complex')
        if isinstance(node, ast.Expression):
            validate(node.body, depth+1)
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or not math.isfinite(node.value) or abs(node.value) > 1000000:
                raise SvgFormatError('Invalid numeric constant')
        elif isinstance(node, ast.Name):
            if node.id not in ('x', 'pi', 'e'):
                raise SvgFormatError('Only x, pi and e are allowed in expressions')
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            validate(node.left, depth+1); validate(node.right, depth+1)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            validate(node.operand, depth+1)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
            if node.keywords or len(node.args) not in FUNCTIONS[node.func.id][1]:
                raise SvgFormatError('Wrong function arguments')
            for argument in node.args:
                validate(argument, depth+1)
        else:
            raise SvgFormatError('Expression supports only numeric arithmetic and listed functions; no code, attributes or imports')
    validate(tree)
    def evaluate(x):
        def visit(node):
            if isinstance(node, ast.Constant):
                value = float(node.value)
            elif isinstance(node, ast.Name):
                value = {'x': x, 'pi': math.pi, 'e': math.e}[node.id]
            elif isinstance(node, ast.UnaryOp):
                value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.Call):
                value = FUNCTIONS[node.func.id][0](*(visit(n) for n in node.args))
            else:
                a, b = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add): value = a+b
                elif isinstance(node.op, ast.Sub): value = a-b
                elif isinstance(node.op, ast.Mult): value = a*b
                elif isinstance(node.op, ast.Div): value = a/b
                else:
                    if abs(b) > 20:
                        raise ValueError('Exponent outside bounds')
                    value = a**b
            if isinstance(value, complex) or not math.isfinite(value) or abs(value) > 1000000:
                raise ValueError('Value outside finite plotting range')
            return value
        try:
            return visit(tree.body)
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
    return evaluate


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 10000:
        raise SvgFormatError('Plot ranges and points must be finite numbers within 10000')
    return float(value)


def _range(value):
    if not isinstance(value, list) or len(value) != 2:
        raise SvgFormatError('Plot range must be [min, max]')
    a, b = map(_number, value)
    if b-a < 0.000001:
        raise SvgFormatError('Plot maximum must exceed minimum')
    return a, b


def _label(value, default):
    if value is None:
        value = default
    if not isinstance(value, str) or len(value) > 80 or any(c in value for c in ('$', '\\', '\n')):
        raise SvgFormatError('Use short plain/Unicode plot labels, not LaTeX')
    return html.escape(value)


def function_plot_svg(raw):
    if len(raw.encode('utf-8')) > 12000:
        raise SvgFormatError('Function plot specification is too large')
    try:
        spec = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise SvgFormatError('functionplot needs valid JSON') from exc
    keys = {'title', 'x_range', 'y_range', 'curves', 'shade_between', 'shade_range', 'points'}
    if not isinstance(spec, dict) or set(spec)-keys:
        raise SvgFormatError('Unknown functionplot fields')
    xmin, xmax = _range(spec.get('x_range'))
    curves = spec.get('curves')
    if not isinstance(curves, list) or not 1 <= len(curves) <= 4:
        raise SvgFormatError('Function plot needs 1 to 4 curves')
    functions, labels = [], []
    for curve in curves:
        if not isinstance(curve, dict) or set(curve)-{'expr', 'label'}:
            raise SvgFormatError('Curve supports only expr and label')
        functions.append(compile_expression(curve.get('expr')))
        labels.append(_label(curve.get('label'), 'y = '+str(curve.get('expr'))))
    points = spec.get('points', [])
    if not isinstance(points, list) or len(points) > 12:
        raise SvgFormatError('At most 12 marked points')
    markers = []
    for point in points:
        if not isinstance(point, list) or len(point) not in (2, 3):
            raise SvgFormatError('Marked point is [x, y, optional label]')
        x, y = _number(point[0]), _number(point[1])
        if not xmin <= x <= xmax:
            raise SvgFormatError('Marked point is outside x_range')
        markers.append((x, y, _label(point[2] if len(point) == 3 else None, f'({x:g}, {y:g})')))
    shade = spec.get('shade_between')
    a, b = _range(spec.get('shade_range', [xmin, xmax]))
    if not xmin <= a < b <= xmax:
        raise SvgFormatError('shade_range must lie inside x_range')
    if shade is not None and (not isinstance(shade, list) or len(shade) != 2
                             or any(type(i) is not int or not 0 <= i < len(curves) for i in shade)
                             or shade[0] == shade[1]):
        raise SvgFormatError('shade_between is two distinct zero-based curve indices')
    # The SAME evaluated samples are reused by the curve and shaded polygon.
    xs = sorted(set([xmin+(xmax-xmin)*i/160 for i in range(161)]+[a, b]+
                    ([0.0] if xmin <= 0 <= xmax else [])+[p[0] for p in markers]))
    ys = [[fn(x) for x in xs] for fn in functions]
    finite = [y for row in ys for y in row if y is not None]
    if not finite or any(all(y is None for y in row) for row in ys):
        raise SvgFormatError('Function has no finite values in the requested interval')
    if 'y_range' in spec:
        ymin, ymax = _range(spec['y_range'])
    else:
        # Ignore extreme pole outliers for automatic framing; still clip and
        # split discontinuities. Explicit y_range remains available.
        ordered = sorted(finite)
        low, high = ordered[len(ordered)//50], ordered[-1-len(ordered)//50]
        ymin, ymax = min(0, low, *(p[1] for p in markers)), max(0, high, *(p[1] for p in markers))
        span = max(ymax-ymin, 1)
        ymin -= span*0.08; ymax += span*0.12
    left, right, top, bottom = 72, 566, 62, 340
    px = lambda x: left+(x-xmin)/(xmax-xmin)*(right-left)
    py = lambda y: bottom-(y-ymin)/(ymax-ymin)*(bottom-top)
    xy = lambda x, y: f'{px(x):.3f},{py(y):.3f}'
    span = ymax-ymin
    # Detect jumps and undefined midpoints so we do not join across simple
    # poles/domain gaps. Numerical plots are still illustrative, not proofs.
    breaks = []
    for fn, row in zip(functions, ys):
        row_breaks = set()
        for i in range(1, len(xs)):
            previous, current = row[i-1], row[i]
            mid = fn((xs[i-1]+xs[i])/2)
            if previous is None or current is None or mid is None or abs(current-previous) > span*0.5:
                row_breaks.add(i)
        breaks.append(row_breaks)
    if shade is not None:
        indices = [i for i, x in enumerate(xs) if a <= x <= b]
        for j in shade:
            if any(ys[j][i] is None or (i != indices[0] and i in breaks[j]) for i in indices):
                raise SvgFormatError('Shading interval crosses a discontinuity; split it into separate functionplots')
    title = _label(spec.get('title'), '函数图与区域示意')
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420"><title>{title}</title>',
              '<rect x="1" y="1" width="638" height="418" rx="14" fill="#f7f9fc" stroke="#dce4ef"/>',
              '<defs><clipPath id="plot-clip"><rect x="72" y="62" width="494" height="278"/></clipPath></defs>',
              f'<g font-family="sans-serif" font-size="18" fill="#334a63"><text x="320" y="32" text-anchor="middle" font-size="21">{title}</text>']
    # Evenly spaced ticks are bounded and independent from model arithmetic.
    for i in range(6):
        x = xmin+(xmax-xmin)*i/5
        y = ymin+(ymax-ymin)*i/5
        output += [f'<line x1="{px(x):.3f}" y1="62" x2="{px(x):.3f}" y2="340" stroke="#e1e8f0"/>',
                   f'<text x="{px(x):.3f}" y="363" text-anchor="middle" font-size="15">{x:.3g}</text>',
                   f'<line x1="72" y1="{py(y):.3f}" x2="566" y2="{py(y):.3f}" stroke="#e1e8f0"/>',
                   f'<text x="62" y="{py(y)+5:.3f}" text-anchor="end" font-size="15">{y:.3g}</text>']
    axis_x, axis_y = px(min(max(0, xmin), xmax)), py(min(max(0, ymin), ymax))
    output += [f'<line x1="72" y1="{axis_y:.3f}" x2="575" y2="{axis_y:.3f}" stroke="#6a7d91" stroke-width="1.5"/>',
               f'<line x1="{axis_x:.3f}" y1="340" x2="{axis_x:.3f}" y2="52" stroke="#6a7d91" stroke-width="1.5"/>',
               f'<text x="583" y="{axis_y+6:.3f}">x</text><text x="{axis_x-8:.3f}" y="51">y</text>', '<g clip-path="url(#plot-clip)">']
    if shade is not None:
        points_string = ' '.join(xy(xs[i], ys[shade[0]][i]) for i in indices)
        points_string += ' '+' '.join(xy(xs[i], ys[shade[1]][i]) for i in reversed(indices))
        output.append(f'<polygon points="{points_string}" fill="#d9e9fa"/>')
    for j, row in enumerate(ys):
        segments, segment = [], []
        for i, y in enumerate(row):
            if y is None or i in breaks[j] or not ymin-span <= y <= ymax+span:
                if len(segment) > 1: segments.append(segment)
                segment = []
            if y is not None and ymin-span <= y <= ymax+span:
                segment.append(xy(xs[i], y))
        if len(segment) > 1: segments.append(segment)
        if not segments:
            raise SvgFormatError('Curve has no visible continuous segment; adjust plot range')
        for segment in segments:
            output.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{COLORS[j]}" stroke-width="2.5" stroke-linejoin="round"/>')
    output.append('</g>')
    for x, y, label in markers:
        if not ymin <= y <= ymax:
            raise SvgFormatError('Marked point is outside y_range')
        dx, anchor = (-8, 'end') if px(x) > 460 else (8, 'start')
        output.append(f'<circle cx="{px(x):.3f}" cy="{py(y):.3f}" r="3.5" fill="#334a63"/><text x="{px(x)+dx:.3f}" y="{max(78,py(y)-9):.3f}" text-anchor="{anchor}" font-size="17">{label}</text>')
    for j, label in enumerate(labels):
        x = 82+(j % 2)*270
        y = 386+(j//2)*22
        output.append(f'<line x1="{x}" y1="{y-6}" x2="{x+22}" y2="{y-6}" stroke="{COLORS[j]}" stroke-width="3"/><text x="{x+30}" y="{y}" font-size="17">{label}</text>')
    output.append('</g></svg>')
    return ''.join(output)
