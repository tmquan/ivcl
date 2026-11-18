"""Labels in LaTeX (Computer Modern) type.

Label mini-markup used by the circuit code:
    'V_in'  'V_{in,p}'  'I_out,n'  'M_1'  '(W/L)_2'  'V_out (or I_out)'
    '!V_on'    leading '!'  = overline (logical complement)
    '~plain'   leading '~'  = upright text, no maths
    '$...$'    LaTeX maths as written, for anything the mini-markup cannot say:  r'$\frac{1}{g_m}$'

tex(label)      -> LaTeX maths source (no $ delimiters)
The outline helpers need matplotlib (its mathtext engine and bundled Computer Modern fonts);
no TeX installation is required.
"""
import os
import re

_WORDS = {'or', 'and', 'to', 'of'}


def _base(b):
    return b.replace('·', r'\cdot ')


def _tok(t):
    m = re.match(r'^(.*?)_(\{[^}]*\}|[A-Za-z0-9,]+)(.*)$', t)
    if m:
        base, sub, rest = m.groups()
        sub = sub.strip('{}')
        subtex = sub if (len(sub) == 1 or sub.isdigit()) else r'\mathrm{%s}' % sub
        return '%s_{%s}%s' % (_base(base), subtex, rest)
    m = re.match(r'^(\(?)([A-Za-z]+)(\)?)$', t)
    if m and m.group(2).lower() in _WORDS:
        return '%s\\mathrm{%s}%s' % m.groups()
    return _base(t)


def tex(label):
    if label.startswith('$'):
        return label.strip('$')
    if label.startswith('~'):
        return r'\mathrm{%s}' % label[1:].replace(' ', r'\ ')
    over = label.startswith('!')
    s = label[1:] if over else label
    body = r'\ '.join(_tok(t) for t in s.split(' ') if t)
    return r'\overline{%s}' % body if over else body


# ---------------------------------------------------------------- outlines (need matplotlib)
_cache = {}


def _fonts():
    import matplotlib
    d = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
    return os.path.join(d, 'cmr10.ttf'), os.path.join(d, 'cmb10.ttf')


def text_path(s, size, mode='math'):
    """mode 'math': s is LaTeX maths;  'text' / 'bold': s is plain text set in Computer Modern Roman / Bold;
    'mixed': s is a sentence with $maths$ in it.  -> matplotlib Path in px, baseline at y = 0, y up."""
    key = (s, round(size, 3), mode)
    if key not in _cache:
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        if mode == 'math':
            tp = TextPath((0, 0), '$%s$' % s, size=size, prop=FontProperties(family='serif', math_fontfamily='cm'))
        elif mode == 'mixed':
            tp = TextPath((0, 0), s, size=size, prop=FontProperties(fname=_fonts()[0], math_fontfamily='cm'))
        else:
            reg, bold = _fonts()
            tp = TextPath((0, 0), s, size=size, prop=FontProperties(fname=bold if mode == 'bold' else reg))
        _cache[key] = tp
    return _cache[key]


def extent(s, size, mode='math'):
    """(xmin, ymin, xmax, ymax) in px, y up, baseline 0"""
    p = text_path(s, size, mode)
    if len(p.vertices) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    b = p.get_extents()
    return (b.x0, b.y0, b.x1, b.y1)


def _x0(s, size, mode, x, anchor):
    xa, _, xb, _ = extent(s, size, mode)
    return {'start': x - xa, 'middle': x - (xa + xb) / 2, 'end': x - xb}[anchor]


def svg_d(s, size, x, y, anchor='start', mode='math'):
    """SVG path data of the text with its baseline at y (y down), anchored at x"""
    p = text_path(s, size, mode)
    ox = _x0(s, size, mode, x, anchor)
    out = []
    v = p.vertices; c = p.codes
    i = 0
    f = lambda q: '%.2f %.2f' % (ox + q[0], y - q[1])
    while i < len(v):
        k = c[i]
        if k == 1:
            out.append('M' + f(v[i])); i += 1
        elif k == 2:
            out.append('L' + f(v[i])); i += 1
        elif k == 3:
            out.append('Q' + f(v[i]) + ' ' + f(v[i + 1])); i += 2
        elif k == 4:
            out.append('C' + f(v[i]) + ' ' + f(v[i + 1]) + ' ' + f(v[i + 2])); i += 3
        elif k == 79:
            out.append('Z'); i += 1
        else:
            i += 1
    return ''.join(out)


def polygons(s, size, x, y, anchor='start', mode='math', res=40.0, tol=0.03):
    """flattened glyph outlines as hole-free closed polygons (y down), baseline at y;
    vertices are thinned to within `tol` px of the true outline"""
    from matplotlib.transforms import Affine2D
    from . import poly
    p = text_path(s, size, mode)
    ox = _x0(s, size, mode, x, anchor)
    rings = []
    for pg in p.to_polygons(Affine2D().scale(res), closed_only=False):
        r = [(ox + float(a) / res, y - float(b) / res) for a, b in pg]
        r = poly.simplify(poly.dedupe(r), tol)
        if len(r) >= 3:
            rings.append(r)
    return poly.solid_rings(rings)
