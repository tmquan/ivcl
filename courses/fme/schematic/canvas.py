"""Plotly renderer: the figures of `mpl.py` on an interactive canvas.

A figure has the same three columns as the matplotlib one (the names of a circuit, the drawing
with BJTs, the drawing with MOSFETs) and the same geometry, taken from `mpl.py`.  The artwork is
one SVG picture, made from the scenes exactly as the SVG export makes it, so wires, symbols and
the Computer Modern labels look as before and grow and shrink together with the page.  On top of
it lies one invisible item per circuit element (transistor, resistor, load, ...):

    pointer over an element   it lights up, and a tooltip gives its name, its kind and its value
    click                     the element stays selected; several can be; a second click releases it

Wires, dots, terminals, grounds and supplies only route and are not selectable.  Names and column
heads are set in the text font of the site, labels and tooltips of elements in Computer Modern.

    from schematic import show
    show(inverter)                                   # in a notebook: a Plotly canvas
    show(amplifier, values={'R_C': '500 Ω', 'Q_1': 'β = 100'})

Outside a notebook (no IPython kernel) `show` and `symbol_sheet` fall back to `mpl.py`, so scripts
that collect matplotlib figures keep working.
"""
import base64
import re
from xml.sax.saxutils import escape

from . import symbols as S
from . import scene as SC
from . import tex as T
from .scene import Scene
from .ckt import BJT, MOSFET

# the layout of a figure, shared with the matplotlib renderer (px)
LABEL_W, PANEL_W = 170, 340    # three columns of 20 % | 40 % | 40 %
HEAD_H, MARGIN = 34, 12
LABEL, LINE, PARA = 16, 21, 9  # label column: font size, line height, space between two names
RULE = '#d1d1d1'
GREY = '#737373'               # a cell without a drawing

TEXT_FONT = '"NVIDIA Sans", "NVIDIASans", -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif'
MATH_FONT = '"Computer Modern Serif", "Latin Modern Roman", "CMU Serif", serif'
SANS = 1.16                    # a line of sans serif text is at most this much wider than in Computer Modern
PAD = 4                        # px around the box of an element: the area that answers to the pointer

KINDS = {'npn': 'npn transistor', 'pnp': 'pnp transistor',
         'nmos': 'NMOS transistor', 'pmos': 'PMOS transistor',
         'nmos_b': 'NMOS transistor with its bulk terminal', 'pmos_b': 'PMOS transistor with its bulk terminal',
         'nmos_na': 'NMOS transistor', 'pmos_na': 'PMOS transistor', 'nmos_thick': 'thick-oxide NMOS transistor',
         'opamp': 'op amp', 'resistor': 'resistor', 'capacitor': 'capacitor', 'inductor': 'inductor',
         'diode': 'diode', 'zener': 'Zener diode', 'vsource': 'voltage source', 'isource': 'current source',
         'battery': 'battery', 'load': 'load'}


TRANSISTORS = {'npn', 'pnp', 'nmos', 'pmos', 'nmos_b', 'pmos_b', 'nmos_na', 'pmos_na', 'nmos_thick'}


def markup(label):
    """a label in the mini-markup of tex.py as the small HTML that Plotly understands: Q_1, R_C, (W/L)_2"""
    if not label:
        return ''
    if label.startswith('~'):
        return escape(label[1:])
    if label.startswith('$'):
        plain = re.sub(r'\\(?:mathrm|text)\{([^}]*)\}', r'\1', label.strip('$'))
        plain = re.sub(r'\\[a-zA-Z]+', '', plain).replace('{', '').replace('}', '')
        return '<i>%s</i>' % escape(plain)
    out = []
    for token in label.lstrip('!').split(' '):
        m = re.match(r'^(.*?)_(\{[^}]*\}|[A-Za-z0-9,]+)(.*)$', token)
        if not m:
            out.append('<i>%s</i>' % escape(token))
            continue
        base, sub, rest = m.groups()
        sub = sub.strip('{}')
        sub = '<i>%s</i>' % escape(sub) if len(sub) == 1 and sub.isalpha() else escape(sub)
        out.append('<i>%s</i><sub>%s</sub>%s' % (escape(base), sub, escape(rest)))
    return ' '.join(out)


def html(latex):
    """a short piece of LaTeX maths, such as the given data of a problem, as the small HTML that Plotly
    understands:  5\\times10^{-17}\\,\\mathrm{A}  ->  5×10<sup>−17</sup> A"""
    s = latex.replace('\\,', ' ').replace('\\%', '%').replace('\\;', ' ').replace('\\mu ', '\\mu')
    for a, b in (('\\times', '×'), ('\\Omega', 'Ω'), ('\\mu', 'μ'), ('\\beta', 'β'), ('\\lambda', 'λ'), ('\\pi', 'π'),
                 ('\\parallel', ' ∥ '), ('\\Delta', 'Δ'), ('\\prime', '′'), ('\\infty', '∞'), ('\\approx', '≈'),
                 ('\\cdot', '·'), ('\\left', ''), ('\\right', '')):
        s = s.replace(a, b)
    s = escape(s).replace('-', '−')
    s = re.sub(r'\^\{([^{}]*)\}', r'<sup>\1</sup>', s)
    s = re.sub(r'\^(.)', r'<sup>\1</sup>', s)
    s = re.sub(r'_\{([^{}]*)\}', r'<sub>\1</sub>', s)
    s = re.sub(r'_(\w)', r'<sub>\1</sub>', s)
    s = re.sub(r'\\(?:mathrm|text)\{([^{}]*)\}', r'\1', s)
    return s.replace('{', '').replace('}', '')


class Item:
    """One circuit element of a drawing: where it is on the figure, and what its tooltip says."""

    def __init__(self, inst, box, name, value=None):
        self.key, self.box, self.name, self.value = inst.key, box, name, value

    @property
    def kind(self):
        return KINDS.get(self.key, self.key)

    def text(self):
        head = '%s: %s' % (markup(self.name), self.kind) if self.name else self.kind[:1].upper() + self.kind[1:]
        size = getattr(self, 'size', None)
        lines = [head] + ([markup(size)] if size else []) + ([self.value] if self.value else [])
        return '<br>'.join(lines)


class Cell:
    """One drawing in its place on the figure: the scene, and the point of the figure where the
    point (x0, y0) of the scene lies."""

    def __init__(self, scene, x, y, x0, y0, values=None):
        self.scene, self.dx, self.dy = scene, x - x0, y - y0
        self.values = values or {}

    def svg(self):
        return ('<g transform="translate(%.2f,%.2f)" stroke="#000" stroke-width="%.3f" stroke-linecap="butt" '
                'stroke-linejoin="miter" stroke-miterlimit="10" fill="none">%s</g>'
                % (self.dx, self.dy, SC.WIRE_W, self.scene.svg_parts()))

    def label(self, inst):
        """the name written beside an element: the one the circuit recorded; for an element that was
        placed without one, the nearest label that reads like the name of a passive element (R_1, C_L)"""
        if getattr(inst, 'label', None):
            return inst.label
        if inst.key in TRANSISTORS:
            return None
        cx, cy = inst.x + inst.w / 2, inst.y + inst.h / 2
        best, near = None, 46.0 + max(inst.w, inst.h) / 2
        for t in self.scene.texts:
            if t[6] not in (None, SC.BLUE, SC.BLACK) or t[5] != 'math' or not re.match(r'^[RCLZ]_', t[0]):
                continue
            x0, y0, x1, y1 = self.scene.text_box(t)
            dx = max(x0 - cx, 0, cx - x1); dy = max(y0 - cy, 0, cy - y1)
            d = (dx * dx + dy * dy) ** 0.5
            if d < near:
                best, near = t[0], d
        return best

    def items(self):
        out = []
        for inst in self.scene.insts:
            if inst.key not in SC.ELEMENTS:
                continue
            name = self.label(inst)
            box = (inst.x + self.dx - PAD, inst.y + self.dy - PAD, inst.x + inst.w + self.dx + PAD, inst.y + inst.h + self.dy + PAD)
            item = Item(inst, box, name, self.values.get(name or inst.key))      # a device without a name: by its symbol
            item.size = getattr(inst, 'size', None)
            out.append(item)
        return out


class Canvas:
    """A figure on a Plotly canvas: an SVG picture of W x H px with the items of its cells on top."""

    count = 0                  # canvases drawn so far in this kernel: gives every canvas its id

    # runs in the page: the element under the pointer lights up; a click keeps it lit, a second click releases it
    script = """
    var gd = document.getElementById('{plot_id}');
    var chosen = {};
    var paint = function (k, state) {
        var trace = gd.querySelectorAll('.scatterlayer .trace')[k];
        if (!trace) { return; }
        var fill = trace.querySelector('.js-fill'), line = trace.querySelector('.js-line');
        var look = {off: ['rgba(11,85,217,0)', 'rgba(11,85,217,0)'], over: ['rgba(11,85,217,0.10)', 'rgba(11,85,217,0.55)'],
                    on: ['rgba(224,0,0,0.12)', 'rgba(224,0,0,0.85)']}[state];
        if (fill) { fill.style.fill = look[0]; fill.style.fillOpacity = 1; }
        if (line) { line.style.stroke = look[1]; line.style.strokeOpacity = 1; }
    };
    var rest = function (k) { paint(k, chosen[k] ? 'on' : 'off'); };
    var over = null;
    gd.on('plotly_hover', function (event) {
        var k = event.points[0].curveNumber;
        if (over !== null && over !== k) { rest(over); }
        over = k;
        if (!chosen[k]) { paint(k, 'over'); }
    });
    gd.on('plotly_unhover', function () {
        if (over !== null) { rest(over); over = null; }
    });
    gd.on('plotly_click', function (event) {
        var k = event.points[0].curveNumber;
        chosen[k] = !chosen[k];
        paint(k, chosen[k] ? 'on' : 'over');
    });
    gd.on('plotly_afterplot', function () {
        for (var k in chosen) { rest(+k); }
    });
    """

    def __init__(self, width, height, zoom=1.0):
        self.W, self.H, self.zoom = width, height, zoom
        self.parts, self.cells = [], []

    # ---- the picture ----------------------------------------------------
    def rule(self, x0, y0, x1, y1):
        self.parts.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" stroke-width="0.8"/>' % (x0, y0, x1, y1, RULE))

    def text(self, s, x, y, anchor='middle', size=LABEL, bold=False, color='#000'):
        """a line of regular text, in the text font of the site, centred on the height y"""
        self.parts.append('<text x="%.2f" y="%.2f" text-anchor="%s" dominant-baseline="central" font-family=\'%s\' '
                          'font-size="%g" font-weight="%s" fill="%s">%s</text>'
                          % (x, y, anchor, TEXT_FONT, size, 700 if bold else 400, color, escape(s)))

    def place(self, cell):
        self.parts.append(cell.svg())
        self.cells.append(cell)

    def svg(self):
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" viewBox="0 0 %.2f %.2f">'
                '<rect width="100%%" height="100%%" fill="#fff"/>%s</svg>' % (self.W, self.H, self.W, self.H, ''.join(self.parts)))

    # ---- the canvas -----------------------------------------------------
    def figure(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        for cell in self.cells:
            for item in cell.items():
                x0, y0, x1, y1 = item.box
                fig.add_trace(go.Scatter(
                    x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0], mode='lines', fill='toself', hoveron='fills',
                    fillcolor='rgba(11,85,217,0)', line=dict(color='rgba(11,85,217,0)', width=1.5, shape='linear'),
                    name=re.sub(r'<[^>]+>', '', item.text().split('<br>')[0]), text=item.text(), hoverinfo='text',
                    showlegend=False))
        picture = 'data:image/svg+xml;base64,' + base64.b64encode(self.svg().encode('utf-8')).decode('ascii')
        axis = dict(visible=False, fixedrange=True, showgrid=False, zeroline=False)
        fig.update_layout(
            images=[dict(source=picture, xref='x', yref='y', x=0, y=0, sizex=self.W, sizey=self.H,
                         xanchor='left', yanchor='top', sizing='stretch', layer='below')],
            xaxis=dict(range=[0, self.W], **axis), yaxis=dict(range=[self.H, 0], **axis),
            margin=dict(l=0, r=0, t=0, b=0), autosize=True, template='simple_white',
            paper_bgcolor='white', plot_bgcolor='white', dragmode=False, hovermode='closest', hoverdistance=1,
            hoverlabel=dict(bgcolor='white', bordercolor='#0b55d9', font=dict(family=MATH_FONT, size=16, color='black'), align='left'))
        return fig

    def html(self):
        """the canvas as a piece of HTML.  It keeps the proportions of the figure at every width of the
        page; the first canvas of a page loads Plotly from its CDN, as the DSP figures do."""
        Canvas.count += 1
        body = self.figure().to_html(
            full_html=False, include_plotlyjs='cdn' if Canvas.count == 1 else False,
            div_id='schematic-%d' % Canvas.count, post_script=self.script,
            default_width='100%', default_height='100%',
            config=dict(responsive=True, displayModeBar=False, scrollZoom=False, doubleClick=False))
        return ('<div class="schematic-canvas" style="width:100%%; max-width:%.0fpx; aspect-ratio:%.2f / %.2f;">%s</div>'
                % (self.W * self.zoom * 1.5, self.W, self.H, body))

    def draw(self):
        from IPython.display import HTML, display
        display(HTML(self.html()))


# ---------------------------------------------------------------- figures

def _wrap(text, size, width):
    """break `text` into lines of at most `width` px, measured in Computer Modern and allowing for a wider font"""
    lines, cur = [], ''
    for word in text.split(' '):
        trial = (cur + ' ' + word).strip()
        x0, _, x1, _ = T.extent(trial, size, 'text')
        if cur and (x1 - x0) * SANS > width:
            lines.append(cur); cur = word
        else:
            cur = trial
    return lines + [cur]


def _label(titles):
    """label column of one row: the names of the circuit, the BJT name first, as [(text, y offset)], height"""
    out, y = [], 0
    for name in [t for t in dict.fromkeys(titles) if t]:              # one entry if both names are the same
        for line in _wrap(name, LABEL, LABEL_W - 2 * MARGIN):
            out.append((line, y)); y += LINE
        y += PARA
    return out, y - PARA


def _reach(sc):
    b = sc.bounds()
    return sc.axis - b[0], b[2] - sc.axis


def grid(rows, heads=('BJT', 'MOSFET'), zoom=1.0, values=None):
    """rows = [(scenes, names), ...]: one row per circuit, with one scene and one name per technology.
    The geometry is that of `mpl.grid`: every drawing has its axis on the centre line of its cell, the
    two drawings of a row sit at the same height, and a scene given as None gives a cell that says
    "No ... version".  values = {label: text} for the tooltips, or one such dict per technology."""
    values = values or {}
    values = [values] * len(heads) if isinstance(values, dict) else list(values)
    boxes, labels = [], []
    for scenes, titles in rows:
        b = [sc.bounds() for sc in scenes if sc is not None]
        boxes.append((min(q[0] for q in b), min(q[1] for q in b), max(q[2] for q in b), max(q[3] for q in b)))
        labels.append(_label(titles))
    n = len(heads)
    far = max(max(_reach(sc)) for scenes, _ in rows for sc in scenes if sc is not None)
    pw = max(PANEL_W, 2 * (far + MARGIN))                             # wider only if a drawing reaches too far
    lw = pw * LABEL_W / PANEL_W
    hd = [q[3] - q[1] + 2 * MARGIN for q in boxes]                    # height of the drawings of a row
    hs = [max(d, lab[1] + 2 * MARGIN, LINE + 2 * MARGIN) for d, lab in zip(hd, labels)]
    canvas = Canvas(lw + n * pw, HEAD_H + sum(hs), zoom)
    for j, head in enumerate(heads):                                  # column heads
        canvas.text(head, lw + j * pw + pw / 2, HEAD_H / 2, size=18, bold=True)
    y = HEAD_H
    for (scenes, titles), q, h, d, (lines, lh) in zip(rows, boxes, hs, hd, labels):
        canvas.rule(0, y, canvas.W, y)
        for text, dy in lines:                                        # first column: the names of the circuit
            canvas.text(text, MARGIN, y + (h - lh) / 2 + dy + 0.5 * LINE, 'start')
        for j, sc in enumerate(scenes):
            xc = lw + j * pw + pw / 2
            if sc is None:                                            # no such version: the cell says so
                canvas.text('No %s version' % heads[j], xc, y + h / 2, color=GREY)
                continue
            cy = (q[1] + q[3]) / 2                                    # both drawings of a row at one height
            canvas.place(Cell(sc, xc, y, sc.axis, cy - d / 2, values[j]))
        y += h
    for j in range(n):
        canvas.rule(lw + j * pw, HEAD_H, lw + j * pw, canvas.H)
    return canvas


def _notebook():
    """True inside a running IPython kernel"""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def show(*circuits, notes=None, zoom=1.0, devices=(BJT, MOSFET), values=None):
    """Three columns: the names of a circuit, the circuit drawn with BJTs, and drawn with MOSFETs.
    One row per circuit.  A circuit whose docstring names one version as "-" exists in the other
    technology only; that cell says so.  Every drawing is centred on its axis: the main route of its
    transistors, or the middle between them.

    values   what the tooltip of an element adds to its name and kind, by the label of the element:
             {'R_C': '500 Ω', 'Q_1': 'β = 100'}; or a pair of such dicts, (BJT, MOSFET).  A transistor
             without a name is listed under its symbol: 'npn', 'pnp', 'nmos', 'pmos'.
    notes    for every circuit a text for its two drawings: a pair (BJT, MOSFET), or one text for both.
             They are set as a table under the canvas, in the type of the page."""
    from . import mpl
    if not _notebook():                                               # a script: the matplotlib figure
        return mpl.show(*circuits, notes=notes, zoom=zoom, devices=devices)
    rows, texts = [], []
    for i, f in enumerate(circuits):
        nm = mpl.names(f)
        rows.append((tuple(mpl.build(f, d) if nm[j] else None for j, d in enumerate(devices)), nm))
        note = notes[i] if notes else ''
        texts.append((nm, (note, note) if isinstance(note, str) else tuple(note)))
    canvas = grid(rows, tuple(d.__name__ for d in devices), zoom, values)
    html = canvas.html()
    if notes:
        html += _notes(texts, tuple(d.__name__ for d in devices))
    from IPython.display import HTML, display
    display(HTML(html))


def _notes(texts, heads):
    """the notes of a figure as a table of the three columns of the figure; $maths$ is left to MathJax"""
    maths = lambda s: re.sub(r'\$([^$]+)\$', lambda m: r'\(%s\)' % escape(m.group(1)), s)
    rows = ''.join('<tr><th style="width:20%%;text-align:left">%s</th>%s</tr>'
                   % (escape(' / '.join(n for n in dict.fromkeys(names) if n)),
                      ''.join('<td style="width:40%%">%s</td>' % maths(t) for t in note))
                   for names, note in texts)
    return ('<table class="tex2jax_process" style="width:100%%;table-layout:fixed"><thead><tr><th></th>%s</tr></thead>'
            '<tbody>%s</tbody></table>' % (''.join('<th>%s</th>' % h for h in heads), rows))


def symbol_sheet(keys=None, cols=6, zoom=1.0, compact=False):
    """the symbols of the library with their names (or the symbols listed in `keys`), on one canvas as
    wide as the circuit figures"""
    from . import mpl
    if not _notebook():
        return mpl.symbol_sheet(keys, cols, zoom, compact)
    sc = Scene('symbols')
    items = [(k, t, s) for k, t, s in S.CATALOG if (s['in_book'] if keys is None else k in keys)]
    x = y = 0
    names = []
    for n, (key, title, sym) in enumerate(items):
        sc.add(key, x + (100 - sym['w']) / 2, y + (100 - sym['h']) / 2)
        names.append((title.split(' (')[0], x + 50, y + 118))
        x += 130
        if (n + 1) % cols == 0:
            x = 0; y += 160
    margin = 14
    x0, y0, x1, y1 = sc.bounds()
    y1 = max(y1, max(ny for _, _, ny in names) + 10)                  # room for the last line of names
    canvas = Canvas(max(x1 - x0 + 2 * margin, LABEL_W + 2 * PANEL_W), y1 - y0 + 2 * margin, zoom)
    cell = Cell(sc, canvas.W / 2, margin, (x0 + x1) / 2, y0)
    canvas.place(cell)
    for name, nx, ny in names:                                        # regular text: in the text font of the site
        canvas.text(name, nx + cell.dx, ny + cell.dy, size=12)
    canvas.draw()
