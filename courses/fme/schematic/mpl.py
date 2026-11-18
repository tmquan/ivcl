"""Matplotlib renderer: scenes -> figures with a label column, a BJT column and a MOSFET column.

Everything is drawn in scene pixels; a figure is sized so that one scene px is exactly
`zoom`/96 inch, which keeps stroke weights, symbols and labels in their true proportions.
Labels are set in Computer Modern through matplotlib's mathtext (no TeX install needed).
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Circle
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties

from . import symbols as S
from . import scene as SC
from . import tex as T
from .scene import Scene
from .ckt import Ckt, BJT, MOSFET

PT = 0.75                      # points per scene px at zoom 1 (96 px per inch)
_HA = {'start': 'left', 'middle': 'center', 'end': 'right'}


def _inst_tf(i):
    """symbol coordinates -> scene coordinates (same convention as the SVG export)"""
    w, h = i.sym['w'], i.sym['h']
    t = Affine2D()
    if i.mirror:
        t = t.scale(-1, 1).translate(w, 0)
    r = i.rot % 4
    if r == 1:
        t = t.rotate_deg(90).translate(h, 0)
    elif r == 2:
        t = t.rotate_deg(180).translate(w, h)
    elif r == 3:
        t = t.rotate_deg(270).translate(0, w)
    return t.translate(i.x, i.y)


def draw(ax, sc, k=PT, color='k'):
    """draw a Scene on `ax` (data units = scene px, y axis pointing down). k = points per px."""
    for w in sc.connectors():                                         # continuous connectors
        ax.add_line(Line2D([p[0] for p in w], [p[1] for p in w], lw=SC.WIRE_W * k, color=color,
                           solid_capstyle='butt', solid_joinstyle='miter', zorder=2))
    black = color
    for i in sc.insts:                                                # symbol bodies
        tf = _inst_tf(i) + ax.transData
        color = i.color or black
        for p in i.sym['body']:
            kind = p[0]
            if kind == 'line':
                ax.add_line(Line2D([q[0] for q in p[1]], [q[1] for q in p[1]], lw=p[2] * k, color=color,
                                   solid_capstyle=p[3], solid_joinstyle=p[4], transform=tf, zorder=3))
            elif kind == 'dash':
                ax.add_line(Line2D([q[0] for q in p[1]], [q[1] for q in p[1]], lw=p[2] * k, color=color,
                                   linestyle=(0, (p[3] / p[2], p[4] / p[2])), dash_capstyle='butt', transform=tf, zorder=3))
            elif kind == 'poly':
                ax.add_patch(Polygon(p[1], closed=True, fc=color, ec=color, lw=p[2] * k, joinstyle=p[3],
                                     transform=tf, zorder=3))
            elif kind == 'circ':
                ax.add_patch(Circle((p[1], p[2]), p[3], fc=color if p[5] else 'white', ec=color,
                                    lw=0 if p[5] else p[4] * k, transform=tf, zorder=3))
    color = black
    for x, y, op in sc.dots:                                          # terminals and node dots
        if op:
            ax.add_patch(Circle((x, y), SC.RING_R, fc='white', ec=color, lw=SC.RING_W * k, zorder=5))
        else:
            ax.add_patch(Circle((x, y), SC.DOT_R, fc=color, ec='none', lw=0, zorder=5))
    for txt, x, y, a, sz, mode, own in sc.texts:                      # labels
        color = own or black
        if mode == 'math':
            ax.text(x, y + SC.BASE * sz, '$%s$' % T.tex(txt), fontsize=sz * k, ha=_HA[a], va='baseline',
                    color=color, math_fontfamily='cm', zorder=6)
        else:
            reg, bold = T._fonts()
            ax.text(x, y + SC.BASE * sz, txt, fontsize=sz * k, ha=_HA[a], va='baseline', color=color,
                    fontproperties=FontProperties(fname=bold if mode == 'bold' else reg, size=sz * k), zorder=6)


def panels(scenes, titles=None, zoom=1.0, margin=14, gap=30, min_w=230, same_window=True):
    """one figure, one panel per scene, all at the same scale.  With same_window the panels
    show the same region, so corresponding parts sit at identical positions."""
    boxes = [sc.bounds() for sc in scenes]
    if same_window:
        b = (min(q[0] for q in boxes), min(q[1] for q in boxes), max(q[2] for q in boxes), max(q[3] for q in boxes))
        boxes = [b] * len(scenes)
    th = 34 if titles else 0
    ws = [max(q[2] - q[0] + 2 * margin, min_w) for q in boxes]
    h = max(q[3] - q[1] for q in boxes) + 2 * margin
    W = sum(ws) + gap * (len(scenes) - 1)
    H = h + th
    fig = plt.figure(figsize=(W * zoom / 96.0, H * zoom / 96.0))
    k = PT * zoom
    x = 0.0
    for j, (sc, q) in enumerate(zip(scenes, boxes)):
        ax = fig.add_axes([x / W, 0, ws[j] / W, h / H])
        cx = (q[0] + q[2]) / 2
        ax.set_xlim(cx - ws[j] / 2, cx + ws[j] / 2)
        ax.set_ylim(q[3] + margin, q[3] + margin - h)
        ax.axis('off')
        draw(ax, sc, k)
        if titles:
            fig.text((x + ws[j] / 2) / W, 1 - 0.5 * th / H, r'$\mathrm{%s}$' % titles[j].replace(' ', r'\ '),
                     ha='center', va='center', fontsize=15 * k, math_fontfamily='cm')
        if j:
            xg = (x - gap / 2) / W
            fig.add_artist(Line2D([xg, xg], [0.03, 0.97], transform=fig.transFigure, color='0.82', lw=0.8))
        x += ws[j] + gap
    return fig


def build(circuit, device):
    """run one circuit function with the transistor class `device` (BJT or MOSFET) -> Scene"""
    sc = Scene(getattr(circuit, '__name__', ''))
    circuit(Ckt(sc, device))
    return sc


def names(circuit):
    """(BJT name, MOSFET name) from the first docstring line:  "BJT name | MOSFET name",  or one name for both.
    A name "-" means that there is no such version; it is returned as ''."""
    doc = (circuit.__doc__ or circuit.__name__.replace('_', ' ')).strip().splitlines()[0]
    parts = [p.strip() for p in doc.split('|')]
    return tuple('' if p == '-' else p for p in (parts[0], parts[-1]))


LABEL_W, PANEL_W = 170, 340    # px: three columns of 20 % | 40 % | 40 %, the same in every figure
HEAD_H, MARGIN = 34, 12
LABEL, LINE, PARA = 16, 21, 9  # label column: font size, line height, space between two names (px)
NOTE, NOTE_W, NOTE_LEAD = 15, 300, 5   # note under a drawing: font size, widest line, space between lines (px)
RULE = '0.82'
GREY = '0.45'                  # grey of a cell without a drawing


def _wrap(text, size, width):
    """break `text` into lines of at most `width` px"""
    lines, cur = [], ''
    for word in text.split(' '):
        trial = (cur + ' ' + word).strip()
        x0, _, x1, _ = T.extent(trial, size, 'text')
        if cur and x1 - x0 > width:
            lines.append(cur); cur = word
        else:
            cur = trial
    return lines + [cur]


def _label(titles):
    """label column of one row: the names of the circuit, the BJT name first, as [(text, y offset)]"""
    names = [t for t in dict.fromkeys(titles) if t]                  # one entry if both names are the same
    out, y = [], 0
    for k, name in enumerate(names):
        for line in _wrap(name, LABEL, LABEL_W - 2 * MARGIN):
            out.append((line, y)); y += LINE
        y += PARA
    return out, y - PARA


def _words(text):
    """words of a sentence; a formula between $ signs stays in one piece"""
    out, cur, maths = [], '', False
    for ch in text:
        if ch == ' ' and not maths:
            if cur:
                out.append(cur)
            cur = ''
        else:
            maths ^= ch == '$'
            cur += ch
    return out + ([cur] if cur else [])


def _note(text):
    """a note set under a drawing: [(line, baseline below the top of the note)], height.
    Lines are at most NOTE_W wide; a line with a fraction in it is as tall as it needs to be."""
    if not text:
        return [], 0
    lines, cur = [], ''
    for word in _words(text):
        trial = (cur + ' ' + word).strip()
        x0, _, x1, _ = T.extent(trial, NOTE, 'mixed')
        if cur and x1 - x0 > NOTE_W:
            lines.append(cur); cur = word
        else:
            cur = trial
    out, y = [], 0.0
    for line in lines + [cur]:
        _, lo, _, hi = T.extent(line, NOTE, 'mixed')
        up, down = max(hi, 0.72 * NOTE), max(-lo, 0.22 * NOTE)
        out.append((line, y + up)); y += up + down + NOTE_LEAD
    return out, y - NOTE_LEAD


def grid(rows, heads=('BJT', 'MOSFET'), zoom=1.0, scale=1.0):
    """rows = [(scenes, names, notes), ...]: one row per circuit, with one scene, one name and one note
    per technology.  Three columns: the names of the circuit, then one cell per technology that holds
    the drawing, centred, and its note right under it.  A scene given as None gives a cell that says
    "No ... version"; its note says why.  scale < 1 draws the circuits smaller, so that a circuit of
    many transistors fits its cell; names and notes keep their size."""
    boxes, labels, notes = [], [], []
    for scenes, titles, texts in rows:
        b = [sc.bounds() for sc in scenes if sc is not None]
        boxes.append((min(q[0] for q in b), min(q[1] for q in b), max(q[2] for q in b), max(q[3] for q in b)))
        labels.append(_label(titles))
        notes.append([_note(t) for t in texts])
    n = len(heads)
    pw = max([PANEL_W] + [scale * (q[2] - q[0]) + 2 * MARGIN for q in boxes])
    lw = pw * LABEL_W / PANEL_W
    hd = [scale * (q[3] - q[1]) + 2 * MARGIN for q in boxes]          # height of the drawings of a row
    hs = []
    for (scenes, _, _), d, row, lab in zip(rows, hd, notes, labels):
        under = max([h for sc, (_, h) in zip(scenes, row) if sc is not None] + [0])     # notes under drawings
        alone = max([LINE + PARA + h for sc, (_, h) in zip(scenes, row) if sc is None] + [0])   # a cell of text only
        hs.append(max(d + (under + MARGIN if under else 0), alone + 2 * MARGIN, lab[1] + 2 * MARGIN))
    W = lw + n * pw
    H = HEAD_H + sum(hs)
    fig = plt.figure(figsize=(W * zoom / 96.0, H * zoom / 96.0))
    k = PT * zoom
    reg, bold = T._fonts()
    roman = lambda size: FontProperties(fname=reg, size=size * k, math_fontfamily='cm')
    for j, head in enumerate(heads):                                  # column heads
        fig.text((lw + j * pw + pw / 2) / W, 1 - 0.5 * HEAD_H / H, head, ha='center', va='center',
                 fontproperties=FontProperties(fname=bold, size=18 * k))
    rule = lambda xs, ys: fig.add_artist(Line2D(xs, ys, transform=fig.transFigure, color=RULE, lw=0.8))
    y = HEAD_H
    for (scenes, titles, texts), h, d, (lines, lh), cells in zip(rows, hs, hd, labels, notes):
        rule([0, 1], [1 - y / H] * 2)
        for text, dy in lines:                                        # first column: the names of the circuit
            fig.text(MARGIN / W, 1 - (y + (h - lh) / 2 + dy + 0.5 * LINE) / H, text, ha='left', va='center',
                     fontproperties=roman(LABEL))
        for j, (sc, (note, nh)) in enumerate(zip(scenes, cells)):
            xc = lw + j * pw + pw / 2
            if sc is None:                                            # no such version: the cell says so, and why,
                top = y + (h - (LINE + PARA + nh)) / 2                # in the middle of the cell
                fig.text(xc / W, 1 - (top + LINE / 2) / H, 'No %s version' % heads[j], ha='center', va='center',
                         color=GREY, fontproperties=roman(LABEL))
                for line, base in note:
                    fig.text(xc / W, 1 - (top + LINE + PARA + base) / H, line, ha='center', va='baseline',
                             color=GREY, fontproperties=roman(NOTE))
                continue
            for line, base in note:                                   # the note, right under the drawing
                fig.text(xc / W, 1 - (y + d + base) / H, line, ha='center', va='baseline',
                         fontproperties=roman(NOTE))
            b = sc.bounds()                                           # each drawing is centred in its own cell
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            ax = fig.add_axes([(lw + j * pw) / W, 1 - (y + d) / H, pw / W, d / H])
            ax.set_xlim(cx - pw / 2 / scale, cx + pw / 2 / scale)
            ax.set_ylim(cy + d / 2 / scale, cy - d / 2 / scale)
            ax.axis('off')
            draw(ax, sc, k * scale)
        y += h
    for j in range(n):
        xg = (lw + j * pw) / W
        rule([xg, xg], [0.0, 1 - HEAD_H / H])
    return fig


def show(*circuits, notes=None, zoom=1.0, scale=1.0, devices=(BJT, MOSFET)):
    """Three columns: the names of a circuit, the circuit drawn with BJTs, and drawn with MOSFETs.
    One row per circuit.  notes holds, for every circuit, the text that goes right under its two
    drawings: a pair (BJT, MOSFET), or one text for both.  A circuit whose docstring names one
    version as "-" exists in the other technology only; that cell says so, and its note says why.
    scale < 1 draws the circuits of this figure smaller."""
    rows = []
    for i, f in enumerate(circuits):
        nm = names(f)
        note = notes[i] if notes else ''
        note = (note, note) if isinstance(note, str) else tuple(note)
        rows.append((tuple(build(f, d) if nm[j] else None for j, d in enumerate(devices)), nm, note))
    grid(rows, tuple(d.__name__ for d in devices), zoom, scale)


def symbol_sheet(keys=None, cols=6, zoom=1.0, compact=False):
    """the symbols printed in the book, with their names (or the symbols listed in `keys`)"""
    sc = Scene('symbols')
    items = [(k, t, s) for k, t, s in S.CATALOG if (s['in_book'] if keys is None else k in keys)]
    x = y = 0
    for n, (key, title, sym) in enumerate(items):
        sc.add(key, x + (100 - sym['w']) / 2, y + (100 - sym['h']) / 2)
        sc.text(title.split(' (')[0], x + 50, y + 118, 'middle', 11, 'text')
        x += 130
        if (n + 1) % cols == 0:
            x = 0; y += 160
    panels([sc], None, zoom)
