"""Matplotlib renderer: scenes -> figures with a BJT column and a MOSFET column.

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
    for i in sc.insts:                                                # symbol bodies
        tf = _inst_tf(i) + ax.transData
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
    for x, y, op in sc.dots:                                          # terminals and node dots
        if op:
            ax.add_patch(Circle((x, y), SC.RING_R, fc='white', ec=color, lw=SC.RING_W * k, zorder=5))
        else:
            ax.add_patch(Circle((x, y), SC.DOT_R, fc=color, ec='none', lw=0, zorder=5))
    for txt, x, y, a, sz, mode in sc.texts:                           # labels
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


PANEL_W = 410                  # px; one width for every panel, so all figures share one scale
HEAD_H, TITLE_H, MARGIN, GAP = 34, 30, 12, 30
RULE = '0.82'


def grid(rows, heads=('BJT', 'MOSFET'), zoom=1.0, panel_w=PANEL_W):
    """rows = [((scene, scene), (title, title)), ...]: one row per circuit, one column per technology.
    Both panels of a row show the same window, so corresponding parts sit at identical positions.
    A scene given as None leaves its panel empty."""
    boxes = []
    for scenes, _ in rows:
        b = [sc.bounds() for sc in scenes if sc is not None]
        boxes.append((min(q[0] for q in b), min(q[1] for q in b), max(q[2] for q in b), max(q[3] for q in b)))
    n = len(heads)
    pw = max([panel_w] + [q[2] - q[0] + 2 * MARGIN for q in boxes])
    hs = [q[3] - q[1] + 2 * MARGIN for q in boxes]
    W = n * pw + (n - 1) * GAP
    H = HEAD_H + sum(TITLE_H + h for h in hs)
    fig = plt.figure(figsize=(W * zoom / 96.0, H * zoom / 96.0))
    k = PT * zoom
    reg, bold = T._fonts()
    for j, head in enumerate(heads):                                  # column heads
        fig.text((j * (pw + GAP) + pw / 2) / W, 1 - 0.5 * HEAD_H / H, head, ha='center', va='center',
                 fontproperties=FontProperties(fname=bold, size=18 * k))
    rule = lambda xs, ys: fig.add_artist(Line2D(xs, ys, transform=fig.transFigure, color=RULE, lw=0.8))
    y = HEAD_H
    ux = (min(q[0] for q in boxes) + max(q[2] for q in boxes)) / 2      # one centre line for all rows, if they fit
    fits = max(q[2] for q in boxes) - min(q[0] for q in boxes) + 2 * MARGIN <= pw
    for r, ((scenes, titles), q, h) in enumerate(zip(rows, boxes, hs)):
        rule([0, 1], [1 - y / H] * 2)
        cx = ux if fits else (q[0] + q[2]) / 2
        for j, sc in enumerate(scenes):
            if sc is None:                                            # no such version: the panel stays empty
                continue
            x = j * (pw + GAP)
            fig.text((x + pw / 2) / W, 1 - (y + 0.6 * TITLE_H) / H, titles[j], ha='center', va='center',
                     fontproperties=FontProperties(fname=reg, size=16 * k))
            ax = fig.add_axes([x / W, 1 - (y + TITLE_H + h) / H, pw / W, h / H])
            ax.set_xlim(cx - pw / 2, cx + pw / 2)
            ax.set_ylim(q[3] + MARGIN, q[1] - MARGIN)
            ax.axis('off')
            draw(ax, sc, k)
        y += TITLE_H + h
    for j in range(1, n):
        xg = (j * (pw + GAP) - GAP / 2) / W
        rule([xg, xg], [0.0, 1 - HEAD_H / H])
    return fig


def show(*circuits, zoom=1.0, devices=(BJT, MOSFET)):
    """Two columns: every circuit drawn with BJTs on the left and with MOSFETs on the right.
    One row per circuit, so  show(a, b)  gives a 2 x 2 figure.  A circuit whose docstring names
    one version as "-" exists in the other technology only, and that panel stays empty."""
    rows = []
    for f in circuits:
        nm = names(f)
        rows.append((tuple(build(f, d) if nm[j] else None for j, d in enumerate(devices)), nm))
    grid(rows, tuple(d.__name__ for d in devices), zoom)


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
