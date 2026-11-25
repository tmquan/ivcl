"""sigplot: every signal drawn twice, over continuous time and over discrete time.

    from sigplot import show
    def sine(t):
        '''Sine'''
        return np.sin(np.pi * t / 8)
    show(sine, notes=[(r'$x(t)=\\sin(\\omega_0t)$', r'$x[n]=\\sin(\\Omega_0n)$')])

A signal is a function of an array of time values.  `show` draws one row per signal: its
name, the curve x(t), and the sequence x[n] = x(nT).  Everything is laid out in pixels
(96 per inch), with the same three columns in every figure, and labels are set in Computer
Modern through matplotlib's mathtext, so no TeX installation is needed.
"""
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.textpath import TextPath

__all__ = ['show', 'step', 'ramp', 'rect', 'tri', 'sinc', 'delta']


# ---------------------------------------------------------------- elementary signals

def step(t):
    """unit step u(t): 1 for t >= 0, else 0"""
    return np.where(np.asarray(t, float) >= 0, 1.0, 0.0)


def ramp(t):
    """unit ramp r(t) = t u(t)"""
    return np.maximum(np.asarray(t, float), 0.0)


def rect(t):
    """unit rectangle: 1 for |t| <= 1/2, else 0"""
    return np.where(np.abs(np.asarray(t, float)) <= 0.5, 1.0, 0.0)


def tri(t):
    """unit triangle: 1 - |t| for |t| <= 1, else 0"""
    return np.maximum(1.0 - np.abs(np.asarray(t, float)), 0.0)


def sinc(t):
    """unit sinc: sin(pi t) / (pi t), with sinc(0) = 1"""
    return np.sinc(np.asarray(t, float))


def delta(t):
    """unit sample: 1 at t = 0, else 0.  Over continuous time the impulse itself is
    drawn as an arrow; give the signal the attribute `impulses = {0: 1}` (time: area)."""
    return np.where(np.asarray(t, float) == 0, 1.0, 0.0)


# ---------------------------------------------------------------- layout, in px

PT = 0.75                      # points per px at zoom 1 (96 px per inch)
LABEL_W, PANEL_W = 170, 340    # three columns of 20 % | 40 % | 40 %, the same in every figure
HEAD_H, MARGIN = 34, 12
PLOT_W, PLOT_H = 264, 140      # the plot area of one cell
TOP, BOTTOM = 26, 24           # room above it for the name of the signal, below it for the tick labels
LABEL, LINE, PARA = 16, 21, 9  # label column: font size, line height, space between two names
NOTE, NOTE_W, NOTE_LEAD = 15, 300, 5   # note under a plot: font size, widest line, space between lines
TICK, TICK_L = 12, 4           # tick labels: font size; tick length
RULE = '0.82'                  # rules between the cells
ZERO = '0.70'                  # the lines t = 0 and x = 0
BLUE = 'b'                     # the signal, and its envelope at half strength
GREY = '0.62'                  # the reference signal
CURVE_W, STEM_W, DOT_R, AXIS_W = 1.7, 1.2, 2.5, 0.9
HEADS = ('Continuous time', 'Discrete time')
_NICE = (1, 2, 4, 5, 8, 10, 20, 40, 50, 100)   # tick steps; the sample axis also counts in fours and eights


def _fonts():
    d = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
    return os.path.join(d, 'cmr10.ttf'), os.path.join(d, 'cmb10.ttf')


def _width(text, size):
    """width in px of a sentence with $maths$ in it"""
    tp = TextPath((0, 0), text, size=size, prop=FontProperties(fname=_fonts()[0], math_fontfamily='cm'))
    if len(tp.vertices) == 0:
        return 0.0, 0.0, 0.0
    b = tp.get_extents()
    return b.x1 - b.x0, b.y0, b.y1


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


def _wrap(text, size, width):
    """[(line, baseline below the top of the block)], height: lines of at most `width` px;
    a line with a fraction in it is as tall as it needs to be"""
    if not text:
        return [], 0.0
    lines, cur = [], ''
    for word in _words(text):
        trial = (cur + ' ' + word).strip()
        if cur and _width(trial, size)[0] > width:
            lines.append(cur); cur = word
        else:
            cur = trial
    out, y = [], 0.0
    for line in lines + [cur]:
        _, lo, hi = _width(line, size)
        up, down = max(hi, 0.72 * size), max(-lo, 0.22 * size)
        out.append((line, y + up)); y += up + down + NOTE_LEAD
    return out, y - NOTE_LEAD


def names(signal):
    """(continuous-time name, discrete-time name) from the first docstring line:
    "CT name | DT name", or one name for both"""
    doc = (signal.__doc__ or signal.__name__.replace('_', ' ')).strip().splitlines()[0]
    parts = [p.strip() for p in doc.split('|')]
    return parts[0], parts[-1]


def _ticks(lo, hi, most, nice=(1, 2, 5, 10, 20, 50, 100)):
    """the multiples of a nice step between lo and hi, at most `most` of them"""
    for s in nice:
        first, last = int(np.ceil(lo / s - 1e-9)), int(np.floor(hi / s + 1e-9))
        if last - first + 1 <= most:
            return [k * s for k in range(first, last + 1)]
    return [lo, hi]


def _num(v):
    return '$%g$' % v


# ---------------------------------------------------------------- one plot

def _plot(ax, k, x0, y0, signal, span, ylim, T, discrete, reference, symbol):
    """the plot of one cell, with the top left corner of its plot area at (x0, y0) px"""
    t0, t1 = span
    v0, v1 = ylim
    X = lambda t: x0 + (np.asarray(t, float) - t0) / (t1 - t0) * PLOT_W
    Y = lambda v: y0 + (v1 - np.asarray(v, float)) / (v1 - v0) * PLOT_H
    reg, _ = _fonts()
    roman = lambda size: FontProperties(fname=reg, size=size * k, math_fontfamily='cm')
    clip = Rectangle((x0 - 4, y0 - 4), PLOT_W + 8, PLOT_H + 8, transform=ax.transData)
    left, right, top, bottom = x0, x0 + PLOT_W, y0, y0 + PLOT_H

    def line(xs, ys, clipped=True, **kw):
        ln = Line2D(xs, ys, **kw)
        ax.add_line(ln)
        if clipped:
            ln.set_clip_path(clip)
        return ln

    # the lines x = 0 and t = 0
    if v0 < 0 < v1:
        line([left, right], [Y(0)] * 2, False, lw=AXIS_W * k, color=ZERO, zorder=1)
    if t0 < 0 < t1:
        line([X(0)] * 2, [top, bottom], False, lw=AXIS_W * k, color=ZERO, zorder=1)

    # the two axes, with their ticks and names
    line([left, left, right], [top, bottom, bottom], False, lw=AXIS_W * k, color='k', zorder=2,
         solid_joinstyle='miter', solid_capstyle='projecting')
    step_n = 1.0 if not discrete else T
    for n in _ticks(t0 / step_n, t1 / step_n, 9, _NICE if discrete else _NICE[:2] + _NICE[3:4] + _NICE[5:]):
        px = X(n * step_n)
        line([px, px], [bottom, bottom + TICK_L], False, lw=AXIS_W * k, color='k', zorder=2)
        ax.text(px, bottom + TICK_L + 3, _num(n), ha='center', va='top', fontproperties=roman(TICK), zorder=6)
    for v in _ticks(v0, v1, 7):
        py = Y(v)
        line([left - TICK_L, left], [py, py], False, lw=AXIS_W * k, color='k', zorder=2)
        ax.text(left - TICK_L - 3, py, _num(v), ha='right', va='center', fontproperties=roman(TICK), zorder=6)
    var = 'n' if discrete else 't'
    ax.text(right + 8, bottom, '$%s$' % var, ha='left', va='center', fontproperties=roman(NOTE), zorder=6)
    name = '$%s[n]$' % symbol if discrete else '$%s(t)$' % symbol
    ax.text(left, top - 9, name, ha='center', va='bottom', fontproperties=roman(NOTE), zorder=6)

    base = Y(min(max(0.0, v0), v1))
    tt = np.linspace(t0, t1, 2641)
    nn = np.arange(np.ceil(t0 / T - 1e-9), np.floor(t1 / T + 1e-9) + 1)

    def values(f, t):
        return np.broadcast_to(np.asarray(f(t), float), np.shape(t)).copy()

    def stems(f, color, z):
        v = values(f, nn * T)
        px, py = X(nn * T), Y(v)
        lc = LineCollection([[(a, base), (a, b)] for a, b in zip(px, py)], colors=color,
                            linewidths=STEM_W * k, zorder=z, capstyle='butt')
        ax.add_collection(lc); lc.set_clip_path(clip)
        line(px, py, ls='none', marker='o', ms=2 * DOT_R * k, mfc=color, mec=color, mew=0, zorder=z + 0.5)

    def curve(f, z, **kw):
        v = values(f, tt)
        for at in getattr(f, 'impulses', {}):               # the sample that falls on an impulse
            i = int(np.argmin(np.abs(tt - at)))
            if abs(tt[i] - at) < 1e-9:
                v[i] = v[i - 1]
        line(X(tt), Y(v), zorder=z, solid_joinstyle='round', **kw)

    # the reference signal, in grey behind the signal
    if reference is not None:
        if discrete:
            stems(reference, GREY, 3)
        else:
            curve(reference, 3, lw=CURVE_W * k, color=GREY)

    # the envelope of an oscillation, dashed
    env = getattr(signal, 'envelope', None)
    if env is not None:
        for sign in (1, -1):
            line(X(tt), Y(sign * values(env, tt)), lw=1.0 * k, color=BLUE, alpha=0.5, zorder=3,
                 linestyle=(0, (5, 3)), dash_capstyle='butt')

    # the signal
    if discrete:
        stems(signal, BLUE, 4)
    else:
        curve(signal, 4, lw=CURVE_W * k, color=BLUE)
        for at, area in getattr(signal, 'impulses', {}).items():      # impulses: arrows that carry their area
            px, py, d = X(at), Y(area), 1 if area >= 0 else -1
            line([px, px], [base, py + 7 * d], lw=CURVE_W * k, color=BLUE, zorder=4, solid_capstyle='butt')
            ax.add_patch(Polygon([(px, py), (px - 4.5, py + 9 * d), (px + 4.5, py + 9 * d)], closed=True,
                                 fc=BLUE, ec=BLUE, lw=0, zorder=5))
            ax.text(px + 9, py + 3 * d, '$(%g)$' % area, ha='left', va='center', fontproperties=roman(TICK), color=BLUE, zorder=6)


# ---------------------------------------------------------------- the figure

def show(*signals, notes=None, span=(-10, 10), ylim=(-2, 2), T=1, reference=None, symbol='x', zoom=1.0):
    """Three columns: the names of a signal, the signal over continuous time, and over discrete time.
    One row per signal.

    signals    functions of an array of time values.  The first docstring line gives the names,
               as "continuous-time name | discrete-time name" where the two differ.
               A signal may carry two attributes: `envelope`, a function drawn dashed as +envelope
               and -envelope, and `impulses`, a dict {time: area} of impulses drawn as arrows
               over continuous time.
    notes      for every signal the text that goes right under its two plots: a pair
               (continuous time, discrete time), or one text for both.
    span       the time axis (t0, t1); the sequence is drawn for the integers n with t0 <= nT <= t1.
    ylim       the amplitude axis (x0, x1).
    T          the sampling period: the sequence is x[n] = x(nT).
    reference  a signal drawn in grey behind every signal, such as the input of a system;
               or a list with one reference (or None) per row.
    symbol     the name of the signal on the axes: 'x' gives x(t) and x[n]; or a list with one name per row."""
    refs = list(reference) if isinstance(reference, (list, tuple)) else [reference] * len(signals)
    syms = list(symbol) if isinstance(symbol, (list, tuple)) else [symbol] * len(signals)
    rows = []
    for i, f in enumerate(signals):
        note = notes[i] if notes else ''
        note = (note, note) if isinstance(note, str) else tuple(note)
        nm = names(f)
        label, y = [], 0
        for name in dict.fromkeys(nm):                                # one entry if both names are the same
            for text, _ in _wrap(name, LABEL, LABEL_W - 2 * MARGIN)[0]:
                label.append((text, y)); y += LINE
            y += PARA
        rows.append((f, (refs[i], syms[i]), (label, y - PARA), [_wrap(t, NOTE, NOTE_W) for t in note]))

    plot_h = MARGIN + TOP + PLOT_H + BOTTOM
    hs = []
    for _, _, (_, lh), cells in rows:
        under = max(h for _, h in cells)
        hs.append(max(plot_h + (under + 8 if under else 0) + MARGIN, lh + 2 * MARGIN))
    W, H = LABEL_W + 2 * PANEL_W, HEAD_H + sum(hs)
    fig = plt.figure(figsize=(W * zoom / 96.0, H * zoom / 96.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    k = PT * zoom
    reg, bold = _fonts()
    roman = lambda size: FontProperties(fname=reg, size=size * k, math_fontfamily='cm')
    rule = lambda xs, ys: ax.add_line(Line2D(xs, ys, color=RULE, lw=0.8, zorder=0))

    for j, head in enumerate(HEADS):                                  # column heads
        ax.text(LABEL_W + j * PANEL_W + PANEL_W / 2, HEAD_H / 2, head, ha='center', va='center',
                fontproperties=FontProperties(fname=bold, size=18 * k))
    y = HEAD_H
    for (f, (ref, sym), (label, lh), cells), h in zip(rows, hs):
        rule([0, W], [y, y])
        for text, dy in label:                                        # first column: the names of the signal
            ax.text(MARGIN, y + (h - lh) / 2 + dy + 0.5 * LINE, text, ha='left', va='center',
                    fontproperties=roman(LABEL))
        for j, (note, _) in enumerate(cells):
            xc = LABEL_W + j * PANEL_W + PANEL_W / 2
            _plot(ax, k, xc - PLOT_W / 2 + 6, y + MARGIN + TOP, f, span, ylim, T, bool(j), ref, sym)
            for text, base in note:                                   # the note, right under the plot
                ax.text(xc, y + plot_h + 8 + base, text, ha='center', va='baseline', fontproperties=roman(NOTE))
        y += h
    for j in range(2):
        xg = LABEL_W + j * PANEL_W
        rule([xg, xg], [HEAD_H, H])
