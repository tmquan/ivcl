"""sigplot: signals, systems and time axes as classes; every signal drawn over two time axes.

    from sigplot import Signal, Sinusoid, TimeShift, show

    class Pulse(Signal):
        '''One-sided pulse'''
        def values(self, t):
            return np.where((t >= 0) & (t < 5), 1.0, 0.0)

    x = Pulse()
    show(x, TimeShift(8)(x), Sinusoid(omega=np.pi / 8))

Three small class families work together:

    Signal   a function of time.  Subclasses override `values`; the elementary signals,
             the exponentials and everything a system returns are signals.
    System   a mapping from one signal to another.  Subclasses override `apply`.
    Time     a time axis.  `ContinuousTime` draws a signal as a curve x(t), `DiscreteTime`
             as the sequence of its samples x[n] = x(nT).  Subclasses override `draw`.

`show` draws one row per signal and one column per time axis.  Everything is laid out in
pixels (96 per inch), with the same three columns in every figure, and labels are set in
Computer Modern through matplotlib's mathtext, so no TeX installation is needed.
"""
import copy
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.textpath import TextPath

__all__ = ['Signal', 'Step', 'Ramp', 'Rect', 'Tri', 'Sinc', 'Impulse',
           'ComplexExponential', 'Sinusoid', 'Exponential',
           'System', 'AmplitudeScaling', 'AmplitudeReversal', 'TimeShift', 'TimeScaling', 'TimeReversal', 'Cascade',
           'Time', 'ContinuousTime', 'DiscreteTime', 'Plot', 'Sheet', 'show']


# ================================================================ signals

class Signal:
    """A function of time.

    A subclass overrides `values(t)`; its first docstring line is the name of the signal,
    written as "continuous-time name | discrete-time name" where the two differ.
    `Signal(f)` wraps an ordinary function.  Signals add, subtract and scale like numbers,
    and the methods `shift`, `scale`, `reverse`, `even`, `odd`, `repeat` and `quantize`
    return new signals."""

    def __init__(self, function=None, name=None, impulses=None):
        self._function = function
        self._name = name
        self._impulses = dict(impulses or {})

    # ---- what a subclass overrides

    def values(self, t):
        """the values of the signal at the times `t`, an array"""
        if self._function is None:
            raise NotImplementedError('%s must override values(t)' % type(self).__name__)
        return self._function(t)

    @property
    def envelope(self):
        """a signal drawn dashed above and below this one, or None"""
        return None

    @property
    def impulses(self):
        """{time: area} of the impulses of the signal; over continuous time they are drawn as arrows"""
        return self._impulses

    # ---- evaluation and names

    def __call__(self, t):
        t = np.asarray(t, float)
        return np.broadcast_to(np.asarray(self.values(t), float), t.shape).copy()

    @property
    def name(self):
        if self._name is not None:
            return self._name
        doc = type(self).__doc__ if type(self) is not Signal else None
        return (doc or '').strip().splitlines()[0] if doc else ''

    def names(self):
        """(continuous-time name, discrete-time name)"""
        parts = [p.strip() for p in self.name.split('|')]
        return parts[0], parts[-1]

    def named(self, name):
        """the same signal under another name"""
        twin = copy.copy(self)
        twin._name = name
        return twin

    @staticmethod
    def of(thing):
        """a Signal from a Signal, a function of time or a number"""
        if isinstance(thing, Signal):
            return thing
        if callable(thing):
            return Signal(thing, (thing.__doc__ or '').strip().splitlines()[0] if thing.__doc__ else None)
        return Signal(lambda t: thing + 0 * t)

    # ---- arithmetic on the amplitude

    def __add__(self, other):
        other = Signal.of(other)
        return Signal(lambda t: self(t) + other(t), impulses={**self.impulses, **other.impulses})

    def __mul__(self, other):
        if np.isscalar(other):
            return Signal(lambda t: other * self(t), impulses={at: other * area for at, area in self.impulses.items()})
        other = Signal.of(other)
        return Signal(lambda t: self(t) * other(t))

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Signal.of(other))

    def __truediv__(self, number):
        return self * (1 / number)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return Signal.of(other) - self

    # ---- operations on the time variable

    def shift(self, b):
        """x(t + b): to the left for b > 0"""
        return Signal(lambda t: self(t + b), impulses={at - b: area for at, area in self.impulses.items()})

    def scale(self, a):
        """x(at): compressed for |a| > 1, mirrored for a < 0"""
        return Signal(lambda t: self(a * t), impulses={at / a: area / abs(a) for at, area in self.impulses.items()})

    def reverse(self):
        """x(-t)"""
        return self.scale(-1)

    # ---- signals made from this one

    def even(self):
        """the even part, [x(t) + x(-t)] / 2"""
        return (self + self.reverse()) / 2

    def odd(self):
        """the odd part, [x(t) - x(-t)] / 2"""
        return (self - self.reverse()) / 2

    def repeat(self, period):
        """the part on 0 <= t < period, repeated every `period`"""
        return Signal(lambda t: self(np.mod(t, period)))

    def quantize(self, delta):
        """rounded to the nearest multiple of `delta`"""
        return Signal(lambda t: delta * np.round(self(t) / delta))


class Step(Signal):
    """Unit step"""

    def values(self, t):
        return np.where(t >= 0, 1.0, 0.0)


class Ramp(Signal):
    """Unit ramp"""

    def values(self, t):
        return np.maximum(t, 0.0)


class Rect(Signal):
    """Unit rectangle"""

    def values(self, t):
        return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


class Tri(Signal):
    """Unit triangle"""

    def values(self, t):
        return np.maximum(1.0 - np.abs(t), 0.0)


class Sinc(Signal):
    """Unit sinc"""

    def values(self, t):
        return np.sinc(t)


class Impulse(Signal):
    """Unit impulse | Unit sample"""

    def __init__(self, area=1):
        super().__init__(impulses={0: area})
        self.area = area

    def values(self, t):                      # the unit sample; the impulse itself is the arrow
        return np.where(t == 0, float(self.area), 0.0)


class ComplexExponential(Signal):
    """Complex exponential"""

    def __init__(self, C=1, r=0, omega=0, phi=0, name=None):
        super().__init__(name=name)
        self.C, self.r, self.omega, self.phi = C, r, omega, phi

    def values(self, t):                      # the real part of C e^(st), s = r + j omega, C = |C| e^(j phi)
        return self.C * np.exp(self.r * t) * np.cos(self.omega * t + self.phi)

    @property
    def envelope(self):
        return Exponential(abs(self.C), self.r)


class Sinusoid(ComplexExponential):
    """Sinusoid"""

    def __init__(self, A=1, omega=1, phi=0, name=None):
        super().__init__(A, 0, omega, phi, name)

    @property
    def envelope(self):                       # flat: not drawn
        return None


class Exponential(ComplexExponential):
    """Real exponential"""

    def __init__(self, C=1, a=0, name=None):
        super().__init__(C, a, 0, 0, name)

    @property
    def envelope(self):                       # the signal is its own envelope: not drawn
        return None


# ================================================================ systems

class System:
    """A mapping from an input signal to an output signal.

    A subclass overrides `apply(x)`, which returns the output signal, and `label`, the name
    that the output gets.  `system(x)` is the output for the input `x`, and `first >> second`
    is the cascade of two systems."""

    label = ''

    def apply(self, x):
        raise NotImplementedError('%s must override apply(x)' % type(self).__name__)

    def __call__(self, x):
        return self.apply(Signal.of(x)).named(self.label)

    def __rshift__(self, other):
        return Cascade(self, other)


class AmplitudeScaling(System):
    """y(t) = K x(t)"""

    def __init__(self, K):
        self.K = K

    def apply(self, x):
        return self.K * x

    @property
    def label(self):
        return 'Amplification' if abs(self.K) > 1 else 'Attenuation'


class AmplitudeReversal(AmplitudeScaling):
    """y(t) = -x(t)"""

    label = 'Amplitude reversal'

    def __init__(self):
        super().__init__(-1)


class TimeShift(System):
    """y(t) = x(t + b)"""

    def __init__(self, b):
        self.b = b

    def apply(self, x):
        return x.shift(self.b)

    @property
    def label(self):
        return 'Advance' if self.b > 0 else 'Delay'


class TimeScaling(System):
    """y(t) = x(at)"""

    def __init__(self, a):
        self.a = a

    def apply(self, x):
        return x.scale(self.a)

    @property
    def label(self):
        return 'Compression | Decimation' if abs(self.a) > 1 else 'Expansion'


class TimeReversal(TimeScaling):
    """y(t) = x(-t)"""

    label = 'Time reversal'

    def __init__(self):
        super().__init__(-1)


class Cascade(System):
    """one system after another: the output of the first is the input of the second"""

    def __init__(self, *systems):
        self.systems = systems

    def apply(self, x):
        for system in self.systems:
            x = system.apply(x)
        return x

    @property
    def label(self):
        return ', then '.join(s.label.split('|')[0].strip().lower() for s in self.systems).capitalize()


# ================================================================ layout, in px

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


def _fonts():
    d = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
    return os.path.join(d, 'cmr10.ttf'), os.path.join(d, 'cmb10.ttf')


def _roman(size):
    return FontProperties(fname=_fonts()[0], size=size, math_fontfamily='cm')


def _extent(text, size):
    """(width, lowest point, highest point) in px of a sentence with $maths$ in it"""
    tp = TextPath((0, 0), text, size=size, prop=_roman(size))
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
        if cur and _extent(trial, size)[0] > width:
            lines.append(cur); cur = word
        else:
            cur = trial
    out, y = [], 0.0
    for line in lines + [cur]:
        _, lo, hi = _extent(line, size)
        up, down = max(hi, 0.72 * size), max(-lo, 0.22 * size)
        out.append((line, y + up)); y += up + down + NOTE_LEAD
    return out, y - NOTE_LEAD


def _ticks(lo, hi, most, steps=(1, 2, 5, 10, 20, 50, 100)):
    """the multiples of a nice step between lo and hi, at most `most` of them"""
    for s in steps:
        first, last = int(np.ceil(lo / s - 1e-9)), int(np.floor(hi / s + 1e-9))
        if last - first + 1 <= most:
            return [k * s for k in range(first, last + 1)]
    return [lo, hi]


# ================================================================ time axes

class Time:
    """A time axis over the span (t0, t1).

    A subclass says how a signal is named on the axis (`label`), where its ticks are
    (`ticks`) and how a signal is drawn on it (`draw`)."""

    head = ''                  # the head of the column
    variable = ''              # the name of the axis

    def __init__(self, span=(-10, 10), T=1):
        self.span, self.T = span, T

    def label(self, symbol):
        raise NotImplementedError

    def ticks(self):
        """[(time, tick label)]"""
        raise NotImplementedError

    def draw(self, plot, signal, color=BLUE, z=4):
        raise NotImplementedError

    def outline(self, plot, signal, **style):
        """a signal as a thin continuous line, on either axis: used for envelopes"""
        t = np.linspace(*self.span, 2641)
        plot.line(plot.X(t), plot.Y(signal(t)), **style)


class ContinuousTime(Time):
    """t is a real number: the signal is a curve, and its impulses are arrows"""

    head, variable = 'Continuous time', 't'

    def label(self, symbol):
        return '$%s(t)$' % symbol

    def ticks(self):
        return [(t, '$%g$' % t) for t in _ticks(*self.span, 9)]

    def draw(self, plot, signal, color=BLUE, z=4):
        t = np.linspace(*self.span, 2641)
        v = signal(t)
        for at in signal.impulses:                      # the sample that falls on an impulse
            i = int(np.argmin(np.abs(t - at)))
            if abs(t[i] - at) < 1e-9:
                v[i] = v[i - 1]
        plot.line(plot.X(t), plot.Y(v), lw=CURVE_W * plot.k, color=color, zorder=z, solid_joinstyle='round')
        for at, area in signal.impulses.items():
            if self.span[0] <= at <= self.span[1]:
                plot.arrow(at, area, color, z)


class DiscreteTime(Time):
    """n is an integer: the signal is the sequence of its samples x[n] = x(nT), drawn as stems"""

    head, variable = 'Discrete time', 'n'

    def label(self, symbol):
        return '$%s[n]$' % symbol

    def samples(self):
        """the integers n with t0 <= nT <= t1"""
        t0, t1 = self.span
        return np.arange(np.ceil(t0 / self.T - 1e-9), np.floor(t1 / self.T + 1e-9) + 1)

    def ticks(self):                                    # the sample axis also counts in fours and eights
        t0, t1 = self.span
        return [(n * self.T, '$%g$' % n) for n in _ticks(t0 / self.T, t1 / self.T, 9, (1, 2, 4, 5, 8, 10, 20, 40, 50, 100))]

    def draw(self, plot, signal, color=BLUE, z=4):
        t = self.samples() * self.T
        plot.stems(t, signal(t), color, z)


# ================================================================ one plot

class Plot:
    """The plot area of one cell: a time axis, an amplitude axis and the marks drawn on them.
    Its top left corner is at (x0, y0) px of the figure."""

    def __init__(self, ax, k, x0, y0, time, ylim):
        self.ax, self.k, self.time, self.ylim = ax, k, time, ylim
        self.left, self.top, self.right, self.bottom = x0, y0, x0 + PLOT_W, y0 + PLOT_H
        self.clip = Rectangle((x0 - 4, y0 - 4), PLOT_W + 8, PLOT_H + 8, transform=ax.transData)
        self.base = self.Y(min(max(0.0, ylim[0]), ylim[1]))          # where the stems stand

    def X(self, t):
        t0, t1 = self.time.span
        return self.left + (np.asarray(t, float) - t0) / (t1 - t0) * PLOT_W

    def Y(self, v):
        v0, v1 = self.ylim
        return self.top + (v1 - np.asarray(v, float)) / (v1 - v0) * PLOT_H

    # ---- marks

    def line(self, xs, ys, clipped=True, **style):
        ln = Line2D(xs, ys, **style)
        self.ax.add_line(ln)
        if clipped:
            ln.set_clip_path(self.clip)
        return ln

    def text(self, x, y, s, size=TICK, **style):
        self.ax.text(x, y, s, fontproperties=_roman(size * self.k), zorder=6, **style)

    def stems(self, t, v, color, z):
        px, py = self.X(t), self.Y(v)
        lc = LineCollection([[(a, self.base), (a, b)] for a, b in zip(px, py)], colors=color,
                            linewidths=STEM_W * self.k, zorder=z, capstyle='butt')
        self.ax.add_collection(lc); lc.set_clip_path(self.clip)
        self.line(px, py, ls='none', marker='o', ms=2 * DOT_R * self.k, mfc=color, mec=color, mew=0, zorder=z + 0.5)

    def arrow(self, at, area, color, z):
        """an impulse: an arrow that carries its area"""
        px, py, d = self.X(at), self.Y(area), 1 if area >= 0 else -1
        self.line([px, px], [self.base, py + 7 * d], lw=CURVE_W * self.k, color=color, zorder=z, solid_capstyle='butt')
        self.ax.add_patch(Polygon([(px, py), (px - 4.5, py + 9 * d), (px + 4.5, py + 9 * d)], closed=True,
                                  fc=color, ec=color, lw=0, zorder=z + 1))
        self.text(px + 9, py + 3 * d, '$(%g)$' % area, ha='left', va='center', color=color)

    # ---- the axes

    def frame(self, symbol):
        """the lines x = 0 and t = 0, the two axes with their ticks, and their names"""
        (t0, t1), (v0, v1), w = self.time.span, self.ylim, AXIS_W * self.k
        if v0 < 0 < v1:
            self.line([self.left, self.right], [self.Y(0)] * 2, False, lw=w, color=ZERO, zorder=1)
        if t0 < 0 < t1:
            self.line([self.X(0)] * 2, [self.top, self.bottom], False, lw=w, color=ZERO, zorder=1)
        self.line([self.left, self.left, self.right], [self.top, self.bottom, self.bottom], False, lw=w, color='k',
                  zorder=2, solid_joinstyle='miter', solid_capstyle='projecting')
        for t, label in self.time.ticks():
            px = self.X(t)
            self.line([px, px], [self.bottom, self.bottom + TICK_L], False, lw=w, color='k', zorder=2)
            self.text(px, self.bottom + TICK_L + 3, label, ha='center', va='top')
        for v in _ticks(v0, v1, 7):
            py = self.Y(v)
            self.line([self.left - TICK_L, self.left], [py, py], False, lw=w, color='k', zorder=2)
            self.text(self.left - TICK_L - 3, py, '$%g$' % v, ha='right', va='center')
        self.text(self.right + 8, self.bottom, '$%s$' % self.time.variable, NOTE, ha='left', va='center')
        self.text(self.left, self.top - 9, self.time.label(symbol), NOTE, ha='center', va='bottom')

    def show(self, signal, reference=None, symbol='x'):
        """the frame, the reference in grey, the envelope dashed, and the signal"""
        self.frame(symbol)
        if reference is not None:
            self.time.draw(self, reference, GREY, 3)
        if signal.envelope is not None:
            for side in (signal.envelope, -signal.envelope):
                self.time.outline(self, side, lw=1.0 * self.k, color=BLUE, alpha=0.5, zorder=3,
                                  linestyle=(0, (5, 3)), dash_capstyle='butt')
        self.time.draw(self, signal)


# ================================================================ the figure

class Sheet:
    """A figure of rows and columns: one row per signal, one column per time axis, after a
    first column that holds the names of the signals.  A note goes right under every plot."""

    def __init__(self, axes, ylim=(-2, 2), zoom=1.0):
        self.axes, self.ylim, self.zoom = list(axes), ylim, zoom
        self.rows = []

    def add(self, signal, note='', reference=None, symbol='x'):
        """one more row; `note` is one text for all columns, or one text per column"""
        signal = Signal.of(signal)
        notes = [note] * len(self.axes) if isinstance(note, str) else list(note)
        label, y = [], 0
        for name in dict.fromkeys(signal.names()):                   # one entry if all names are the same
            for text, _ in _wrap(name, LABEL, LABEL_W - 2 * MARGIN)[0]:
                label.append((text, y)); y += LINE
            y += PARA
        reference = None if reference is None else Signal.of(reference)
        self.rows.append((signal, reference, symbol, (label, y - PARA), [_wrap(t, NOTE, NOTE_W) for t in notes]))
        return self

    def draw(self):
        plot_h = MARGIN + TOP + PLOT_H + BOTTOM
        heights = []
        for *_, (_, lh), cells in self.rows:
            under = max(h for _, h in cells)
            heights.append(max(plot_h + (under + 8 if under else 0) + MARGIN, lh + 2 * MARGIN))
        W, H = LABEL_W + len(self.axes) * PANEL_W, HEAD_H + sum(heights)
        fig = plt.figure(figsize=(W * self.zoom / 96.0, H * self.zoom / 96.0))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
        k = PT * self.zoom
        rule = lambda xs, ys: ax.add_line(Line2D(xs, ys, color=RULE, lw=0.8, zorder=0))

        for j, time in enumerate(self.axes):                         # column heads
            ax.text(LABEL_W + j * PANEL_W + PANEL_W / 2, HEAD_H / 2, time.head, ha='center', va='center',
                    fontproperties=FontProperties(fname=_fonts()[1], size=18 * k))
        y = HEAD_H
        for (signal, reference, symbol, (label, lh), cells), h in zip(self.rows, heights):
            rule([0, W], [y, y])
            for text, dy in label:                                   # first column: the names of the signal
                ax.text(MARGIN, y + (h - lh) / 2 + dy + 0.5 * LINE, text, ha='left', va='center',
                        fontproperties=_roman(LABEL * k))
            for j, (time, (note, _)) in enumerate(zip(self.axes, cells)):
                xc = LABEL_W + j * PANEL_W + PANEL_W / 2
                Plot(ax, k, xc - PLOT_W / 2 + 6, y + MARGIN + TOP, time, self.ylim).show(signal, reference, symbol)
                for text, base in note:                              # the note, right under the plot
                    ax.text(xc, y + plot_h + 8 + base, text, ha='center', va='baseline',
                            fontproperties=_roman(NOTE * k))
            y += h
        for j in range(len(self.axes)):
            xg = LABEL_W + j * PANEL_W
            rule([xg, xg], [HEAD_H, H])
        return fig


def show(*signals, notes=None, span=(-10, 10), ylim=(-2, 2), T=1, reference=None, symbol='x', zoom=1.0,
         axes=(ContinuousTime, DiscreteTime)):
    """Three columns: the names of a signal, the signal over continuous time, and over discrete time.
    One row per signal.

    signals    Signal objects (or functions of an array of time values).
    notes      for every signal the text that goes right under its plots: a pair
               (continuous time, discrete time), or one text for both.
    span       the time axis (t0, t1); the sequence is drawn for the integers n with t0 <= nT <= t1.
    ylim       the amplitude axis (x0, x1).
    T          the sampling period: the sequence is x[n] = x(nT).
    reference  a signal drawn in grey behind every signal, such as the input of a system;
               or a list with one reference (or None) per row.
    symbol     the name of the signal on the axes: 'x' gives x(t) and x[n]; or a list with one name per row.
    axes       the Time classes of the columns."""
    many = lambda v: list(v) if isinstance(v, (list, tuple)) else [v] * len(signals)
    sheet = Sheet([cls(span, T) for cls in axes], ylim, zoom)
    for i, (signal, ref, sym) in enumerate(zip(signals, many(reference), many(symbol))):
        sheet.add(signal, notes[i] if notes else '', ref, sym)
    sheet.draw()
