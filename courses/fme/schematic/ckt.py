"""Transistors and the drawing context.

`Transistor` is the common base class of `BJT` and `MOSFET`.  A circuit is written once, against
the base class, and drawn with either subclass:

                      Transistor        BJT                 MOSFET
    terminals         .bg  .cd  .es     .b  .c  .e          .g  .d  .s
    kind 'n' / 'p'                      npn / pnp           NMOS / PMOS
    name 1                              Q_1                 M_1
    supply                              V_CC                V_DD
    size label                          none                W/L
    ideal                               V_A = infinity      lambda = 0
    pick(a, b)                          a                   b

The shared terminal names read "base or gate", "collector or drain", "emitter or source".
"""
import math

from . import symbols as S
from .scene import Scene, Inst, RED, LABEL

STEP = 20   # standard lead length / grid step


class Transistor:
    """One transistor in a drawing.  Subclasses say which symbol is drawn and what things are called."""
    letter = '?'            # device letter: Q_1, M_1
    supply = 'V'            # name of the positive supply
    ideal = ''              # the condition under which the output resistance r_O is infinite
    own = {}                # shared terminal name -> own terminal name, e.g. 'bg' -> 'b'

    def __init__(self, ckt, kind, at, anchor, side='left', rot=0, **options):
        """kind 'n' or 'p'.  The terminal `anchor` is put at the point `at`.
        side = side of the base or gate lead, rot = quarter turns."""
        self.ckt, self.kind, self.rot, self.mir = ckt, kind, rot, side == 'right'
        self.inst = ckt.sc.put(self.symbol(kind, **options), self.terminal(anchor), at, rot, self.mir)
        t = self.inst.t
        self.bg, self.cd, self.es = (t[self.own[k]] for k in ('bg', 'cd', 'es'))
        # the vertical line through the device: its main route if it stands upright, its middle if it lies
        self.line = self.cd[0] if rot % 2 == 0 else (self.cd[0] + self.es[0]) / 2

    # -- what a subclass defines ----------------------------------------
    def symbol(self, kind, **options):
        raise NotImplementedError

    def size(self, text):
        """size label beside the device (only a MOSFET has one)"""

    # -- common behaviour -------------------------------------------------
    @classmethod
    def terminal(cls, name):
        """own terminal name for a shared name ('bg' -> 'b' or 'g'); own names pass through"""
        return cls.own.get(name, name)

    @classmethod
    def pin(cls, name):
        """terminal letter as printed in a drawing: 'cd' -> C or D"""
        return cls.terminal(name).upper()

    def lead(self, name):
        """the point where the lead of a terminal ('bg', 'cd', 'es') meets the body of the device"""
        return self.inst.a[self.terminal(name)]

    @property
    def top(self):
        """the terminal on the supply side of an upright device"""
        return self.cd if self.kind == 'n' else self.es

    @property
    def bottom(self):
        return self.es if self.kind == 'n' else self.cd

    @classmethod
    def pick(cls, bjt, mosfet):
        """the one of two things that belongs to this kind of transistor"""
        raise NotImplementedError

    def name(self, index, loc='auto', dy=0, dx=0):
        """label the device: name(1) writes Q_1 or M_1"""
        self.inst.label = '%s_%s' % (self.letter, index)             # the name of the element, for a renderer that asks
        self.note(self.inst.label, loc, dy, dx)

    def note(self, text, loc='auto', dy=0, dx=0):
        """write something beside the device, such as its name or the condition `ideal`"""
        sc, i = self.ckt.sc, self.inst
        cx, cy = i.x + i.w / 2, i.y + i.h / 2
        if loc == 'auto':
            loc = 'l' if self.mir else 'r'
        if self.rot in (1, 3):
            above = self.bg[1] < cy
            sc.text(text, cx + 10, (i.y - 10) if above else (i.y + i.h + 10), 'start')
        elif loc == 'r':
            sc.text(text, i.x + i.w + 8 + dx, cy + dy, 'start')
        else:
            sc.text(text, i.x - 8 - dx, cy + dy, 'end')


class BJT(Transistor):
    """Bipolar junction transistor: base .b, collector .c, emitter .e"""
    letter, supply, ideal = 'Q', 'V_CC', r'$V_A=\infty$'
    own = {'bg': 'b', 'cd': 'c', 'es': 'e'}
    b = property(lambda self: self.bg)
    c = property(lambda self: self.cd)
    e = property(lambda self: self.es)

    def symbol(self, kind, **options):
        return 'npn' if kind == 'n' else 'pnp'

    @classmethod
    def pick(cls, bjt, mosfet):
        return bjt


class MOSFET(Transistor):
    """MOS field-effect transistor: gate .g, drain .d, source .s"""
    letter, supply, ideal = 'M', 'V_DD', r'$\lambda=0$'
    own = {'bg': 'g', 'cd': 'd', 'es': 's'}
    g = property(lambda self: self.bg)
    d = property(lambda self: self.cd)
    s = property(lambda self: self.es)

    def symbol(self, kind, bulk=False, thick=False):
        if thick:
            return 'nmos_thick'
        return ('nmos' if kind == 'n' else 'pmos') + ('_b' if bulk else '')

    @classmethod
    def pick(cls, bjt, mosfet):
        return mosfet

    def size(self, text):
        i = self.inst
        i.size = text
        if self.mir:
            self.ckt.sc.text(text, i.x - 8, i.y + i.h / 2 - 20, 'end')
        else:
            self.ckt.sc.text(text, i.x + i.w + 8, i.y + i.h / 2 - 20, 'start')


class Ckt:
    """Drawing context: a scene, and the transistor class that the circuit is drawn with."""

    def __init__(self, sc, device=MOSFET):
        self.sc, self.device = sc, device
        self.devices, self._axis = [], None

    # ---- where the drawing sits in its cell -----------------------------
    def center(self, x):
        """align the drawing on the vertical line x instead of on its transistors"""
        self._axis = x

    @property
    def axis(self):
        """The vertical line that a figure aligns this drawing on: the main route, if all transistors
        share one, else the middle between them (as in a current mirror)."""
        if self._axis is not None:
            return self._axis
        lines = [t.line for t in self.devices]
        return sum(lines) / len(lines) if lines else 0.0

    # ---- devices ------------------------------------------------------
    def transistor(self, kind, at, anchor, side='left', rot=0, name=None, lbl='auto', **options):
        """Place a transistor of the class of this drawing: BJT or MOSFET.
        kind 'n' or 'p'; the terminal `anchor` ('bg', 'cd', 'es') is put at the point `at`."""
        t = self.device(self, kind, at, anchor, side, rot, **options)
        self.devices.append(t)
        if name is not None:
            t.name(name, lbl)
        return t

    def pin(self, name):
        """terminal letter in this drawing: 'cd' -> C or D"""
        return self.device.pin(name)

    def pick(self, bjt, mosfet):
        """the first argument in a BJT drawing, the second in a MOSFET drawing"""
        return self.device.pick(bjt, mosfet)

    def res(self, p, label=None, to='down', side=None):
        """resistor with one end at p, running 'down', 'left' or 'right'; returns the other end.
        The label goes on the right of an upright resistor and below a lying one; side 'l' or 'u' puts it
        on the left or above."""
        want = {'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}[to]
        rot = next(r for r in range(4) if _dir(Inst('resistor', 0, 0, r, False, S.SYM['resistor']).t, 'a', 'b') == want)
        i = self.sc.put('resistor', 'a', p, rot)
        i.label = label
        if label and to == 'down':
            if side == 'l':
                self.sc.text(label, p[0] - 18, p[1] + 30, 'end')
            else:
                self.sc.text(label, p[0] + 18, p[1] + 30, 'start')
        elif label:
            self.sc.text(label, (p[0] + i.t['b'][0]) / 2, p[1] + (-22 if side == 'u' else 22), 'middle')
        return i.t['b']

    def load(self, p):
        """generic load, drawn as a box, with its upper end at p; returns the lower end"""
        return self.sc.put('load', 'a', p).t['b']

    def look(self, p, into, side, label=None, form='plain', shift=None, gap=None, nudge=0):
        """Mark the impedance seen at the terminal p with the hooked arrow of the book, in red.
        into: direction of view, 'l', 'r', 'u' or 'd'.  The arrow runs beside the lead on the given
        side; its tail leaves towards that side and carries the label.
        form 'plain': as printed at a base or a collector, the arrow passes the terminal.
             'low':   as printed at an emitter, the arrow stops at the terminal and has a short tail.
             'long':  long shaft and short tail, as printed with a long expression.
        shift moves the arrow along its direction, gap sets its distance from the lead, and nudge
        moves a label that sits under or over its tail sideways (px)."""
        a, b = _UNIT[into], _UNIT[side]
        k = S.LOOK
        key = {'plain': 'look', 'low': 'look_short', 'long': 'look_long'}[form]
        for rot, mir in ((r, m) for r in range(4) for m in (False, True)):
            t = Inst(key, 0, 0, rot, mir, S.SYM[key]).t
            if _dir(t, 'corner', 'tip') == a and _dir(t, 'corner', 'tail') == b:
                break
        back = (shift or 0) - S.S * (k['low'] if form == 'low' else k['lead'])
        gap = S.S * k['gap'] if gap is None else gap
        corner = (p[0] + a[0] * back + b[0] * gap, p[1] + a[1] * back + b[1] * gap)
        i = self.sc.add(key, corner[0] - t['corner'][0], corner[1] - t['corner'][1], rot, mir, color=RED)
        if not label:
            return
        x, y = i.t['tail']
        if b[1]:                                    # tail up or down: the label sits at its end, and a long
            x0, _, x1, _ = self.sc.text_box((label, 0, 0, 'middle', LABEL, 'math'))    # one grows away from the circuit
            over = max((x1 - x0) / 2 - 25, 0)
            self.sc.text(label, x - a[0] * over + nudge, y + 13 * b[1], 'middle', color=RED)
        elif form == 'long':                        # a long expression sits above the tail, so that the drawing
            self.sc.text(label, x + 10 * b[0], y - 17, 'middle', color=RED)   # stays narrow about its main route
        else:                                       # tail to the left or right: the label follows it
            pad = 16 if 'frac' in label.split('}')[0] else 7
            self.sc.text(label, x + pad * b[0], y, 'start' if b[0] > 0 else 'end', color=RED)

    # ---- rails / ports ------------------------------------------------
    def supply(self, p, side='r'):
        """supply bar with its stub ending at p, labelled V_CC or V_DD beside the bar as in the book;
        side None leaves the bar without a label"""
        self.sc.put('vdd', 't', p)
        if side is None:
            return
        if side == 'r':
            self.sc.text(self.device.supply, p[0] + 23, p[1] - 14, 'start')
        else:
            self.sc.text(self.device.supply, p[0] - 23, p[1] - 14, 'end')

    def gnd(self, p):
        self.sc.put('ground', 't', p)

    def ac(self, p):
        """ac ground: a ground symbol marked "ac" """
        self.sc.put('ground', 't', p)
        self.sc.text('~ac', p[0], p[1] + 33, 'middle', 13)

    def port(self, p, label, side='l', dx=10, solid=False, color=None):
        """terminal with a label on the given side: 'l', 'r', 't' or 'b'.  Open, or solid as the book
        draws a terminal that is held at a fixed voltage."""
        sc = self.sc
        sc.dot(p, not solid)
        if not label:
            return
        if side == 'l':
            sc.text(label, p[0] - dx, p[1], 'end', color=color)
        elif side == 'r':
            sc.text(label, p[0] + dx, p[1], 'start', color=color)
        elif side == 't':
            sc.text(label, p[0], p[1] - 17, 'middle', color=color)
        else:
            sc.text(label, p[0], p[1] + 17, 'middle', color=color)

    def w(self, *pts):
        self.sc.wire(*pts)

    def dot(self, p):
        self.sc.dot(p)

    # ---- compound helpers ---------------------------------------------
    def chain(self, specs, x, y, side='left', gap=STEP):
        """Series stack of upright transistors; specs = [(kind, name, options)], top terminal of the
        first device at (x, y).  Returns (transistors, junction points between them)."""
        devs, js = [], []
        cy = y
        for i, spec in enumerate(specs):
            kind, name = spec[0], spec[1]
            opts = dict(spec[2]) if len(spec) > 2 else {}
            t = self.transistor(kind, (x, cy), 'cd' if kind == 'n' else 'es', side=opts.pop('side', side),
                                name=name, lbl=opts.pop('lbl', 'auto'), **opts)
            if devs:
                self.w(devs[-1].bottom, t.top)
                js.append((x, (devs[-1].bottom[1] + t.top[1]) / 2))
            devs.append(t)
            cy = t.bottom[1] + gap
        return devs, js

    @staticmethod
    def cross(x, y1, y2, angle=60):
        """Two wires that swap levels y1 and y2 cross as an X centred on the vertical line x.
        Returns the x positions (left, right) between which the two diagonals run, each at
        `angle` degrees to the horizontal."""
        half = abs(y2 - y1) / 2 / math.tan(math.radians(angle))
        return x - half, x + half

    def bus(self, devs, x):
        """tie the bases or gates of devs together on a vertical line at x"""
        for t in devs:
            self.w(t.bg, (x, t.bg[1]))
        ys = [t.bg[1] for t in devs]
        self.w((x, min(ys)), (x, max(ys)))


_UNIT = {'l': (-1, 0), 'r': (1, 0), 'u': (0, -1), 'd': (0, 1)}


def _dir(points, a, b):
    """unit step from point a to point b of a symbol, as (-1 | 0 | 1, -1 | 0 | 1)"""
    sign = lambda v: (v > 1e-9) - (v < -1e-9)
    return (sign(points[b][0] - points[a][0]), sign(points[b][1] - points[a][1]))


def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def new(name, title, note=''):
    return Scene(name, title, note)
