"""Transistors and the drawing context.

`Transistor` is the common base class of `BJT` and `MOSFET`.  A circuit is written once, against
the base class, and drawn with either subclass:

                      Transistor        BJT                 MOSFET
    terminals         .bg  .cd  .es     .b  .c  .e          .g  .d  .s
    kind 'n' / 'p'                      npn / pnp           NMOS / PMOS
    name 1                              Q_1                 M_1
    supply                              V_CC                V_DD
    size label                          none                W/L

The shared terminal names read "base or gate", "collector or drain", "emitter or source".
"""
import math

from . import symbols as S
from .scene import Scene, Inst

STEP = 20   # standard lead length / grid step


class Transistor:
    """One transistor in a drawing.  Subclasses say which symbol is drawn and what things are called."""
    letter = '?'            # device letter: Q_1, M_1
    supply = 'V'            # name of the positive supply
    own = {}                # shared terminal name -> own terminal name, e.g. 'bg' -> 'b'

    def __init__(self, ckt, kind, at, anchor, side='left', rot=0, **options):
        """kind 'n' or 'p'.  The terminal `anchor` is put at the point `at`.
        side = side of the base or gate lead, rot = quarter turns."""
        self.ckt, self.kind, self.rot, self.mir = ckt, kind, rot, side == 'right'
        self.inst = ckt.sc.put(self.symbol(kind, **options), self.terminal(anchor), at, rot, self.mir)
        t = self.inst.t
        self.bg, self.cd, self.es = (t[self.own[k]] for k in ('bg', 'cd', 'es'))

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

    @property
    def top(self):
        """the terminal on the supply side of an upright device"""
        return self.cd if self.kind == 'n' else self.es

    @property
    def bottom(self):
        return self.es if self.kind == 'n' else self.cd

    def name(self, index, loc='auto', dy=0, dx=0):
        """label the device: name(1) writes Q_1 or M_1"""
        text = '%s_%s' % (self.letter, index)
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
    letter, supply = 'Q', 'V_CC'
    own = {'bg': 'b', 'cd': 'c', 'es': 'e'}
    b = property(lambda self: self.bg)
    c = property(lambda self: self.cd)
    e = property(lambda self: self.es)

    def symbol(self, kind, **options):
        return 'npn' if kind == 'n' else 'pnp'


class MOSFET(Transistor):
    """MOS field-effect transistor: gate .g, drain .d, source .s"""
    letter, supply = 'M', 'V_DD'
    own = {'bg': 'g', 'cd': 'd', 'es': 's'}
    g = property(lambda self: self.bg)
    d = property(lambda self: self.cd)
    s = property(lambda self: self.es)

    def symbol(self, kind, bulk=False, thick=False):
        if thick:
            return 'nmos_thick'
        return ('nmos' if kind == 'n' else 'pmos') + ('_b' if bulk else '')

    def size(self, text):
        i = self.inst
        if self.mir:
            self.ckt.sc.text(text, i.x - 8, i.y + i.h / 2 - 20, 'end')
        else:
            self.ckt.sc.text(text, i.x + i.w + 8, i.y + i.h / 2 - 20, 'start')


class Ckt:
    """Drawing context: a scene, and the transistor class that the circuit is drawn with."""

    def __init__(self, sc, device=MOSFET):
        self.sc, self.device = sc, device

    # ---- devices ------------------------------------------------------
    def transistor(self, kind, at, anchor, side='left', rot=0, name=None, lbl='auto', **options):
        """Place a transistor of the class of this drawing: BJT or MOSFET.
        kind 'n' or 'p'; the terminal `anchor` ('bg', 'cd', 'es') is put at the point `at`."""
        t = self.device(self, kind, at, anchor, side, rot, **options)
        if name is not None:
            t.name(name, lbl)
        return t

    def pin(self, name):
        """terminal letter in this drawing: 'cd' -> C or D"""
        return self.device.pin(name)

    def res(self, p, label=None, to='down'):
        """resistor with one end at p, running 'down', 'left' or 'right'; returns the other end"""
        want = {'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}[to]
        rot = next(r for r in range(4) if _dir(Inst('resistor', 0, 0, r, False, S.SYM['resistor']).t, 'a', 'b') == want)
        i = self.sc.put('resistor', 'a', p, rot)
        if label and to == 'down':
            self.sc.text(label, p[0] + 18, p[1] + 30, 'start')
        elif label:
            self.sc.text(label, (p[0] + i.t['b'][0]) / 2, p[1] + 22, 'middle')
        return i.t['b']

    def load(self, p):
        """generic load, drawn as a box, with its upper end at p; returns the lower end"""
        return self.sc.put('load', 'a', p).t['b']

    def look(self, p, into, side, label=None, shift=0, gap=17):
        """Mark the impedance seen at the point p with an arrow that has a 90-degree corner.
        into: direction of view, 'l', 'r', 'u' or 'd'.  The arrow sits beside the lead on the
        given side; its tail leaves towards that side and carries the label."""
        a, b = _UNIT[into], _UNIT[side]
        for rot, mir in ((r, m) for r in range(4) for m in (False, True)):
            t = Inst('look', 0, 0, rot, mir, S.SYM['look']).t
            if _dir(t, 'corner', 'tip') == a and _dir(t, 'corner', 'tail') == b:
                break
        back = shift - S.SYM['look']['w'] / 2 + 4
        corner = (p[0] + a[0] * back + b[0] * gap, p[1] + a[1] * back + b[1] * gap)
        i = self.sc.add('look', corner[0] - t['corner'][0], corner[1] - t['corner'][1], rot, mir)
        if label:
            x, y = i.t['tail']
            if b[1]:
                self.sc.text(label, x, y + 13 * b[1], 'middle')
            else:
                self.sc.text(label, x + 7 * b[0], y, 'start' if b[0] > 0 else 'end')

    # ---- rails / ports ------------------------------------------------
    def supply(self, p, side='r'):
        """supply bar with its stub ending at p, labelled V_CC or V_DD beside the bar as in the book"""
        self.sc.put('vdd', 't', p)
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

    def port(self, p, label, side='l', dx=10):
        """open terminal with a label on the given side: 'l', 'r', 't' or 'b'"""
        sc = self.sc
        sc.dot(p, True)
        if not label:
            return
        if side == 'l':
            sc.text(label, p[0] - dx, p[1], 'end')
        elif side == 'r':
            sc.text(label, p[0] + dx, p[1], 'start')
        elif side == 't':
            sc.text(label, p[0], p[1] - 17, 'middle')
        else:
            sc.text(label, p[0], p[1] + 17, 'middle')

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
