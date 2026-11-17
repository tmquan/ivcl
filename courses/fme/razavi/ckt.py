"""Mode-aware drawing helpers: the same layout code draws the BJT or the MOSFET version.

In BJT mode  'n' -> npn, 'p' -> pnp, gate -> base, drain -> collector, source -> emitter,
M_1 -> Q_1, V_DD -> V_CC, V_SS -> V_EE.
"""
from .scene import Scene

STEP = 20   # standard lead length / grid step
_BJT_NAMES = (('V_DD', 'V_CC'), ('V_SS', 'V_EE'))


class Dev:
    pass


class Ckt:
    def __init__(s, sc, bjt):
        s.sc, s.bjt = sc, bjt

    # ---- naming -----------------------------------------------------
    def term(s, k):
        """terminal letter: 'g','d','s' -> B, C, E (BJT) or G, D, S (MOSFET)"""
        return dict(g='B', d='C', s='E')[k] if s.bjt else k.upper()

    def label(s, txt):
        """label as shown in this mode"""
        if s.bjt:
            for a, b in _BJT_NAMES:
                txt = txt.replace(a, b)
        return txt

    # ---- devices ----------------------------------------------------
    def Q(s, kind, at, anchor, gate='left', rot=0, bulk=False, thick=False, name=None, lbl='auto'):
        """Place a transistor.  kind 'n' (npn / NMOS) or 'p' (pnp / PMOS).
        The terminal `anchor` ('g', 'd', 's') is put at the point `at`; gate = side of the control terminal."""
        sc = s.sc
        mir = (gate == 'right')
        if s.bjt:
            key = 'npn' if kind == 'n' else 'pnp'
            amap = {'g': 'b', 'd': 'c', 's': 'e', 'b': 'b'}
        else:
            key = 'nmos_thick' if thick else ('nmos' if kind == 'n' else 'pmos') + ('_b' if bulk else '')
            amap = {'g': 'g', 'd': 'd', 's': 's', 'b': 'b'}
        inst = sc.put(key, amap[anchor], at, rot, mir)
        d = Dev()
        d.inst = inst
        t = inst.t
        d.g = t[amap['g']]
        d.d = t[amap['d']]
        d.s = t[amap['s']]
        d.b = t.get('b') if (not s.bjt and bulk) else None
        d.kind = kind
        d.rot, d.mir = rot, mir
        if name:
            s.name(d, name, lbl)
        return d

    def name(s, d, nm, loc='auto', dy=0, dx=0):
        if s.bjt and nm.startswith('M'):
            nm = 'Q' + nm[1:]
        i = d.inst
        cx, cy = i.x + i.w / 2, i.y + i.h / 2
        if loc == 'auto':
            loc = 'r' if not d.mir else 'l'
        if d.rot in (1, 3):
            gate_top = d.g[1] < cy
            s.sc.text(nm, cx + 10, (i.y - 10) if gate_top else (i.y + i.h + 10), 'start')
            return
        if loc == 'r':
            s.sc.text(nm, i.x + i.w + 8 + dx, cy + dy, 'start')
        else:
            s.sc.text(nm, i.x - 8 - dx, cy + dy, 'end')

    def res(s, p, label=None):
        """vertical resistor with its upper end at p; returns the lower end"""
        i = s.sc.put('resistor', 'a', p)
        if label:
            s.sc.text(label, p[0] + 18, p[1] + 30, 'start')
        return i.t['b']

    # ---- rails / ports ------------------------------------------------
    def vdd(s, p, label='V_DD', side='r'):
        """supply bar with its stub ending at p; label beside the bar as in the book"""
        s.sc.put('vdd', 't', p)
        label = s.label(label)
        if side == 'r':
            s.sc.text(label, p[0] + 23, p[1] - 14, 'start')
        else:
            s.sc.text(label, p[0] - 23, p[1] - 14, 'end')

    def gnd(s, p):
        s.sc.put('ground', 't', p)

    def port(s, p, label, side='l', dx=10):
        sc = s.sc
        sc.dot(p, True)
        label = s.label(label)
        if side == 'l':
            sc.text(label, p[0] - dx, p[1], 'end')
        elif side == 'r':
            sc.text(label, p[0] + dx, p[1], 'start')
        elif side == 't':
            sc.text(label, p[0], p[1] - 17, 'middle')
        else:
            sc.text(label, p[0], p[1] + 17, 'middle')

    def w(s, *pts):
        s.sc.wire(*pts)

    def dot(s, p):
        s.sc.dot(p)

    # ---- compound helpers ---------------------------------------------
    @staticmethod
    def top(d):
        return d.d if d.kind == 'n' else d.s

    @staticmethod
    def bot(d):
        return d.s if d.kind == 'n' else d.d

    def chain(s, specs, x, y, gate='left', gap=STEP, link=True):
        """Vertical series stack; specs = [(kind, name, opts)], upper terminal of the first device at (x, y).
        Returns (devices, junction points between them)."""
        devs, js = [], []
        cy = y
        for i, sp in enumerate(specs):
            kind, nm = sp[0], sp[1]
            opts = sp[2] if len(sp) > 2 else {}
            d = s.Q(kind, (x, cy), 'd' if kind == 'n' else 's', gate=opts.get('gate', gate),
                    bulk=opts.get('bulk', False), thick=opts.get('thick', False), name=nm,
                    lbl=opts.get('lbl', 'auto'))
            devs.append(d)
            if i > 0 and link:
                s.w(s.bot(devs[i - 1]), s.top(d))
                js.append((x, (s.bot(devs[i - 1])[1] + s.top(d)[1]) / 2))
            cy = s.bot(d)[1] + gap
        return devs, js

    def gate_bus(s, devs, xin, side=-1):
        """tie the gates (bases) of devs together on a vertical bus at xin"""
        for d in devs:
            s.w(d.g, (xin, d.g[1]))
        ys = [d.g[1] for d in devs]
        s.w((xin, min(ys)), (xin, max(ys)))


def pt(x, y):
    return (x, y)


def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def new(name, title, note=''):
    return Scene(name, title, note)
