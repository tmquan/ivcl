"""Circuit symbols measured from Razavi, "Fundamentals of Microelectronics" (2nd ed.).

Geometry is in BOOK PIXELS of the 150-ppi page scan (1 px = 0.48 pt), obtained by
least-squares fitting of parametric outlines to the printed figures and averaging over
many instances (29 MOSFETs, 22 BJTs, 10 resistors, 7 grounds, ...).  Reference scale =
the dominant figure scale of chapters 7-10.

Two deliberate normalisations (within the scan's +-0.3 px uncertainty):
  * the BJT and the MOSFET share one footprint: same bar length (24.0), same bar weight
    (3.0), same lead weight (1.5), same terminal positions, same bounding box;
  * only two pen weights are used for bars and leads: WIRE (1.5) and THICK (3.0).

Every symbol comes in two forms:
  prims : complete symbol with leads running to grid terminals (`term`)
  body  : the symbol without leads (short nubs only).  A connector is meant to run
          continuously INTO the body and stop at `anchor[terminal]`, so a wire is one
          unbroken line instead of "wire + lead" pieces.

Output units: S output px per book px, uniform for every symbol.

Primitive tuples (output px):
  ('line', pts, width, cap, join)        open polyline
  ('poly', pts, stroke_w, join)          filled polygon (+ outline stroke)
  ('circ', cx, cy, r, stroke_w, filled)
  ('dash', pts, width, on, off)          dashed polyline
"""
import math
import base64

S = 1.5          # output px per book px  (= 2.25 x printed size at 100 px/in)
PAD = 10         # transparent margin of exported symbol images; keeps terminals on the 10-px grid

WIRE = 1.5       # book px: wires and leads (0.72 pt)
THICK = 3.0      # book px: plates, bars, rails (1.44 pt)

# ---------------------------------------------------------------- measured numbers (book px)
MOS = dict(xg=-5.40, wg=THICK, Lg=18.0, wc=THICK, Lc=24.0, ys=6.73, xl=14.40, ws=WIRE,
           aL=7.95, aW=5.65, asw=1.5, n_tip=-0.10, p_tip=0.85)
BJT = dict(wb=THICK, Lb=24.0, xl=16.0, y1=5.40, y2=12.0, wa=WIRE,
           aL=7.85, aW=6.30, asw=1.5, n_back=1.35, p_from=2.30)
RES = dict(a=5.40, q=2.70, w=1.63)                       # 3 peaks per side, mitred
CAP = dict(g=6.70, L=16.0, t=2.9)
IND = dict(R=8.68, k=1.195, h=3.86, e1=4.34, e2=3.77, e3=4.60, e4=4.86, sh=0.30, h2=-0.29, w=1.8, turns=3)
DIO = dict(hb=5.50, ht=12.40, wt=1.8, Lb=13.0, tb=3.1, yb=12.25)
ZEN = dict(Lz=17.0, tab=6.15, tb=3.0)
SRC = dict(r=10.80, wc=1.65)                             # circle of V and I sources (centre-line radius)
ISRC = dict(tail=6.8, ws=1.6, tip=7.0, hL=8.35, hW=8.05, hs=1.8)
VSIGN = dict(px=11.6, py=-13.6, pw=8.1, ph=7.5, ps=1.9, mx=11.55, my=13.3, mw=8.5, ms=1.5)
BAT = dict(L1=30.8, L2=16.25, t=2.5, g=6.65, px=12.1, py=-9.95, pw=7.4, ps=1.5, mx=12.15, my=16.8, mw=7.1, ms=1.5)
GND = dict(b1=13.3, b2=8.1, b3=4.8, t=THICK, s1=5.5, s2=5.15)
RAIL = dict(B=21.4, t=2.9)
RING = dict(r=2.56, w=1.66)                              # open terminal (outer dia 6.8)
DOT = dict(r=3.46)                                       # junction dot (dia 6.9)
OPA = dict(H=53.6, Wd=53.3, wt=3.2, yi=10.9, px=9.05, py=-13.5, mx=8.95, my=13.2, sw=7.9, ss=1.8)
BULK = dict(on=3.0, off=3.0)                             # dashed substrate lead (Fig. 6.29)
NUB = 2.0                                                # book px: stub kept on a lead-free body
# hooked arrow of an impedance (inside back cover), mean of the ten arrows printed there, in book px:
# shaft = corner to the back of the head, head hL x hW, tail = corner to the end that carries the label.
# Two more forms are printed: a short tail (label beside the arrow) and a long shaft with a short tail.
# gap = distance between the arrow and the lead it runs along; lead = how far the corner sits before the terminal.
LOOK = dict(shaft=7.5, long=16.4, hL=19.2, hW=9.2, tail=22.7, short=11.3, gap=12.0, lead=11.0, low=22.5)
LOADBOX = dict(w=10.0, h=24.0, t=2.6)                    # generic load drawn as a box (inside back cover)


def _tri(tip, d, L, Wd):
    dx, dy = d
    n = math.hypot(dx, dy); dx /= n; dy /= n
    bx, by = tip[0] - L * dx, tip[1] - L * dy
    px, py = -dy, dx
    return [tip, (bx + px * Wd / 2, by + py * Wd / 2), (bx - px * Wd / 2, by - py * Wd / 2)]


def _plus(cx, cy, w, h, s):
    return [('line', [(cx - w / 2, cy), (cx + w / 2, cy)], s, 'butt', 'miter'),
            ('line', [(cx, cy - h / 2), (cx, cy + h / 2)], s, 'butt', 'miter')]


def _minus(cx, cy, w, s):
    return [('line', [(cx - w / 2, cy), (cx + w / 2, cy)], s, 'butt', 'miter')]


def _ln(pts, w, cap='butt', join='miter'):
    """a line primitive, or nothing when it has no length (lead-free bodies)"""
    if any(p is None for p in pts):
        return []
    if all(abs(p[0] - pts[0][0]) < 1e-9 and abs(p[1] - pts[0][1]) < 1e-9 for p in pts):
        return []
    return [('line', list(pts), w, cap, join)]


# ---------------------------------------------------------------- builders (book px, local origin)
# every builder takes the lead END coordinates; None = no lead (body form)
def b_mos(kind, gx, top, bot, bulk=False, arrow=True, thick=False, bx=None):
    """origin = channel centre. gx = x of the gate-lead end, top/bot = y of the two vertical lead ends."""
    m = MOS
    wg, xg = m['wg'], m['xg']
    if thick:                       # derived variant (not in the book): heavier gate plate, same gap
        wg = 5.0; xg = m['xg'] + m['wg'] / 2 - wg / 2
    pr = [('line', [(xg, -m['Lg'] / 2), (xg, m['Lg'] / 2)], wg, 'butt', 'miter'),
          ('line', [(0, -m['Lc'] / 2), (0, m['Lc'] / 2)], m['wc'], 'butt', 'miter')]
    if gx is not None:
        pr += _ln([(gx, 0), (xg, 0)], WIRE)
    pr += [('line', [(0, -m['ys']), (m['xl'], -m['ys']), (m['xl'], top)], m['ws'], 'butt', 'miter'),
           ('line', [(0, m['ys']), (m['xl'], m['ys']), (m['xl'], bot)], m['ws'], 'butt', 'miter')]
    if bulk:
        pr.append(('dash', [(0, 0), (bx, 0)], WIRE, BULK['on'], BULK['off']))
    if arrow:
        if kind == 'n':
            pr.append(('poly', _tri((m['xl'] + m['n_tip'], m['ys']), (1, 0), m['aL'], m['aW']), m['asw'], 'round'))
        else:
            pr.append(('poly', _tri((m['p_tip'], -m['ys']), (-1, 0), m['aL'], m['aW']), m['asw'], 'round'))
    return pr


def b_bjt(kind, bx, top, bot):
    """origin = base plate centre"""
    b = BJT
    xl, y1, y2 = b['xl'], b['y1'], b['y2']
    pr = [('line', [(0, -b['Lb'] / 2), (0, b['Lb'] / 2)], b['wb'], 'butt', 'miter')]
    if bx is not None:
        pr += _ln([(bx, 0), (0, 0)], WIRE)
    pr += [('line', [(0, -y1), (xl, -y2), (xl, top)], b['wa'], 'butt', 'miter'),
           ('line', [(0, y1), (xl, y2), (xl, bot)], b['wa'], 'butt', 'miter')]
    n = math.hypot(xl, y2 - y1); u = (xl / n, (y2 - y1) / n)
    if kind == 'npn':
        tip = (xl - u[0] * b['n_back'], y2 - u[1] * b['n_back'])
        pr.append(('poly', _tri(tip, u, b['aL'], b['aW']), b['asw'], 'round'))
    else:
        u2 = (u[0], -u[1])
        tip = (u2[0] * b['p_from'], -y1 + u2[1] * b['p_from'])
        pr.append(('poly', _tri(tip, (-u2[0], -u2[1]), b['aL'], b['aW']), b['asw'], 'round'))
    return pr


def b_res(top, bot, first=-1):
    a, q = RES['a'], RES['q']
    pts = [(0, top), (0, -3 * q)] + [(first * a * (-1) ** k, -3 * q + q / 2 + k * q) for k in range(6)] + [(0, 3 * q), (0, bot)]
    return [('line', pts, RES['w'], 'butt', 'miter')]


def b_cap(top, bot):
    g, L, t = CAP['g'], CAP['L'], CAP['t']
    pr = []
    if top is not None:
        pr += _ln([(0, top), (0, -g / 2)], WIRE)
    if bot is not None:
        pr += _ln([(0, g / 2), (0, bot)], WIRE)
    return pr + [('line', [(-L / 2, -g / 2), (L / 2, -g / 2)], t, 'butt', 'miter'),
                 ('line', [(-L / 2, g / 2), (L / 2, g / 2)], t, 'butt', 'miter')]


def ind_extent():
    p = IND; T = 2 * math.pi * p['turns']; mid = T / 2
    ya = p['k'] * (math.pi / 2 - mid) + p['h']
    yb = p['k'] * (T - math.pi / 2 - mid) - p['h']
    return ya - p['e2'] - p['e1'], yb + p['e3'] + p['e4']


def b_ind(top, bot, nseg=48):
    p = IND; R, k, h = p['R'], p['k'], p['h']
    T = 2 * math.pi * p['turns']; mid = T / 2
    hel = lambda th: (R * math.cos(th), k * (th - mid) + h * math.sin(th) + p['sh'] * math.cos(th) + p['h2'] * math.sin(2 * th))
    pts = []
    ya = hel(math.pi / 2)[1]; yext = ya - p['e2']
    for i in range(13):
        a = -math.pi / 2 + (math.pi / 2) * i / 12; pts.append((R * math.cos(a), yext + p['e1'] * math.sin(a)))
    for i in range(1, 13):
        a = (math.pi / 2) * i / 12; pts.append((R * math.cos(a), yext + p['e2'] * math.sin(a)))
    N = int(nseg * (p['turns'] - 0.5))
    for i in range(1, N + 1):
        pts.append(hel(math.pi / 2 + (T - math.pi) * i / N))
    yb = hel(T - math.pi / 2)[1]; yext2 = yb + p['e3']
    for i in range(1, 13):
        a = -math.pi / 2 + (math.pi / 2) * i / 12; pts.append((R * math.cos(a), yext2 + p['e3'] * math.sin(a)))
    for i in range(1, 13):
        a = (math.pi / 2) * i / 12; pts.append((R * math.cos(a), yext2 + p['e4'] * math.sin(a)))
    return [('line', [(0, top), pts[0]], WIRE, 'butt', 'round'), ('line', pts, p['w'], 'round', 'round'),
            ('line', [pts[-1], (0, bot)], WIRE, 'butt', 'round')]


def b_diode(top, bot):
    """origin = centre of the triangle base (anode side); points down"""
    d = DIO; hb, ht = d['hb'], d['ht']
    pr = []
    if top is not None:
        pr += _ln([(0, top), (0, 0)], WIRE)
    pr += [('line', [(-hb, 0), (hb, 0), (0, ht), (-hb, 0), (hb, 0)], d['wt'], 'butt', 'round'),
           ('line', [(-d['Lb'] / 2, d['yb']), (d['Lb'] / 2, d['yb'])], d['tb'], 'butt', 'miter')]
    if bot is not None:
        pr += _ln([(0, d['yb']), (0, bot)], WIRE)
    return pr


def b_zener(top, bot):
    """as printed in Fig. 3.45: cathode bar on TOP, triangle points up. origin = bar centre."""
    d = DIO; z = ZEN; hb, ht = d['hb'], d['ht']; L, t = z['Lz'], z['tb']
    yb = d['yb']
    pr = []
    if top is not None:
        pr += _ln([(0, top), (0, 0)], WIRE)
    pr += [('line', [(-L / 2 + t / 2, z['tab']), (-L / 2 + t / 2, 0), (L / 2 - t / 2, 0), (L / 2 - t / 2, -z['tab'])], t, 'butt', 'miter'),
           ('line', [(-hb, yb), (hb, yb), (0, yb - ht), (-hb, yb), (hb, yb)], d['wt'], 'butt', 'round')]
    if bot is not None:
        pr += _ln([(0, yb), (0, bot)], WIRE)
    return pr


def b_vsrc(top, bot):
    r = SRC['r']; v = VSIGN
    pr = [('circ', 0, 0, r, SRC['wc'], False)]
    if top is not None:
        pr += _ln([(0, top), (0, -r)], WIRE)
    if bot is not None:
        pr += _ln([(0, r), (0, bot)], WIRE)
    return pr + _plus(v['px'], v['py'], v['pw'], v['ph'], v['ps']) + _minus(v['mx'], v['my'], v['mw'], v['ms'])


def b_isrc(top, bot):
    r = SRC['r']; i = ISRC
    pr = [('circ', 0, 0, r, SRC['wc'], False)]
    if top is not None:
        pr += _ln([(0, top), (0, -r)], WIRE)
    if bot is not None:
        pr += _ln([(0, r), (0, bot)], WIRE)
    return pr + [('line', [(0, i['tail']), (0, -i['tip'] + i['hL'])], i['ws'], 'butt', 'miter'),
                 ('poly', _tri((0, -i['tip']), (0, -1), i['hL'], i['hW']), i['hs'], 'round')]


def b_battery(top, bot):
    """origin = centre of the long (+) plate"""
    b = BAT
    pr = []
    if top is not None:
        pr += _ln([(0, top), (0, 0)], WIRE)
    pr += [('line', [(-b['L1'] / 2, 0), (b['L1'] / 2, 0)], b['t'], 'butt', 'miter'),
           ('line', [(-b['L2'] / 2, b['g']), (b['L2'] / 2, b['g'])], b['t'], 'butt', 'miter')]
    if bot is not None:
        pr += _ln([(0, b['g']), (0, bot)], WIRE)
    return pr + _plus(b['px'], b['py'], b['pw'], b['pw'], b['ps']) + _minus(b['mx'], b['my'], b['mw'], b['ms'])


def b_ground(top):
    """origin = centre of top bar"""
    g = GND
    pr = []
    if top is not None:
        pr += _ln([(0, top), (0, 0)], WIRE)
    return pr + [('line', [(-g['b1'] / 2, 0), (g['b1'] / 2, 0)], g['t'], 'butt', 'miter'),
                 ('line', [(-g['b2'] / 2, g['s1']), (g['b2'] / 2, g['s1'])], g['t'], 'butt', 'miter'),
                 ('line', [(-g['b3'] / 2, g['s1'] + g['s2']), (g['b3'] / 2, g['s1'] + g['s2'])], g['t'], 'butt', 'miter')]


def b_rail(end):
    """origin = bar centre; stub runs to y=end"""
    pr = [('line', [(-RAIL['B'] / 2, 0), (RAIL['B'] / 2, 0)], RAIL['t'], 'butt', 'miter')]
    if end is not None:
        pr += _ln([(0, 0), (0, end)], WIRE)
    return pr


def b_opamp(xin, xout):
    """origin = centre of the vertical edge"""
    o = OPA; H, Wd = o['H'], o['Wd']
    pr = [('line', [(0, -H / 2), (0, H / 2), (Wd, 0), (0, -H / 2), (0, H / 2)], o['wt'], 'butt', 'miter')]
    if xin is not None:
        pr += _ln([(xin, -o['yi']), (0, -o['yi'])], WIRE) + _ln([(xin, o['yi']), (0, o['yi'])], WIRE)
    if xout is not None:
        pr += _ln([(Wd, 0), (xout, 0)], WIRE)
    return pr + _plus(o['px'], o['py'], o['sw'], o['sw'], o['ss']) + _minus(o['mx'], o['my'], o['sw'], o['ss'])


# ---------------------------------------------------------------- book px -> output px
def _conv(prims, ox, oy):
    f = lambda p: (round(ox + S * p[0], 3), round(oy + S * p[1], 3))
    out = []
    for p in prims:
        k = p[0]
        if k == 'line':
            out.append(('line', [f(q) for q in p[1]], round(p[2] * S, 3), p[3], p[4]))
        elif k == 'poly':
            out.append(('poly', [f(q) for q in p[1]], round(p[2] * S, 3), p[3]))
        elif k == 'circ':
            c = f((p[1], p[2])); out.append(('circ', c[0], c[1], round(p[3] * S, 3), round(p[4] * S, 3), p[5]))
        elif k == 'dash':
            out.append(('dash', [f(q) for q in p[1]], round(p[2] * S, 3), round(p[3] * S, 3), round(p[4] * S, 3)))
    return out


def _sym(w, h, ox, oy, full, body, term, anchor, in_book=True):
    """full(B)/body() -> prims in book px.  B(x=, y=) converts an output coordinate to local book px.
    anchor: {terminal: (bx, by)} in book px = where a connector ends inside the body."""
    B = lambda x=None, y=None: ((x - ox) / S if x is not None else None, (y - oy) / S if y is not None else None)
    anc = {k: (round(ox + S * v[0], 3), round(oy + S * v[1], 3)) for k, v in anchor.items()}
    return dict(w=w, h=h, prims=_conv(full(B), ox, oy), body=_conv(body(), ox, oy), term=term, anchor=anc,
                origin=(ox, oy), in_book=in_book)


def mos(kind='n', bulk=False, arrow=True, thick=False, h=80):
    """h=80: leads to the 10-px grid (g at (0,h/2), d/s at (40,0)/(40,h)).  h=40 gives the compact form."""
    m = MOS
    ox, oy = 40 - S * m['xl'], h / 2
    term = {'g': (0, h / 2)}
    term.update(dict(d=(40, 0), s=(40, h)) if kind == 'n' else dict(s=(40, 0), d=(40, h)))
    top, bot = ('d', 's') if kind == 'n' else ('s', 'd')
    anchor = {'g': (m['xg'], 0), top: (m['xl'], -m['ys']), bot: (m['xl'], m['ys'])}
    if bulk:
        term['b'] = (40, h / 2); anchor['b'] = (m['xl'], 0)
    nub = 20.0 / S
    return _sym(40, h, ox, oy,
                lambda B: b_mos(kind, B(x=0)[0], B(y=0)[1], B(y=h)[1], bulk, arrow, thick, B(x=40)[0]),
                lambda: b_mos(kind, None, -nub, nub, bulk, arrow, thick, m['xl']),
                term, anchor, in_book=(arrow and not thick and not (bulk and kind == 'p')))


def bjt(kind='npn', h=80):
    b = BJT
    ox, oy = 40 - S * b['xl'], h / 2
    term = dict(b=(0, h / 2), c=(40, 0), e=(40, h)) if kind == 'npn' else dict(b=(0, h / 2), e=(40, 0), c=(40, h))
    top, bot = ('c', 'e') if kind == 'npn' else ('e', 'c')
    anchor = {'b': (0, 0), top: (b['xl'], -b['y2']), bot: (b['xl'], b['y2'])}
    nub = 20.0 / S
    return _sym(40, h, ox, oy,
                lambda B: b_bjt(kind, B(x=0)[0], B(y=0)[1], B(y=h)[1]),
                lambda: b_bjt(kind, None, -nub, nub),
                term, anchor)


def resistor():
    q = RES['q']
    return _sym(20, 60, 10, 30, lambda B: b_res(B(y=0)[1], B(y=60)[1]), lambda: b_res(-3 * q - NUB, 3 * q + NUB),
                dict(a=(10, 0), b=(10, 60)), dict(a=(0, -3 * q), b=(0, 3 * q)))


def capacitor():
    g = CAP['g']
    return _sym(20, 40, 10, 20, lambda B: b_cap(B(y=0)[1], B(y=40)[1]), lambda: b_cap(None, None),
                dict(a=(10, 0), b=(10, 40)), dict(a=(0, -g / 2), b=(0, g / 2)))


def inductor():
    y0, y1 = ind_extent()
    oy = 30 - S * (y0 + y1) / 2
    return _sym(20, 60, 10, oy, lambda B: b_ind(B(y=0)[1], B(y=60)[1]), lambda: b_ind(y0 - NUB, y1 + NUB),
                dict(a=(10, 0), b=(10, 60)), dict(a=(0, y0), b=(0, y1)))


def diode():
    oy = 30 - S * (DIO['yb'] + DIO['tb'] / 2 - DIO['wt'] / 2) / 2
    return _sym(20, 60, 10, oy, lambda B: b_diode(B(y=0)[1], B(y=60)[1]), lambda: b_diode(None, None),
                dict(a=(10, 0), k=(10, 60)), dict(a=(0, 0), k=(0, DIO['yb'])))


def zener():
    oy = 30 - S * (DIO['yb'] + DIO['wt'] / 2 - ZEN['tab']) / 2
    return _sym(20, 60, 10, oy, lambda B: b_zener(B(y=0)[1], B(y=60)[1]), lambda: b_zener(None, None),
                dict(k=(10, 0), a=(10, 60)), dict(k=(0, 0), a=(0, DIO['yb'])))


def vsource():
    r = SRC['r']
    return _sym(40, 80, 20, 40, lambda B: b_vsrc(B(y=0)[1], B(y=80)[1]), lambda: b_vsrc(None, None),
                dict(p=(20, 0), n=(20, 80)), dict(p=(0, -r), n=(0, r)))


def isource():
    r = SRC['r']
    return _sym(40, 80, 20, 40, lambda B: b_isrc(B(y=0)[1], B(y=80)[1]), lambda: b_isrc(None, None),
                dict(top=(20, 0), bot=(20, 80)), dict(top=(0, -r), bot=(0, r)))


def battery():
    oy = 20 - S * BAT['g'] / 2
    return _sym(40, 40, 20, oy, lambda B: b_battery(B(y=0)[1], B(y=40)[1]), lambda: b_battery(None, None),
                dict(p=(20, 0), n=(20, 40)), dict(p=(0, 0), n=(0, BAT['g'])))


def ground():
    return _sym(20, 30, 10, 10, lambda B: b_ground(B(y=0)[1]), lambda: b_ground(None), dict(t=(10, 0)), dict(t=(0, 0)))


def vdd():
    return _sym(40, 10, 20, 0, lambda B: b_rail(B(y=10)[1]), lambda: b_rail(None), dict(t=(20, 10)), dict(t=(0, 0)))


def vee():
    return _sym(40, 10, 20, 10, lambda B: b_rail(B(y=0)[1]), lambda: b_rail(None), dict(t=(20, 0)), dict(t=(0, 0)))


def port():
    f = lambda *a: [('circ', 0, 0, RING['r'], RING['w'], False)]
    return _sym(10, 10, 5, 5, f, f, dict(t=(5, 5)), dict(t=(0, 0)))


def junction():
    f = lambda *a: [('circ', 0, 0, DOT['r'], 0, True)]
    return _sym(10, 10, 5, 5, f, f, dict(t=(5, 5)), dict(t=(0, 0)))


def opamp():
    oy = 50
    yi = S * OPA['yi']
    return _sym(100, 100, 10, oy, lambda B: b_opamp(B(x=0)[0], B(x=100)[0]), lambda: b_opamp(None, None),
                dict(inp=(0, round(oy - yi, 3)), inn=(0, round(oy + yi, 3)), out=(100, 50)),
                dict(inp=(0, -OPA['yi']), inn=(0, OPA['yi']), out=(OPA['Wd'], 0)))


def b_load(top, bot):
    w, h, t = LOADBOX['w'] / 2, LOADBOX['h'] / 2, LOADBOX['t']
    pr = [('line', [(0, -h), (w, -h), (w, h), (-w, h), (-w, -h), (0, -h)], t, 'butt', 'miter')]
    if top is not None:
        pr += _ln([(0, top), (0, -h)], WIRE)
    if bot is not None:
        pr += _ln([(0, h), (0, bot)], WIRE)
    return pr


def load():
    """generic load: a box between two leads"""
    h = LOADBOX['h'] / 2
    return _sym(20, 60, 10, 30, lambda B: b_load(B(y=0)[1], B(y=60)[1]), lambda: b_load(-h - NUB, h + NUB),
                dict(a=(10, 0), b=(10, 60)), dict(a=(0, -h), b=(0, h)), in_book=False)


def look(shaft=None, tail=None):
    """arrow with a 90-degree corner that marks an impedance: the head points into the circuit,
    the tail leads to the label.  Corner at the origin, head along +x, tail along +y.
    The line has the weight of a wire and the corner is sharp, as printed."""
    k = LOOK
    shaft, tail = shaft or k['shaft'], tail or k['tail']
    L = shaft + k['hL']
    f = lambda *a: [('line', [(0, tail), (0, 0), (shaft + 0.5, 0)], WIRE, 'butt', 'miter'),
                    ('poly', _tri((L, 0), (1, 0), k['hL'], k['hW']), 0.0, 'miter')]
    w, h = round(S * L, 3), round(S * tail, 3)
    return _sym(w, h, 0, 0, f, f, dict(corner=(0, 0), tip=(w, 0), tail=(0, h)),
                dict(corner=(0, 0), tip=(L, 0), tail=(0, tail)), in_book=False)


# ---------------------------------------------------------------- catalogue
TRANSISTORS = {
    'nmos': lambda h: mos('n', h=h), 'pmos': lambda h: mos('p', h=h),
    'nmos_b': lambda h: mos('n', bulk=True, h=h), 'pmos_b': lambda h: mos('p', bulk=True, h=h),
    'nmos_na': lambda h: mos('n', arrow=False, h=h), 'pmos_na': lambda h: mos('p', arrow=False, h=h),
    'nmos_thick': lambda h: mos('n', thick=True, h=h),
    'npn': lambda h: bjt('npn', h=h), 'pnp': lambda h: bjt('pnp', h=h),
}
CATALOG = [
    ('nmos', 'NMOS', TRANSISTORS['nmos'](80)),
    ('pmos', 'PMOS', TRANSISTORS['pmos'](80)),
    ('nmos_b', 'NMOS with substrate lead (Fig. 6.29)', TRANSISTORS['nmos_b'](80)),
    ('npn', 'npn BJT', TRANSISTORS['npn'](80)),
    ('pnp', 'pnp BJT', TRANSISTORS['pnp'](80)),
    ('resistor', 'Resistor', resistor()),
    ('capacitor', 'Capacitor', capacitor()),
    ('inductor', 'Inductor', inductor()),
    ('diode', 'Diode', diode()),
    ('zener', 'Zener diode', zener()),
    ('vsource', 'Voltage source', vsource()),
    ('isource', 'Current source', isource()),
    ('battery', 'Battery (dc source)', battery()),
    ('ground', 'Ground', ground()),
    ('vdd', 'Supply rail (VDD / VCC)', vdd()),
    ('vee', 'Supply rail (VEE / VSS)', vee()),
    ('port', 'Terminal', port()),
    ('junction', 'Node dot', junction()),
    ('opamp', 'Op amp (Fig. 8.1)', opamp()),
    # variants needed by the two-transistor circuits; built from the book MOSFET but not printed in the book
    ('pmos_b', 'PMOS with substrate lead (derived)', TRANSISTORS['pmos_b'](80)),
    ('nmos_na', 'NMOS without arrow (derived)', TRANSISTORS['nmos_na'](80)),
    ('pmos_na', 'PMOS without arrow (derived)', TRANSISTORS['pmos_na'](80)),
    ('nmos_thick', 'NMOS thick-oxide (derived)', TRANSISTORS['nmos_thick'](80)),
]
SYM = {k: v for k, _, v in CATALOG}
TITLE = {k: t for k, t, _ in CATALOG}
COMPACT = {k: f(40) for k, f in TRANSISTORS.items()}      # 40 x 40 transistors with short leads
# used by the figures only, not part of the library
SYM.update(load=load(), look=look(), look_short=look(tail=LOOK['short']), look_long=look(LOOK['long'], LOOK['short']))
TITLE.update(load='Load', look='Impedance arrow', look_short='Impedance arrow, short tail',
             look_long='Impedance arrow, long shaft')


# ---------------------------------------------------------------- transform / export
def xf_point(p, w, h, rot, mirror):
    x, y = p
    if mirror:
        x = w - x
    cw, ch = w, h
    for _ in range(rot % 4):
        x, y = ch - y, x
        cw, ch = ch, cw
    return (x, y)


def oriented(sym, rot=0, mirror=False):
    w, h = sym['w'], sym['h']
    return (w, h) if rot % 2 == 0 else (h, w)


def _pts(pts):
    return ' '.join('%.2f,%.2f' % p for p in pts)


def prim_svg(pr):
    k = pr[0]
    if k == 'line':
        return ('<polyline points="%s" fill="none" stroke-width="%.3f" stroke-linecap="%s" stroke-linejoin="%s" '
                'stroke-miterlimit="10"/>' % (_pts(pr[1]), pr[2], pr[3], pr[4]))
    if k == 'dash':
        return ('<polyline points="%s" fill="none" stroke-width="%.3f" stroke-linecap="butt" '
                'stroke-dasharray="%.3f %.3f"/>' % (_pts(pr[1]), pr[2], pr[3], pr[4]))
    if k == 'poly':
        if pr[2] > 0:
            return '<polygon points="%s" fill="#000" stroke-width="%.3f" stroke-linejoin="%s"/>' % (_pts(pr[1]), pr[2], pr[3])
        return '<polygon points="%s" fill="#000" stroke="none"/>' % _pts(pr[1])
    if k == 'circ':
        if pr[5]:
            return '<circle cx="%.2f" cy="%.2f" r="%.3f" fill="#000" stroke="none"/>' % (pr[1], pr[2], pr[3])
        return '<circle cx="%.2f" cy="%.2f" r="%.3f" fill="#fff" stroke-width="%.3f"/>' % (pr[1], pr[2], pr[3], pr[4])
    raise ValueError(k)


def svg_body(sym, rot=0, mirror=False, which='prims'):
    w, h = sym['w'], sym['h']
    cw, ch = oriented(sym, rot, mirror)
    gtf = 'translate(%s,0) scale(-1,1)' % w if mirror else ''
    rots = {0: '', 1: 'translate(%s,0) rotate(90)' % h, 2: 'translate(%s,%s) rotate(180)' % (w, h),
            3: 'translate(0,%s) rotate(270)' % w}[rot % 4]
    body = ''.join(prim_svg(p) for p in sym[which])
    tf = (rots + ' ' + gtf).strip()
    return cw, ch, ('<g transform="%s">%s</g>' % (tf, body)) if tf else '<g>%s</g>' % body


def svg_doc(sym, rot=0, mirror=False, pad=PAD, which='prims'):
    cw, ch, body = svg_body(sym, rot, mirror, which)
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="%d %d %d %d" '
            'stroke="#000" fill="none">%s</svg>'
            % (cw + 2 * pad, ch + 2 * pad, -pad, -pad, cw + 2 * pad, ch + 2 * pad, body))


def terminals(sym, rot=0, mirror=False):
    return {k: xf_point(v, sym['w'], sym['h'], rot, mirror) for k, v in sym['term'].items()}


def anchors(sym, rot=0, mirror=False):
    return {k: xf_point(v, sym['w'], sym['h'], rot, mirror) for k, v in sym['anchor'].items()}


def conn_points(sym, rot=0, mirror=False, pad=PAD, which='term'):
    """draw.io connection constraints (relative to the padded image bounds)"""
    cw, ch = oriented(sym, rot, mirror)
    pts = terminals(sym, rot, mirror) if which == 'term' else anchors(sym, rot, mirror)
    return [[round((x + pad) / (cw + 2 * pad), 4), round((y + pad) / (ch + 2 * pad), 4)] for x, y in pts.values()]


def data_uri(svg):
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()
