"""One transistor in eight connections, and forty-two circuits of two transistors.

Each function draws one circuit through a drawing context `c`.  It is written against the base
class `Transistor` (terminals .bg, .cd, .es) and is drawn twice: once with the subclass `BJT`
(.b, .c, .e; npn / pnp; Q_1; V_CC) and once with the subclass `MOSFET` (.g, .d, .s; NMOS / PMOS;
M_1; V_DD).  The first line of the docstring names the circuit, as  "BJT name | MOSFET name"
where the two differ.  A name "-" means that the circuit has no such version: that cell of
the figure says so.
Coordinates are px on a 10-px grid; the main column of leads is at x = 0.

The two-transistor circuits follow Pretl & Eberlein, "Fifty Nifty Variations of Two-Transistor
Circuits", IEEE Solid-State Circuits Magazine 13(3), 2021.
"""
from .ckt import mid, RED, MOSFET

IN, OUT = 'V_in', 'V_out'


# ================================================================ types of transistors
def n_type(c):
    """npn | NMOS"""
    t = c.transistor('n', (0, 0), 'cd')
    c.w(t.cd, (0, -30)); c.port((0, -30), t.pin('cd'), 't')
    c.w(t.es, (0, 110)); c.port((0, 110), t.pin('es'), 'b')
    c.w(t.bg, (t.bg[0] - 30, 40)); c.port((t.bg[0] - 30, 40), t.pin('bg'), 'l')


def p_type(c):
    """pnp | PMOS"""
    t = c.transistor('p', (0, 0), 'es')
    c.w(t.es, (0, -30)); c.port((0, -30), t.pin('es'), 't')
    c.w(t.cd, (0, 110)); c.port((0, 110), t.pin('cd'), 'b')
    c.w(t.bg, (t.bg[0] - 30, 40)); c.port((t.bg[0] - 30, 40), t.pin('bg'), 'l')


# ================================================================ one transistor: eight connections
def role_amplifier(c):
    """Common emitter | Common source"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    c.supply((0, -130)); c.w((0, -130), (0, -110))
    out = (0, -20)
    c.w(c.res((0, -110), 'R_' + t1.pin('cd')), t1.cd); c.dot(out)
    c.w(out, (50, -20)); c.port((50, -20), OUT, 'r')
    c.w(t1.es, (0, 100)); c.gnd((0, 100))
    c.w(t1.bg, (t1.bg[0] - 30, 40)); c.port((t1.bg[0] - 30, 40), IN, 'l')


def role_follower(c):
    """Emitter follower | Source follower"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    c.supply((0, -20)); c.w((0, -20), t1.cd)
    out = (0, 100)
    c.w(t1.es, (0, 130)); c.dot(out)
    c.w(out, (50, 100)); c.port((50, 100), OUT, 'r')
    c.w(c.res((0, 130), 'R_' + t1.pin('es')), (0, 210)); c.gnd((0, 210))
    c.w(t1.bg, (t1.bg[0] - 30, 40)); c.port((t1.bg[0] - 30, 40), IN, 'l')


def role_current_buffer(c):
    """Common base | Common gate"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    c.w(t1.cd, (0, -40)); c.port((0, -40), 'I_out', 'r')
    c.w(t1.es, (0, 120)); c.port((0, 120), 'I_in', 'r'); c.look((0, 120), 'u', 'l', 'R_in')
    c.w(t1.bg, (t1.bg[0] - 30, 40)); c.port((t1.bg[0] - 30, 40), 'V_bias', 'l')


def role_current_source(c):
    """Current source"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    c.w(t1.cd, (0, -40)); c.port((0, -40), 'I_out', 'l'); c.look((0, -40), 'd', 'r', 'R_out')
    c.w(t1.es, (0, 100)); c.gnd((0, 100))
    c.w(t1.bg, (t1.bg[0] - 30, 40)); c.port((t1.bg[0] - 30, 40), 'V_bias', 'l')


def role_diode(c):
    """Diode-connected transistor"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    top = (0, -20)
    c.w(t1.cd, (0, -70)); c.dot(top); c.port((0, -70), 'V_p', 'l'); c.look((0, -70), 'd', 'r', 'R')
    c.w(t1.bg, (t1.bg[0] - 20, 40), (t1.bg[0] - 20, -20), top)
    c.w(t1.es, (0, 120)); c.port((0, 120), 'V_n', 'l')


def role_switch(c):
    """Switch"""
    t1 = c.transistor('n', (0, 0), 'cd', side='right', rot=3, name=1)
    c.w(t1.cd, (t1.cd[0] - 40, 0)); c.port((t1.cd[0] - 40, 0), 'V_1', 'l')
    c.w(t1.es, (t1.es[0] + 40, 0)); c.port((t1.es[0] + 40, 0), 'V_2', 'r')
    c.w(t1.bg, (t1.bg[0], t1.bg[1] - 20)); c.port((t1.bg[0], t1.bg[1] - 20), 'V_on', 't')
    if c.device is MOSFET:                                  # only the MOSFET is a resistance when it is on
        c.look((t1.cd[0] - 40, 0), 'r', 'd', 'R_on')


def role_resistor(c):
    """- | Resistor"""
    t1 = c.transistor('n', (0, 0), 'cd', side='right', rot=3, name=1)
    c.w(t1.cd, (t1.cd[0] - 40, 0)); c.port((t1.cd[0] - 40, 0), 'V_p', 'l'); c.look((t1.cd[0] - 40, 0), 'r', 'd', 'R_on')
    c.w(t1.es, (t1.es[0] + 40, 0)); c.port((t1.es[0] + 40, 0), 'V_n', 'r')
    c.w(t1.bg, (t1.bg[0], t1.bg[1] - 20)); c.port((t1.bg[0], t1.bg[1] - 20), 'V_bias', 't')


def role_capacitor(c):
    """Capacitor"""
    t1 = c.transistor('n', (0, 0), 'bg', side='right', rot=3, name=1)
    c.w(t1.bg, (0, -40)); c.port((0, -40), 'V_p', 'r')
    yb = t1.cd[1] + 20
    c.w(t1.cd, (t1.cd[0], yb), (t1.es[0], yb), t1.es); c.dot((0, yb))
    c.w((0, yb), (0, yb + 40)); c.port((0, yb + 40), 'V_n', 'r')


# ================================================================ one transistor: input and output impedances
# The ten drawings inside the back cover of Razavi.  The arrow carries the impedance, as printed.
def into_base(c):
    """Into the base | Into the gate"""
    t = c.transistor('n', (0, 0), 'cd')
    c.w(t.cd, (0, -20))
    c.w(t.es, (0, 100)); c.ac((0, 100))
    p = (t.bg[0] - 20, 40)
    c.w(t.bg, p); c.port(p, None); c.look(p, 'r', 'd', c.pick(r'$r_\pi$', r'$\approx\infty$'))


def into_base_degenerated(c):
    """Into the base, with emitter degeneration | -"""
    t = c.transistor('n', (0, 0), 'cd'); t.note(t.ideal)
    c.w(t.cd, (0, -20))
    c.w(t.es, (0, 100)); c.w(c.res((0, 100), 'R_' + t.pin('es')), (0, 180)); c.ac((0, 180))
    p = (t.bg[0] - 20, 40)
    c.w(t.bg, p); c.port(p, None); c.look(p, 'r', 'd', r'$r_\pi+(\beta+1)R_E$')


def into_collector(c):
    """Into the collector | Into the drain"""
    t = c.transistor('n', (0, 0), 'cd')
    c.w(t.bg, (-60, 40), (-60, 100)); c.ac((-60, 100))
    c.w(t.es, (0, 100)); c.ac((0, 100))
    p = (0, -20)
    c.w(t.cd, p); c.port(p, None); c.look(p, 'd', 'r', '$r_O$')


def into_collector_degenerated(c):
    """Into the collector, with emitter degeneration | Into the drain, with source degeneration"""
    t = c.transistor('n', (0, 0), 'cd')
    c.w(t.bg, (-60, 40), (-60, 100)); c.ac((-60, 100))
    c.w(t.es, (0, 100)); c.w(c.res((0, 100), 'R_' + t.pin('es')), (0, 180)); c.ac((0, 180))
    p = (0, -20)
    c.w(t.cd, p); c.port(p, None)
    c.look(p, 'd', 'r', c.pick(r'$(1+g_mr_O)(R_E\|r_\pi)+r_O$', r'$(1+g_mr_O)R_S+r_O$'), form='long')


def into_emitter(c):
    """Into the emitter | Into the source"""
    t = c.transistor('n', (0, 0), 'cd'); t.note(t.ideal)
    c.supply((0, -110), side=None); c.w((0, -110), (0, -80)); c.w(c.load((0, -80)), t.cd)
    c.w(t.bg, (-60, 40), (-60, 100)); c.ac((-60, 100))
    p = (0, 110)
    c.w(t.es, p); c.port(p, None); c.look(p, 'u', 'r', r'$\dfrac{1}{g_m}$', form='low')


def into_emitter_base_resistance(c):
    """Into the emitter, with base resistance | -"""
    t = c.transistor('n', (0, 0), 'cd'); t.note(t.ideal)
    c.supply((0, -110), side=None); c.w((0, -110), (0, -80)); c.w(c.load((0, -80)), t.cd)
    c.w(t.bg, (t.bg[0] - 10, 40))
    left = c.res((t.bg[0] - 10, 40), 'R_' + t.pin('bg'), to='left', side='u')
    c.w(left, (left[0] - 20, 40), (left[0] - 20, 100)); c.ac((left[0] - 20, 100))
    p = (0, 110)
    c.w(t.es, p); c.port(p, None); c.look(p, 'u', 'r', r'$\dfrac{1}{g_m}+\dfrac{R_B}{\beta+1}$', form='low')


# ================================================================ one transistor: voltage gain
# The eight stages inside the back cover of Razavi.  The gain, printed under each stage, is its note.
def gain_common_emitter(c):
    """Common emitter | Common source"""
    t = c.transistor('n', (0, 0), 'cd')
    C = t.pin('cd')
    c.supply((0, -130)); c.w((0, -130), (0, -110))
    out = (0, -20)
    c.w(c.res((0, -110), 'R_' + C, side='l'), t.cd); c.dot(out)
    c.w(out, (50, -20)); c.port((50, -20), 'v_out', 'r')
    c.w(t.es, (0, 100)); c.gnd((0, 100))
    c.w(t.bg, (t.bg[0] - 30, 40)); c.port((t.bg[0] - 30, 40), 'v_in', 'l')


def gain_degenerated(c):
    """Degenerated common emitter | Degenerated common source"""
    t = c.transistor('n', (0, 0), 'cd'); t.note(t.ideal)
    C, E = t.pin('cd'), t.pin('es')
    c.supply((0, -130)); c.w((0, -130), (0, -110))
    out = (0, -20)
    c.w(c.res((0, -110), 'R_' + C, side='l'), t.cd); c.dot(out)
    c.w(out, (50, -20)); c.port((50, -20), 'v_out', 'r')
    c.w(t.es, (0, 100)); c.w(c.res((0, 100), 'R_' + E), (0, 180)); c.gnd((0, 180))
    c.w(t.bg, (t.bg[0] - 30, 40)); c.port((t.bg[0] - 30, 40), 'v_in', 'l')


def gain_common_base(c):
    """Common base | Common gate"""
    t = c.transistor('n', (0, 0), 'cd', side='right'); t.note(t.ideal)
    C, E = t.pin('cd'), t.pin('es')
    c.supply((0, -130)); c.w((0, -130), (0, -110))
    out = (0, -20)
    c.w(c.res((0, -110), 'R_' + C, side='l'), t.cd); c.dot(out)
    c.w(out, (50, -20)); c.port((50, -20), 'v_out', 'r')
    c.w(t.bg, (t.bg[0] + 10, 40)); c.port((t.bg[0] + 10, 40), 'V_b', 'r', solid=True)
    c.w(t.es, (0, 100), (-20, 100))
    left = c.res((-20, 100), 'R_' + E, to='left')
    c.w(left, (left[0] - 20, 100)); c.port((left[0] - 20, 100), 'v_in', 'l')


def gain_follower(c):
    """Emitter follower | Source follower"""
    t = c.transistor('n', (0, 0), 'cd')
    E = t.pin('es')
    c.supply((0, -20)); c.w((0, -20), t.cd)
    out = (0, 100)
    c.w(t.es, (0, 130)); c.dot(out)
    c.w(out, (50, 100)); c.port((50, 100), 'v_out', 'r')
    c.w(c.res((0, 130), 'R_' + E), (0, 210)); c.gnd((0, 210))
    c.w(t.bg, (t.bg[0] - 30, 40)); c.port((t.bg[0] - 30, 40), 'v_in', 'l')


# ================================================================ helpers shared by several circuits
def capacitor_to_ground(c, node):
    """second transistor, connected as a capacitor from `node` to ground"""
    t2 = c.transistor('n', (node[0], node[1] + 40), 'bg', side='right', rot=3, name=2)
    c.w(node, t2.bg)
    yb = t2.cd[1] + 20
    c.w(t2.cd, (t2.cd[0], yb), (t2.es[0], yb), t2.es)
    c.dot((node[0], yb)); c.w((node[0], yb), (node[0], yb + 10)); c.gnd((node[0], yb + 10))


def pass_device_and_capacitor(c, control, resistance=None):
    """first transistor as a pass device from the input to a node that the second one loads as a capacitor;
    `resistance` names the resistance of the pass device, if the note of the circuit uses it"""
    t1 = c.transistor('n', (0, 0), 'cd', side='right', rot=3, name=1)
    c.w(t1.cd, (-30, 0)); c.port((-30, 0), IN, 'l')
    if resistance:
        c.look((-30, 0), 'r', 'd', resistance)
    c.w(t1.bg, (t1.bg[0], t1.bg[1] - 20)); c.port((t1.bg[0], t1.bg[1] - 20), control, 't')
    node = (115, 0)
    c.w(t1.es, node); c.dot(node); c.w(node, (180, 0)); c.port((180, 0), OUT, 'r')
    capacitor_to_ground(c, node)


# ================================================================ 1. bias circuits
def voltage_divider(c):
    """Voltage divider"""
    (t1, t2), (j,) = c.chain([('n', 1), ('n', 2)], 0, 0, side='right')
    top = (0, -20)
    c.port((0, -70), IN, 't'); c.w((0, -70), top, t1.cd); c.dot(top)
    g = (0, t2.es[1] + 20); c.gnd(g); c.w(t2.es, g)
    xt = 55
    c.w(t1.bg, (xt, t1.bg[1]), (xt, top[1]), top)
    c.w(t2.bg, (xt, t2.bg[1]), (xt, j[1]), j); c.dot((xt, j[1])); c.dot(j)
    c.w((xt, j[1]), (xt + 55, j[1])); c.port((xt + 55, j[1]), OUT, 'r')


def ptat_generator(c):
    """PTAT generator"""
    (t2, t1), (j,) = c.chain([('n', 2), ('n', 1)], 0, 0)
    t2.size('K·(W/L)'); t1.size('(W/L)')
    g = (0, t1.es[1] + 20); c.gnd(g); c.w(t1.es, g)
    top = (0, -20); c.w(t2.cd, top); c.dot(top); c.w(top, (0, -70)); c.port((0, -70), 'I_bias', 'r')
    xg = -70
    c.w(top, (xg, top[1]), (xg, t1.bg[1])); c.w(t2.bg, (xg, t2.bg[1])); c.w(t1.bg, (xg, t1.bg[1])); c.dot((xg, t2.bg[1]))
    c.dot(j); c.w(j, (60, j[1])); c.port((60, j[1]), 'V_ptat', 'r')


def bias_generator(c):
    """- | Bias generator"""
    (t1, t2), (j,) = c.chain([('p', 1), ('n', 2, {'side': 'right'})], 0, 0)
    c.supply((0, -20)); c.w((0, -20), t1.es)
    g = (0, t2.es[1] + 20); c.gnd(g); c.w(t2.es, g)
    xg = -60
    c.w(t1.bg, (xg, t1.bg[1]), (xg, t2.es[1] + 20)); c.gnd((xg, t2.es[1] + 20))
    c.w(j, (70, j[1]), (70, t2.bg[1])); c.dot(j); c.dot((70, t2.bg[1]))
    c.w((70, t2.bg[1]), t2.bg); c.w((70, t2.bg[1]), (110, t2.bg[1])); c.port((110, t2.bg[1]), 'V_bias', 'r')


def tie_cell(c):
    """Tie cell"""
    p = c.transistor('p', (0, 0), 'es', side='right', name=1)
    c.supply((0, -20)); c.w((0, -20), p.es)
    n = c.transistor('n', (30, 145), 'bg', name=2)
    c.w(n.bg, (0, n.bg[1]), p.cd)
    tap = (0, 100)
    c.w(p.bg, (p.bg[0], 100), tap); c.dot(tap)
    c.w(n.cd, (n.cd[0], p.cd[1]), (n.cd[0] + 40, p.cd[1])); c.port((n.cd[0] + 40, p.cd[1]), 'V_low', 'r')
    g = (n.es[0], n.es[1] + 20); c.gnd(g); c.w(n.es, g)
    c.center(0)                                             # aligned on the route from the supply


def current_mirror(c):
    """Current mirror"""
    t1 = c.transistor('n', (0, 0), 'cd', side='right', name=1, lbl='l')
    t2 = c.transistor('n', (140, 0), 'cd', name=2, lbl='r')
    t1.size('(W/L)_1'); t2.size('(W/L)_2')
    top = (0, -20)
    c.w(t1.cd, top); c.dot(top); c.w(top, (0, -70)); c.port((0, -70), 'I_in', 'r')
    c.w(t2.cd, (140, -70)); c.port((140, -70), 'I_out', 'r')
    gy = t1.bg[1]
    c.w(t1.bg, t2.bg); mx = (t1.bg[0] + t2.bg[0]) / 2
    c.dot((mx, gy)); c.w((mx, gy), (mx, top[1]), top)
    yb = t1.es[1] + 20
    c.w(t1.es, (0, yb), (140, yb), t2.es)
    c.dot((70, yb)); c.w((70, yb), (70, yb + 10)); c.gnd((70, yb + 10))


def current_divider(c):
    """- | Current divider"""
    (t2, t1), (j,) = c.chain([('n', 2), ('n', 1)], 0, 0)
    xin = -70
    c.bus([t2, t1], xin)
    yg = j[1]; c.dot((xin, yg)); c.w((xin, yg), (xin - 30, yg)); c.port((xin - 30, yg), 'V_bias', 'l')
    c.w(t2.cd, (0, -50)); c.port((0, -50), 'I_out1', 'r')
    c.w(t1.es, (0, t1.es[1] + 40)); c.port((0, t1.es[1] + 40), 'I_out2', 'b')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), 'I_in', 'r')


# ================================================================ 2. gain stages
def active_load(c):
    """Common emitter with active load | Common source with active load"""
    (t2, t1), (j,) = c.chain([('p', 2), ('n', 1)], 0, 0)
    c.supply((0, -20)); c.w((0, -20), t2.es)
    g = (0, t1.es[1] + 20); c.gnd(g); c.w(t1.es, g)
    c.w(t2.bg, (t2.bg[0] - 30, t2.bg[1])); c.port((t2.bg[0] - 30, t2.bg[1]), 'V_bias', 'l')
    c.w(t1.bg, (t1.bg[0] - 30, t1.bg[1])); c.port((t1.bg[0] - 30, t1.bg[1]), IN, 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r'); c.look((50, j[1]), 'l', 'd', 'R_out')


def inverter(c):
    """Inverter"""
    (p, n), (j,) = c.chain([('p', 2), ('n', 1)], 0, 0)
    c.supply((0, -20)); c.w((0, -20), p.top)
    g = (0, n.es[1] + 20); c.gnd(g); c.w(n.es, g)
    xin = -70
    c.bus([p, n], xin)
    c.dot((xin, j[1])); c.w((xin, j[1]), (xin - 30, j[1])); c.port((xin - 30, j[1]), IN, 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')
    if c.device is MOSFET:                                  # the BJT version has no gain to speak of
        c.look((50, j[1]), 'l', 'd', 'R_out')


def diode_load(c):
    """Common emitter with diode load | Common source with diode load"""
    (t2, t1), (j,) = c.chain([('n', 2), ('n', 1)], 0, 0)
    t2.size('(W/L)_2'); t1.size('(W/L)_1')
    top = (0, -20); c.supply((0, -55)); c.w((0, -55), top, t2.cd); c.dot(top)
    c.w(t2.bg, (t2.bg[0] - 20, t2.bg[1]), (t2.bg[0] - 20, top[1]), top)
    g = (0, t1.es[1] + 20); c.gnd(g); c.w(t1.es, g)
    c.w(t1.bg, (t1.bg[0] - 30, t1.bg[1])); c.port((t1.bg[0] - 30, t1.bg[1]), IN, 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')


def folded_diode_load(c):
    """Folded diode load"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    t2 = c.transistor('n', (100, 0), 'cd', name=2)
    for t in (t1, t2):
        g = (t.es[0], t.es[1] + 20); c.gnd(g); c.w(t.es, g)
    c.w(t1.bg, (t1.bg[0] - 30, t1.bg[1])); c.port((t1.bg[0] - 30, t1.bg[1]), IN, 'l')
    node = (0, -20)
    c.w(t1.cd, (0, -70)); c.dot(node); c.port((0, -70), 'I_bias', 'r')
    c.w(node, (150, -20)); c.port((150, -20), OUT, 'r')
    c.w(t2.cd, (100, -20)); c.dot((100, -20))                         # the second transistor is diode-connected
    c.w((50, -20), (50, t2.bg[1]), t2.bg); c.dot((50, -20))


def degeneration(c):
    """- | Degenerated common source"""
    (t1, t2), (j,) = c.chain([('n', 1), ('n', 2)], 0, 0)
    g = (0, t2.es[1] + 20); c.gnd(g); c.w(t2.es, g)
    c.w(t1.cd, (0, -50)); c.port((0, -50), 'I_out', 'r')
    xin = -70
    c.bus([t1, t2], xin)
    yi = j[1]; c.dot((xin, yi)); c.w((xin, yi), (xin - 30, yi)); c.port((xin - 30, yi), IN, 'l')


def tunable_degeneration(c):
    """- | Tunable degeneration"""
    (t1, t2), (j,) = c.chain([('n', 1), ('n', 2)], 0, 0)
    g = (0, t2.es[1] + 20); c.gnd(g); c.w(t2.es, g)
    c.w(t1.cd, (0, -50)); c.port((0, -50), 'I_out', 'r')
    c.w(t1.bg, (t1.bg[0] - 30, t1.bg[1])); c.port((t1.bg[0] - 30, t1.bg[1]), IN, 'l')
    c.w(t2.bg, (t2.bg[0] - 30, t2.bg[1])); c.port((t2.bg[0] - 30, t2.bg[1]), 'V_bias', 'l')
    c.look(j, 'd', 'r', 'R_on2')


# ================================================================ 3. buffers
def follower(c):
    """Emitter follower | Source follower"""
    (t1, t2), (j,) = c.chain([('n', 1), ('n', 2)], 0, 0)
    c.supply((0, -20)); c.w((0, -20), t1.cd)
    g = (0, t2.es[1] + 20); c.gnd(g); c.w(t2.es, g)
    c.w(t1.bg, (t1.bg[0] - 30, t1.bg[1])); c.port((t1.bg[0] - 30, t1.bg[1]), IN, 'l')
    c.w(t2.bg, (t2.bg[0] - 30, t2.bg[1])); c.port((t2.bg[0] - 30, t2.bg[1]), 'V_bias', 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r'); c.look((50, j[1]), 'l', 'd', 'R_out')


def push_pull(c):
    """Push-pull emitter follower | Push-pull source follower"""
    (t1, t2), (j,) = c.chain([('n', 1), ('p', 2)], 0, 0)
    c.supply((0, -20)); c.w((0, -20), t1.cd)
    g = (0, t2.cd[1] + 20); c.gnd(g); c.w(t2.cd, g)
    xin = -70
    c.bus([t1, t2], xin)
    yi = j[1]; c.dot((xin, yi)); c.w((xin, yi), (xin - 30, yi)); c.port((xin - 30, yi), IN, 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')


def buffer_stage(c):
    """Common-base stage | Common-gate stage"""
    (t1, t2), (j,) = c.chain([('n', 1), ('n', 2)], 0, 0)
    g = (0, t2.es[1] + 20); c.gnd(g); c.w(t2.es, g)
    c.w(t1.bg, (t1.bg[0] - 30, t1.bg[1])); c.port((t1.bg[0] - 30, t1.bg[1]), 'V_bias2', 'l')
    c.w(t2.bg, (t2.bg[0] - 30, t2.bg[1])); c.port((t2.bg[0] - 30, t2.bg[1]), 'V_bias1', 'l')
    xl = t1.bg[0] - 90
    c.dot(j); c.w(j, (xl, j[1])); c.port((xl, j[1]), 'I_in', 'l'); c.look((xl, j[1]), 'r', 'd', 'R_in')
    c.w(t1.cd, (0, -40)); c.port((0, -40), 'I_out', 'r')


def stacked_buffers(c):
    """Cascoded common base | Cascoded common gate"""
    (t2, t1), (j,) = c.chain([('n', 2), ('n', 1)], 0, 0)
    c.w(t1.bg, (t1.bg[0] - 30, t1.bg[1])); c.port((t1.bg[0] - 30, t1.bg[1]), 'V_bias1', 'l')
    c.w(t2.bg, (t2.bg[0] - 30, t2.bg[1])); c.port((t2.bg[0] - 30, t2.bg[1]), 'V_bias2', 'l')
    c.w(t2.cd, (0, -40)); c.port((0, -40), 'I_out', 'l'); c.look((0, -40), 'd', 'r', 'R_out')
    c.w(t1.es, (0, t1.es[1] + 40)); c.port((0, t1.es[1] + 40), 'I_in', 'l')


def two_outputs(c):
    """Common base + common emitter | Common gate + common source"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    t2 = c.transistor('n', (-80, 0), 'es', name=2)
    yt = -110
    c.w(t1.cd, (0, yt)); c.port((0, yt), 'I_out,n', 'r')
    c.w(t2.cd, (-80, yt)); c.port((-80, yt), 'I_out,p', 'r')
    g = (0, t1.es[1] + 20); c.gnd(g); c.w(t1.es, g)
    c.w(t2.bg, (t2.bg[0] - 25, t2.bg[1])); c.port((t2.bg[0] - 25, t2.bg[1]), 'V_bias', 'l')
    nd = (-80, t1.bg[1])
    c.w(t2.es, nd); c.dot(nd); c.w(nd, t1.bg)
    c.w(nd, (-135, nd[1])); c.port((-135, nd[1]), IN, 'l'); c.look((-135, nd[1]), 'r', 'd', 'R_in')
    c.w(nd, (-80, nd[1] + 70)); c.port((-80, nd[1] + 70), 'I_bias', 'b')


def two_paths(c):
    """Emitter follower + common base | Source follower + common gate"""
    (t1, t2), (j,) = c.chain([('n', 1), ('n', 2)], 0, 0)
    c.supply((0, -20)); c.w((0, -20), t1.cd)
    inp = (0, t2.es[1] + 20); c.w(t2.es, inp); c.dot(inp); c.w(inp, (0, inp[1] + 50)); c.port((0, inp[1] + 50), 'I_bias', 'b')
    c.w(t2.bg, (t2.bg[0] - 20, t2.bg[1])); c.port((t2.bg[0] - 20, t2.bg[1]), 'V_bias', 't', dx=8)
    xl = -100
    c.w(inp, (xl, inp[1]), (xl, t1.bg[1]), t1.bg)
    tap = (xl, (inp[1] + t1.bg[1]) / 2)
    c.dot(tap); c.w(tap, (xl - 30, tap[1])); c.port((xl - 30, tap[1]), IN, 't')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')


# ================================================================ 4. feedback pairs
def cascode(c):
    """Cascode"""
    (t2, t1), (j,) = c.chain([('n', 2), ('n', 1)], 0, 0)
    g = (0, t1.es[1] + 20); c.gnd(g); c.w(t1.es, g)
    c.w(t1.bg, (t1.bg[0] - 30, t1.bg[1])); c.port((t1.bg[0] - 30, t1.bg[1]), IN, 'l')
    c.w(t2.bg, (t2.bg[0] - 30, t2.bg[1])); c.port((t2.bg[0] - 30, t2.bg[1]), 'V_bias', 'l')
    c.w(t2.cd, (0, -40)); c.port((0, -40), 'I_out', 'l'); c.look((0, -40), 'd', 'r', 'R_out')


def regulated_cascode(c):
    """Regulated cascode"""
    c.center(0)                                             # aligned on the route from I_out to I_in
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    c.w(t1.cd, (0, -60)); c.port((0, -60), 'I_out', 'l'); c.look((0, -60), 'd', 'r', 'R_out')
    c.w(t1.es, (0, 150)); c.port((0, 150), 'I_in', 'b')
    t2 = c.transistor('n', (-110, 60), 'cd', side='right', name=2, lbl='l')
    gy = t1.bg[1]
    c.w(t2.cd, (-110, gy)); c.dot((-110, gy)); c.w((-110, gy), t1.bg)
    c.w((-110, gy), (-110, -10)); c.port((-110, -10), 'I_bias', 'r')
    c.w(t2.bg, (0, t2.bg[1])); c.dot((0, t2.bg[1]))
    c.w(t2.es, (-110, 150)); c.port((-110, 150), 'V_bias', 'b')


def flipped_follower(c):
    """Flipped voltage follower"""
    (t1, t2), (j,) = c.chain([('n', 1), ('n', 2)], 0, 0)
    g = (0, t2.es[1] + 20); c.gnd(g); c.w(t2.es, g)
    top = (0, -20); c.w(t1.cd, top); c.dot(top); c.w(top, (0, -70)); c.port((0, -70), 'I_bias', 'r')
    c.w(t1.bg, (t1.bg[0] - 40, t1.bg[1])); c.port((t1.bg[0] - 40, t1.bg[1]), IN, 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r'); c.look((50, j[1]), 'l', 'd', 'R_out')
    xb = -130
    c.w(top, (xb, top[1]), (xb, t2.bg[1]), t2.bg)


def active_inductor(c):
    """Active inductor"""
    (t2, t1), (j,) = c.chain([('n', 2), ('n', 1)], 0, 0)
    g = (0, t1.es[1] + 20); c.gnd(g); c.w(t1.es, g)
    top = (0, -20); c.w(t2.cd, top); c.dot(top); c.w(top, (0, -70)); c.port((0, -70), 'I_bias', 'r')
    c.w(t2.bg, (t2.bg[0] - 40, t2.bg[1])); c.port((t2.bg[0] - 40, t2.bg[1]), 'V_bias', 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), 'Z_in', 'r', color=RED); c.look((50, j[1]), 'l', 'd')
    xb = -130
    c.w(top, (xb, top[1]), (xb, t1.bg[1]), t1.bg)


def differential_pair(c):
    """Emitter-coupled pair | Source-coupled pair"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    t2 = c.transistor('n', (110, 0), 'cd', side='right', name=2)
    c.w(t1.bg, (t1.bg[0] - 20, t1.bg[1])); c.port((t1.bg[0] - 20, t1.bg[1]), 'V_in,p', 'l')
    c.w(t2.bg, (t2.bg[0] + 20, t2.bg[1])); c.port((t2.bg[0] + 20, t2.bg[1]), 'V_in,n', 'r')
    c.w(t1.cd, (0, -40)); c.port((0, -40), 'I_out,n', 'r')
    c.w(t2.cd, (110, -40)); c.port((110, -40), 'I_out,p', 'r')
    yb = t1.es[1] + 20
    c.w(t1.es, (0, yb), (110, yb), t2.es); c.dot((55, yb))
    c.w((55, yb), (55, yb + 40)); c.port((55, yb + 40), 'I_bias', 'b')


def cross_coupled_pair(c):
    """Cross-coupled pair"""
    t1 = c.transistor('n', (0, 0), 'cd', side='right', name=1)
    t2 = c.transistor('n', (150, 0), 'cd', name=2)
    a, b = (0, -20), (150, -20)
    c.w(t1.cd, (0, -70)); c.dot(a); c.port((0, -70), 'V_p', 'l'); c.look((0, -70), 'd', 'r', 'R')
    c.w(t2.cd, (150, -70)); c.dot(b); c.port((150, -70), 'V_n', 'r')
    xl, xr = c.cross(75, a[1], t1.bg[1])                    # the two wires cross as an X
    c.w(a, (xl, a[1]), (xr, t2.bg[1]), t2.bg)
    c.w(b, (xr, b[1]), (xl, t1.bg[1]), t1.bg)
    ys = 100
    c.w(t1.es, (0, ys), (150, ys), t2.es); c.dot((75, ys)); c.w((75, ys), (75, ys + 40)); c.port((75, ys + 40), 'I_bias', 'b')


# ================================================================ 5. circuit elements
def controlled_resistor(c):
    """- | Controlled resistor"""
    t1 = c.transistor('n', (0, 0), 'es', rot=1, name=1)
    tap = (-40, 0)
    c.w(t1.es, tap); c.dot(tap); c.w(tap, (-115, 0)); c.port((-115, 0), 'V_p', 'l'); c.look((-115, 0), 'r', 'd', 'R')
    c.w(t1.cd, (130, 0)); c.port((130, 0), 'V_n', 'r')
    t2 = c.transistor('n', (-40, -40), 'es', side='right', name=2)
    c.w(tap, t2.es)
    gn = (40, -80)
    c.w(t2.bg, gn); c.dot(gn); c.w(gn, t1.bg)
    tp = (-40, -130)
    c.w(t2.cd, tp); c.dot(tp); c.w(tp, (-40, -170)); c.port((-40, -170), 'I_bias', 'r')
    c.w(tp, (40, -130), gn)


def large_resistor(c):
    """- | Large resistor"""
    t1 = c.transistor('p', (0, 0), 'es', side='right', rot=3, name=1)
    t2 = c.transistor('p', (120, 0), 'cd', rot=1, name=2)
    c.w(t1.cd, (100, 0), t2.cd); c.dot((100, 0))
    c.w(t1.es, (-30, 0)); c.port((-30, 0), 'V_p', 'l'); c.look((-30, 0), 'r', 'd', 'R')
    c.w(t2.es, (230, 0)); c.port((230, 0), 'V_n', 'r')
    c.w((100, 0), (100, -60)); c.dot((100, -60))
    c.w(t1.bg, (t1.bg[0], -60), (100, -60))
    c.w(t2.bg, (t2.bg[0], -60), (100, -60))


def varactor(c):
    """Varactor"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    t2 = c.transistor('n', (125, 0), 'cd', side='right', name=2)
    mx = 62.5
    c.w(t1.cd, (0, -20), (125, -20), t2.cd); c.dot((mx, -20))
    c.w(t1.es, (0, 100), (125, 100), t2.es); c.dot((mx, 100))
    c.w((mx, -20), (mx, 100))
    c.w((mx, -20), (mx, -60)); c.port((mx, -60), 'V_bias', 'r')
    c.w(t1.bg, (t1.bg[0] - 30, t1.bg[1])); c.port((t1.bg[0] - 30, t1.bg[1]), 'V_p', 'l')
    c.w(t2.bg, (t2.bg[0] + 30, t2.bg[1])); c.port((t2.bg[0] + 30, t2.bg[1]), 'V_n', 'r')


def antiparallel_capacitors(c):
    """- | Anti-parallel capacitors"""
    t1 = c.transistor('n', (0, 0), 'cd', side='right')
    t1.name(1, 'r', dy=-14)
    t2 = c.transistor('n', (90, -130), 'cd')
    t2.name(2, 'l', dy=-14)
    xl, xr = -30, 120
    c.w(t1.cd, (xl, 0)); c.w(t1.es, (xl, 80)); c.w(t2.bg, (xl, t2.bg[1])); c.w((xl, t2.bg[1]), (xl, 80))
    c.dot((xl, -45)); c.w((xl, -45), (xl - 30, -45)); c.port((xl - 30, -45), 'V_p', 'l')
    c.w(t2.cd, (xr, t2.cd[1])); c.w(t2.es, (xr, t2.es[1])); c.w(t1.bg, (xr, t1.bg[1])); c.w((xr, t1.bg[1]), (xr, t2.cd[1]))
    c.dot((xr, -90)); c.w((xr, -90), (xr + 30, -90)); c.port((xr + 30, -90), 'V_n', 'r')


def antiparallel_diodes(c):
    """Anti-parallel diodes"""
    t1 = c.transistor('n', (0, 0), 'cd', rot=3, name=1)
    t2 = c.transistor('n', (0, -60), 'es', rot=1, name=2)
    xl, xr = -20, 100
    c.w(t1.cd, (xl, 0), (xl, -60), t2.es); c.dot((xl, -30)); c.w((xl, -30), (xl - 30, -30)); c.port((xl - 30, -30), 'V_p', 'l')
    c.w(t1.es, (xr, 0), (xr, -60), t2.cd); c.dot((xr, -30)); c.w((xr, -30), (xr + 30, -30)); c.port((xr + 30, -30), 'V_n', 'r')
    c.w(t1.bg, (t1.bg[0], t1.bg[1] + 20), (xl, t1.bg[1] + 20), (xl, 0)); c.dot((xl, 0))
    c.w(t2.bg, (t2.bg[0], t2.bg[1] - 20), (xr, t2.bg[1] - 20), (xr, -60)); c.dot((xr, -60))


def low_leakage_diode(c):
    """- | Ultra-low-power diode"""
    t1 = c.transistor('p', (0, 0), 'es', side='right', rot=3, name=1)
    t2 = c.transistor('n', (100, 0), 'cd', side='right', rot=3, name=2)
    c.w(t1.cd, t2.cd)
    a, b = (-5, 0), (t2.es[0] + 5, 0)                        # half-way between the terminal and the device
    c.w(t1.es, (-40, 0)); c.dot(a); c.port((-40, 0), 'V_p', 'l')
    c.w(t2.es, (220, 0)); c.dot(b); c.port((220, 0), 'V_n', 'r')
    y1, y2 = -60, -90
    xl, xr = c.cross(90, y1, y2)                            # the two wires cross as an X
    c.w(t1.bg, (t1.bg[0], y1), (xl, y1), (xr, y2), (b[0], y2), b)
    c.w(t2.bg, (t2.bg[0], y1), (xr, y1), (xl, y2), (a[0], y2), a)


def level_shift(c):
    """Level shift"""
    c.center(0)                                             # aligned on the route from the supply
    t2 = c.transistor('n', (0, 0), 'cd', name=2)
    c.supply((0, -20)); c.w((0, -20), t2.cd)
    node = (0, 100)
    c.w(t2.es, node); c.dot(node); c.w(node, (0, 150)); c.port((0, 150), 'I_bias', 'b')
    t1 = c.transistor('n', (-30, 100), 'bg', side='right', name=1)
    c.w(t1.bg, node)
    c.w(t1.cd, (t1.cd[0], 40)); c.dot((t1.cd[0], 40)); c.w((t1.cd[0], 40), (t1.cd[0] - 50, 40)); c.port((t1.cd[0] - 50, 40), 'V_p', 'l')
    c.w((t1.cd[0], 40), t2.bg)
    c.w(t1.es, (t1.es[0], 160), (t1.es[0] - 50, 160)); c.port((t1.es[0] - 50, 160), 'V_n', 'l')


def linear_i_to_v(c):
    """- | Linear I-to-V converter"""
    (t1, t2), (j,) = c.chain([('n', 1), ('n', 2)], 0, 0)
    c.supply((0, -20)); c.w((0, -20), t1.cd)
    g = (0, t2.es[1] + 20); c.gnd(g); c.w(t2.es, g)
    term, tx = -120, -60
    c.w(t1.bg, (term, t1.bg[1])); c.port((term, t1.bg[1]), 'V_bias', 'l')
    c.dot(j); c.w(j, (60, j[1])); c.port((60, j[1]), OUT, 'r')
    c.w(j, (term, j[1])); c.dot((tx, j[1])); c.port((term, j[1]), 'I_in', 'l'); c.look((term, j[1]), 'r', 'd', 'R_in')
    c.w((tx, j[1]), (tx, t2.bg[1]), t2.bg)


# ================================================================ 6. switched circuits
def transmission_gate(c):
    """- | Transmission gate"""
    t1 = c.transistor('n', (0, 0), 'es', side='right', rot=1, name=1)
    t2 = c.transistor('p', (0, -60), 'es', side='right', rot=3, name=2)
    xl, xr = -20, 100
    c.w(t1.es, (xl, 0), (xl, -60), t2.es); c.dot((xl, -30)); c.w((xl, -30), (xl - 30, -30)); c.port((xl - 30, -30), 'V_1', 'l')
    c.w(t1.cd, (xr, 0), (xr, -60), t2.cd); c.dot((xr, -30)); c.w((xr, -30), (xr + 30, -30)); c.port((xr + 30, -30), 'V_2', 'r')
    c.w(t1.bg, (t1.bg[0], t1.bg[1] + 30)); c.port((t1.bg[0], t1.bg[1] + 30), 'V_on', 'b')
    c.w(t2.bg, (t2.bg[0], t2.bg[1] - 30)); c.port((t2.bg[0], t2.bg[1] - 30), '!V_on', 't')


def multiplexer(c):
    """- | Multiplexer"""
    t1 = c.transistor('n', (0, 0), 'cd', side='right', rot=3, name=1)
    t2 = c.transistor('n', (0, -110), 'cd', side='right', rot=3, name=2)
    for t, nm in ((t1, 'V_1'), (t2, 'V_2')):
        c.w(t.cd, (t.cd[0] - 40, t.cd[1])); c.port((t.cd[0] - 40, t.cd[1]), nm, 'l')
    c.w(t1.bg, (t1.bg[0], t1.bg[1] - 20)); c.port((t1.bg[0], t1.bg[1] - 20), 'V_sel1', 't')
    c.w(t2.bg, (t2.bg[0], t2.bg[1] - 20)); c.port((t2.bg[0], t2.bg[1] - 20), 'V_sel2', 't')
    xc, yc = 115, -55
    c.w(t1.es, (xc, 0), (xc, yc)); c.dot((xc, yc)); c.w(t2.es, (xc, -110), (xc, yc))
    c.w((xc, yc), (xc + 40, yc)); c.port((xc + 40, yc), 'V_com', 'r')


def sample_and_hold(c):
    """- | Sample-and-hold"""
    pass_device_and_capacitor(c, 'V_sample')


def low_pass(c):
    """- | Low-pass filter"""
    pass_device_and_capacitor(c, 'V_bias', resistance='R_on1')


def peak_detector(c):
    """Peak detector"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    top = (0, -20)
    c.w(t1.cd, top); c.dot(top); c.w(top, (0, -70)); c.port((0, -70), IN, 'r')
    c.w(top, (-60, -20), (-60, t1.bg[1]), t1.bg)
    pk = (0, 107)
    c.w(t1.es, pk); c.dot(pk); c.w(pk, (60, pk[1])); c.port((60, pk[1]), 'V_peak', 'r')
    capacitor_to_ground(c, pk)


def voltage_doubler(c):
    """- | Voltage doubler"""
    t1 = c.transistor('p', (0, 0), 'es')
    t1.name(1, 'l', dy=-24)                                # clear of the name of the input
    bar = 30
    c.w(t1.es, (bar, 0), (bar, 80), t1.cd)
    c.dot((bar, 40)); c.w((bar, 40), (110, 40)); c.port((110, 40), OUT, 'r')
    c.w(t1.bg, (-55, 40)); c.port((-55, 40), IN, 'l')
    t2 = c.transistor('n', (70, 100), 'es', side='right', rot=2, name=2, lbl='r')
    c.w(t2.es, (70, 40)); c.dot((70, 40))
    join = (70, t2.cd[1] + 10)                               # half-way between the transistor and the ground
    c.w(t2.cd, (70, join[1] + 30)); c.dot(join); c.gnd((70, join[1] + 30))
    c.w(t2.bg, (t2.bg[0] - 20, t2.bg[1]), (t2.bg[0] - 20, join[1]), join)
    c.center(70)                                            # aligned on the route to ground


# ================================================================ 7. logic gates
def nand(c):
    """NAND"""
    (t2, t1), (j,) = c.chain([('n', 2), ('n', 1)], 0, 0)
    g = (0, t1.es[1] + 20); c.gnd(g); c.w(t1.es, g)
    for t, nm in ((t1, 'V_a'), (t2, 'V_b')):
        c.w(t.bg, (t.bg[0] - 30, t.bg[1])); c.port((t.bg[0] - 30, t.bg[1]), nm, 'l')
    top = (0, -20); c.w(t2.cd, top); c.dot(top); c.w(top, (50, top[1])); c.port((50, top[1]), OUT, 'r')
    c.w(top, (0, -70)); c.port((0, -70), 'I_bias', 'r')


def nor(c):
    """NOR"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    t2 = c.transistor('n', (130, 0), 'cd', side='right', name=2)
    for t in (t1, t2):
        g = (t.es[0], t.es[1] + 20); c.gnd(g); c.w(t.es, g)
    mx = 65
    c.w(t1.bg, (t1.bg[0] - 20, t1.bg[1])); c.port((t1.bg[0] - 20, t1.bg[1]), 'V_a', 'l')
    c.w(t2.bg, (t2.bg[0] + 20, t2.bg[1])); c.port((t2.bg[0] + 20, t2.bg[1]), 'V_b', 'r')
    yt = -20
    c.w(t1.cd, (0, yt), (130, yt)); c.w(t2.cd, (130, yt)); c.dot((mx, yt))
    top = (mx, -50); c.w((mx, yt), top); c.dot(top); c.w(top, (170, top[1])); c.port((170, top[1]), OUT, 'r')
    c.w(top, (mx, -80)); c.port((mx, -80), 'I_bias', 'r')


def xnor(c):
    """XNOR"""
    t1 = c.transistor('n', (0, 0), 'cd', name=1)
    t2 = c.transistor('n', (100, 0), 'cd', side='right', name=2)
    mx = 50
    ga = (t1.bg[0] - 10, t1.bg[1]); gb = (t2.bg[0] + 10, t2.bg[1])
    c.w(t1.bg, ga); c.dot(ga); c.w(ga, (ga[0] - 30, ga[1])); c.port((ga[0] - 30, ga[1]), 'V_a', 'l')
    c.w(t2.bg, gb); c.dot(gb); c.w(gb, (gb[0] + 30, gb[1])); c.port((gb[0] + 30, gb[1]), 'V_b', 'r')
    yt = -20
    c.w(t1.cd, (0, yt), (mx, yt)); c.w(t2.cd, (100, yt), (mx, yt)); c.dot((mx, yt))
    c.w((mx, yt), (mx, -50)); c.dot((mx, -50)); c.w((mx, -50), (100, -50)); c.port((100, -50), OUT, 'r')
    c.w((mx, -50), (mx, -80)); c.port((mx, -80), 'I_bias', 'r')
    y1, y2 = t1.es[1] + 20, t1.es[1] + 50
    xl, xr = c.cross(mx, y1, y2)                            # the two wires cross as an X
    c.w(t1.es, (0, y1), (xl, y1), (xr, y2), (gb[0], y2), gb)
    c.w(t2.es, (100, y1), (xr, y1), (xl, y2), (ga[0], y2), ga)


def mixer(c):
    """- | Mixer"""
    (t2, t1), (j,) = c.chain([('n', 2), ('n', 1)], 0, 0)
    g = (0, t1.es[1] + 20); c.gnd(g); c.w(t1.es, g)
    c.w(t2.cd, (0, -50)); c.port((0, -50), 'I_out', 'r')
    c.w(t2.bg, (t2.bg[0] - 30, t2.bg[1])); c.port((t2.bg[0] - 30, t2.bg[1]), 'V_lo', 'l')
    c.w(t1.bg, (t1.bg[0] - 30, t1.bg[1])); c.port((t1.bg[0] - 30, t1.bg[1]), 'V_rf', 'l')


# ================================================================ more transistors: a two-stage op amp
def op_amp(c, blocks):
    """The first `blocks` blocks of a two-stage op amp.  Every block is a circuit from further up:
    1 differential pair, 2 current source in its tail, 3 current mirror as its load,
    4 second stage with a current-source load, 5 current mirror that supplies both bias currents.
    The circuit is drawn narrow, so that all of it fits a cell at the size of every other drawing.
    The bias line runs above the lowest transistors and crosses the lead of the tail source;
    it goes down in line with the two vertical lines of the pair.  No wire runs below the ground symbols."""
    c.center(50)                                                                   # every row is aligned on the pair
    t1 = c.transistor('n', (0, 0), 'cd', name=1, lbl='r')                          # 1: the pair
    t2 = c.transistor('n', (100, 0), 'cd', side='right', name=2, lbl='l')
    c.port(t1.bg, 'V_1', 't'); c.port(t2.bg, 'V_2', 't')
    tail = (50, 100)
    c.w(t1.es, (0, 100), (100, 100), t2.es); c.dot(tail)
    if blocks < 2:
        c.w(tail, (50, 140)); c.port((50, 140), 'I_bias', 'b')
    else:                                                                          # 2: the tail current source
        t5 = c.transistor('n', (50, 150), 'cd', name=5)
        c.w(tail, t5.cd); c.w(t5.es, (50, 250)); c.gnd((50, 250))
        bias = (0, 190)                                                            # the gate node of the bias transistors, under the left line of the pair
        line = (tail[1] + t5.lead('cd')[1]) / 2                                    # bias line: half-way down the tail lead
        c.w(t5.bg, bias)
    if blocks < 3:
        c.w(t1.cd, (0, -40)); c.port((0, -40), 'I_1', 'r')
        c.w(t2.cd, (100, -40)); c.port((100, -40), 'I_2', 'r')
    else:                                                                          # 3: the current-mirror load
        t3 = c.transistor('p', (0, -100), 'es', side='right', name=3, lbl='l')
        t4 = c.transistor('p', (100, -100), 'es', name=4, lbl='r')
        c.w(t3.cd, t1.cd); c.w(t4.cd, t2.cd)
        c.w(t3.bg, t4.bg); c.dot((50, -60)); c.w((50, -60), (50, -10), (0, -10)); c.dot((0, -10))
        rail = 155 if blocks > 3 else 100
        c.w(t3.es, (0, -120), (rail, -120)); c.w(t4.es, (100, -120)); c.dot((50, -120))
        c.w((50, -120), (50, -160)); c.supply((50, -160))                         # as far above the rail as a ground hangs below
        first = (100, -10); c.dot(first)
    if blocks == 3:
        c.w(first, (160, -10)); c.port((160, -10), OUT, 'r'); c.look((160, -10), 'l', 'u', 'R_out')
    if blocks > 3:                                                                 # 4: the second stage
        t6 = c.transistor('p', (155, -50), 'es', name=6)                         # its input level with the first output
        t7 = c.transistor('n', (155, 150), 'cd', name=7)
        c.dot((100, -120)); c.w((155, -120), t6.es)
        c.w(first, t6.bg)
        c.w(t6.cd, t7.cd); c.w(t7.es, (155, 250)); c.gnd((155, 250))
        c.w(bias, (bias[0], line), (100, line), (100, t7.bg[1]), t7.bg); c.dot(bias)   # up, across, and down under the right line of the pair
        out = (155, 90); c.dot(out); c.w(out, (185, 90)); c.port((185, 90), OUT, 't')    # half-way between 6 and 7
    if blocks == 4:
        c.look((185, 90), 'l', 'd', 'R_out', nudge=-14)             # the arrow ends short of the second stage
    if 2 <= blocks < 5:
        c.w(bias, (-30, 190)); c.port((-30, 190), 'V_bias', 'l')
    if blocks == 5:                                                                # 5: the bias mirror
        t8 = c.transistor('n', (-50, 150), 'cd', side='right', name=8, lbl='l')
        c.w(t8.es, (-50, 250)); c.gnd((-50, 250))
        c.w(t8.bg, bias)
        c.w(t8.cd, (-50, 90)); c.port((-50, 90), 'I_ref', 'l')
        c.dot((-50, line)); c.w((-50, line), (bias[0], line)); c.dot((bias[0], line))    # the bias line starts at the diode


def op_amp_pair(c):
    """Emitter-coupled pair | Source-coupled pair"""
    op_amp(c, 1)


def op_amp_tail(c):
    """+ tail current source"""
    op_amp(c, 2)


def op_amp_mirror(c):
    """+ current-mirror load"""
    op_amp(c, 3)


def op_amp_second(c):
    """+ common-emitter stage | + common-source stage"""
    op_amp(c, 4)


def op_amp_bias(c):
    """+ bias mirror"""
    op_amp(c, 5)


# ================================================================ registry
def names(f):
    """(BJT name, MOSFET name) from the first docstring line; '' where the circuit has no such version"""
    parts = [p.strip() for p in (f.__doc__ or f.__name__).strip().splitlines()[0].split('|')]
    return tuple('' if p == '-' else p for p in (parts[0], parts[-1]))


TYPES = (n_type, p_type)
HELPERS = (capacitor_to_ground, pass_device_and_capacitor, op_amp)

# one transistor: (heading, first connection, second connection)
ROLES = [
    ('Amplifier and Follower', role_amplifier, role_follower),
    ('Current Buffer and Current Source', role_current_buffer, role_current_source),
    ('Diode and Switch', role_diode, role_switch),
    ('Resistor and Capacitor', role_resistor, role_capacitor),
]

# one transistor, after the summary inside the back cover of Razavi: (heading, first, second)
IMPEDANCES = [
    ('Looking into the Base or Gate', into_base, into_base_degenerated),
    ('Looking into the Collector or Drain', into_collector, into_collector_degenerated),
    ('Looking into the Emitter or Source', into_emitter, into_emitter_base_resistance),
]
GAINS = [
    ('Common-Emitter and Common-Source Stages', gain_common_emitter, gain_degenerated),
    ('Common-Base or Common-Gate Stage and Follower', gain_common_base, gain_follower),
]

# two transistors: (family, [(heading, first circuit, second circuit), ...])
FAMILIES = [
    ('Bias Circuits', [
        ('Voltage Divider and PTAT Generator', voltage_divider, ptat_generator),
        ('Bias Generator and Tie Cell', bias_generator, tie_cell),
        ('Current Mirror and Current Divider', current_mirror, current_divider)]),
    ('Gain Stages', [
        ('Active Load and Inverter', active_load, inverter),
        ('Stacked and Folded Diode Loads', diode_load, folded_diode_load),
        ('Fixed and Tunable Degeneration', degeneration, tunable_degeneration)]),
    ('Buffers', [
        ('Class-A and Push-Pull Followers', follower, push_pull),
        ('Single and Stacked Current Buffers', buffer_stage, stacked_buffers),
        ('Two Outputs and Two Paths', two_outputs, two_paths)]),
    ('Feedback Pairs', [
        ('Cascode and Regulated Cascode', cascode, regulated_cascode),
        ('Flipped Follower and Active Inductor', flipped_follower, active_inductor),
        ('Differential and Cross-Coupled Pairs', differential_pair, cross_coupled_pair)]),
    ('Circuit Elements', [
        ('Controlled and Large Resistors', controlled_resistor, large_resistor),
        ('Varactor and Anti-Parallel Capacitors', varactor, antiparallel_capacitors),
        ('Anti-Parallel and Ultra-Low-Power Diodes', antiparallel_diodes, low_leakage_diode),
        ('Level Shift and Linear I-to-V Converter', level_shift, linear_i_to_v)]),
    ('Switched Circuits', [
        ('Transmission Gate and Multiplexer', transmission_gate, multiplexer),
        ('Sample-and-Hold and Low-Pass Filter', sample_and_hold, low_pass),
        ('Peak Detector and Voltage Doubler', peak_detector, voltage_doubler)]),
    ('Logic Gates', [
        ('NAND and NOR', nand, nor),
        ('XNOR and Mixer', xnor, mixer)]),
]

# more transistors: (family, [(heading, circuits...), ...]); each circuit is the one before plus one block
BUILD = [
    ('Building a Two-Stage Op Amp', [
        ('Pair and Tail Source', op_amp_pair, op_amp_tail),
        ('Mirror Load and Second Stage', op_amp_mirror, op_amp_second),
        ('Bias Mirror', op_amp_bias)]),
]

# figure number of each circuit in Pretl & Eberlein (2021)
SOURCE_FIG = {
    'voltage_divider': 34, 'ptat_generator': 26, 'bias_generator': 24, 'tie_cell': 3, 'current_mirror': 7,
    'current_divider': 35, 'active_load': 10, 'inverter': 1, 'diode_load': 17, 'folded_diode_load': 18,
    'degeneration': 20, 'tunable_degeneration': 21, 'follower': 9, 'push_pull': 19, 'buffer_stage': 13,
    'stacked_buffers': 12, 'two_outputs': 22, 'two_paths': 23, 'cascode': 11, 'regulated_cascode': 48,
    'flipped_follower': 47, 'active_inductor': 33, 'differential_pair': 8, 'cross_coupled_pair': 28,
    'controlled_resistor': 41, 'large_resistor': 42, 'varactor': 30, 'antiparallel_capacitors': 32,
    'antiparallel_diodes': 31, 'low_leakage_diode': 43, 'level_shift': 44, 'linear_i_to_v': 45,
    'transmission_gate': 14, 'multiplexer': 15, 'sample_and_hold': 36, 'low_pass': 38, 'peak_detector': 39,
    'voltage_doubler': 40, 'nand': 5, 'nor': 6, 'xnor': 4, 'mixer': 37,
}

# flat list in page order: (number, function)
CIRCUITS = []
for _family, _pairs in FAMILIES:
    for _heading, _a, _b in _pairs:
        CIRCUITS += [(len(CIRCUITS) + 1, _a), (len(CIRCUITS) + 2, _b)]
NUMBER = {f.__name__: n for n, f in CIRCUITS}
