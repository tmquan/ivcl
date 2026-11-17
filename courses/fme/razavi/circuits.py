"""One transistor in eight connections, and forty-two circuits of two transistors.

Each function draws one circuit through a `Ckt` object `c`.  The same code yields the BJT
version (c.bjt True: 'n' -> npn, 'p' -> pnp, gate -> base, drain -> collector, source -> emitter)
and the MOSFET version (c.bjt False).  The first line of the docstring names the circuit,
as  "BJT name | MOSFET name"  where the two differ.  A name "-" means that the circuit has no
such version: that column of the figure stays empty.
Coordinates are px on a 10-px grid; the main column of leads is at x = 0.

The two-transistor circuits follow Pretl & Eberlein, "Fifty Nifty Variations of Two-Transistor
Circuits", IEEE Solid-State Circuits Magazine 13(3), 2021.
"""
from .ckt import mid

IN, OUT = 'V_in', 'V_out'


# ================================================================ types of transistors
def n_type(c):
    """npn | NMOS"""
    q = c.Q('n', (0, 0), 'd')
    c.w(q.d, (0, -30)); c.port((0, -30), c.term('d'), 't')
    c.w(q.s, (0, 110)); c.port((0, 110), c.term('s'), 'b')
    c.w(q.g, (q.g[0] - 30, 40)); c.port((q.g[0] - 30, 40), c.term('g'), 'l')


def p_type(c):
    """pnp | PMOS"""
    q = c.Q('p', (0, 0), 's')
    c.w(q.s, (0, -30)); c.port((0, -30), c.term('s'), 't')
    c.w(q.d, (0, 110)); c.port((0, 110), c.term('d'), 'b')
    c.w(q.g, (q.g[0] - 30, 40)); c.port((q.g[0] - 30, 40), c.term('g'), 'l')


# ================================================================ one transistor: eight connections
def role_amplifier(c):
    """Common emitter | Common source"""
    q = c.Q('n', (0, 0), 'd', name='M_1')
    c.vdd((0, -120)); c.w((0, -120), (0, -100))
    out = (0, -20)
    c.w(c.res((0, -100), 'R_' + c.term('d')), q.d); c.dot(out)
    c.w(out, (50, -20)); c.port((50, -20), OUT, 'r')
    c.w(q.s, (0, 100)); c.gnd((0, 100))
    c.w(q.g, (q.g[0] - 30, 40)); c.port((q.g[0] - 30, 40), IN, 'l')


def role_follower(c):
    """Emitter follower | Source follower"""
    q = c.Q('n', (0, 0), 'd', name='M_1')
    c.vdd((0, -20)); c.w((0, -20), q.d)
    out = (0, 100)
    c.w(q.s, (0, 120)); c.dot(out)
    c.w(out, (50, 100)); c.port((50, 100), OUT, 'r')
    c.w(c.res((0, 120), 'R_' + c.term('s')), (0, 200)); c.gnd((0, 200))
    c.w(q.g, (q.g[0] - 30, 40)); c.port((q.g[0] - 30, 40), IN, 'l')


def role_current_buffer(c):
    """Common base | Common gate"""
    q = c.Q('n', (0, 0), 'd', name='M_1')
    c.w(q.d, (0, -40)); c.port((0, -40), 'I_out', 'r')
    c.w(q.s, (0, 120)); c.port((0, 120), 'I_in', 'r')
    c.w(q.g, (q.g[0] - 30, 40)); c.port((q.g[0] - 30, 40), 'V_bias', 'l')


def role_current_source(c):
    """Current source"""
    q = c.Q('n', (0, 0), 'd', name='M_1')
    c.w(q.d, (0, -40)); c.port((0, -40), 'I_out', 'r')
    c.w(q.s, (0, 100)); c.gnd((0, 100))
    c.w(q.g, (q.g[0] - 30, 40)); c.port((q.g[0] - 30, 40), 'V_bias', 'l')


def role_diode(c):
    """Diode-connected transistor"""
    q = c.Q('n', (0, 0), 'd', name='M_1')
    top = (0, -20)
    c.w(q.d, (0, -60)); c.dot(top); c.port((0, -60), 'V_p', 'r')
    c.w(q.g, (q.g[0] - 20, 40), (q.g[0] - 20, -20), top)
    c.w(q.s, (0, 120)); c.port((0, 120), 'V_n', 'r')


def role_switch(c):
    """Switch"""
    q = c.Q('n', (0, 0), 'd', gate='right', rot=3, name='M_1')
    c.w(q.d, (q.d[0] - 40, 0)); c.port((q.d[0] - 40, 0), 'V_1', 'l')
    c.w(q.s, (q.s[0] + 40, 0)); c.port((q.s[0] + 40, 0), 'V_2', 'r')
    c.w(q.g, (q.g[0], q.g[1] - 20)); c.port((q.g[0], q.g[1] - 20), 'V_on', 't')


def role_resistor(c):
    """- | Resistor"""
    q = c.Q('n', (0, 0), 'd', gate='right', rot=3, name='M_1')
    c.w(q.d, (q.d[0] - 40, 0)); c.port((q.d[0] - 40, 0), 'V_p', 'l')
    c.w(q.s, (q.s[0] + 40, 0)); c.port((q.s[0] + 40, 0), 'V_n', 'r')
    c.w(q.g, (q.g[0], q.g[1] - 20)); c.port((q.g[0], q.g[1] - 20), 'V_bias', 't')


def role_capacitor(c):
    """Capacitor"""
    q = c.Q('n', (0, 0), 'g', gate='right', rot=3, name='M_1')
    c.w(q.g, (0, -40)); c.port((0, -40), 'V_p', 'r')
    yb = q.d[1] + 20
    c.w(q.d, (q.d[0], yb), (q.s[0], yb), q.s); c.dot((0, yb))
    c.w((0, yb), (0, yb + 40)); c.port((0, yb + 40), 'V_n', 'r')


# ================================================================ helpers shared by several circuits
def size_label(c, d, txt):
    """W/L label beside a MOSFET (not drawn in the BJT version)"""
    if not c.bjt:
        i = d.inst
        if d.mir:
            c.sc.text(txt, i.x - 8, i.y + i.h / 2 - 20, 'end')
        else:
            c.sc.text(txt, i.x + i.w + 8, i.y + i.h / 2 - 20, 'start')


def capacitor_to_ground(c, node):
    """transistor M_2 connected as a capacitor from `node` to ground"""
    m2 = c.Q('n', (node[0], node[1] + 40), 'g', gate='right', rot=3, name='M_2')
    c.w(node, m2.g)
    yb = m2.d[1] + 20
    c.w(m2.d, (m2.d[0], yb), (m2.s[0], yb), m2.s)
    c.dot((node[0], yb)); c.w((node[0], yb), (node[0], yb + 10)); c.gnd((node[0], yb + 10))


def pass_device_and_capacitor(c, gate_name):
    """pass transistor M_1 from the input to a node that M_2 loads as a capacitor"""
    m1 = c.Q('n', (0, 0), 'd', gate='right', rot=3, name='M_1')
    c.w(m1.d, (-30, 0)); c.port((-30, 0), IN, 'l')
    c.w(m1.g, (m1.g[0], m1.g[1] - 20)); c.port((m1.g[0], m1.g[1] - 20), gate_name, 't')
    node = (130, 0)
    c.w(m1.s, node); c.dot(node); c.w(node, (180, 0)); c.port((180, 0), OUT, 'r')
    capacitor_to_ground(c, node)


# ================================================================ 1. bias circuits
def voltage_divider(c):
    """Voltage divider"""
    (m1, m2), (j,) = c.chain([('n', 'M_1'), ('n', 'M_2')], 0, 0, gate='right')
    top = (0, -20)
    c.port((0, -60), IN, 't'); c.w((0, -60), top, m1.d); c.dot(top)
    g = (0, m2.s[1] + 20); c.gnd(g); c.w(m2.s, g)
    xt = 70
    c.w(m1.g, (xt, m1.g[1]), (xt, top[1]), top)
    c.w(m2.g, (xt, m2.g[1]), (xt, j[1]), j); c.dot((xt, j[1])); c.dot(j)
    c.w((xt, j[1]), (xt + 40, j[1])); c.port((xt + 40, j[1]), OUT, 'r')


def ptat_generator(c):
    """PTAT generator"""
    (m2, m1), (j,) = c.chain([('n', 'M_2'), ('n', 'M_1')], 0, 0)
    size_label(c, m2, 'K·(W/L)'); size_label(c, m1, '(W/L)')
    g = (0, m1.s[1] + 20); c.gnd(g); c.w(m1.s, g)
    t = (0, -20); c.w(m2.d, t); c.dot(t); c.w(t, (0, -60)); c.port((0, -60), 'I_bias', 'r')
    xg = -70
    c.w(t, (xg, t[1]), (xg, m1.g[1])); c.w(m2.g, (xg, m2.g[1])); c.w(m1.g, (xg, m1.g[1])); c.dot((xg, m2.g[1]))
    c.dot(j); c.w(j, (60, j[1])); c.port((60, j[1]), 'V_ptat', 'r')


def bias_generator(c):
    """- | Bias generator"""
    (m1, m2), (j,) = c.chain([('p', 'M_1'), ('n', 'M_2', {'gate': 'right'})], 0, 0)
    c.vdd((0, -20)); c.w((0, -20), m1.s)
    g = (0, m2.s[1] + 20); c.gnd(g); c.w(m2.s, g)
    xg = -60
    c.w(m1.g, (xg, m1.g[1]), (xg, m2.s[1] + 20)); c.gnd((xg, m2.s[1] + 20))
    c.w(j, (70, j[1]), (70, m2.g[1])); c.dot(j); c.dot((70, m2.g[1]))
    c.w((70, m2.g[1]), m2.g); c.w((70, m2.g[1]), (110, m2.g[1])); c.port((110, m2.g[1]), 'V_bias', 'r')


def tie_cell(c):
    """Tie cell"""
    p = c.Q('p', (0, 0), 's', gate='right', name='M_1')
    c.vdd((0, -20)); c.w((0, -20), p.s)
    n = c.Q('n', (30, 140), 'g', gate='left', name='M_2')
    c.w(n.g, (0, n.g[1]), p.d)
    tap = (0, 100)
    c.w(p.g, (p.g[0], 100), tap); c.dot(tap)
    c.w(n.d, (n.d[0], p.d[1]), (n.d[0] + 40, p.d[1])); c.port((n.d[0] + 40, p.d[1]), 'V_low', 'r')
    g = (n.s[0], n.s[1] + 20); c.gnd(g); c.w(n.s, g)


def current_mirror(c):
    """Current mirror"""
    m1 = c.Q('n', (0, 0), 'd', gate='right', name='M_1', lbl='l')
    m2 = c.Q('n', (140, 0), 'd', gate='left', name='M_2', lbl='r')
    t = (0, -20)
    c.w(m1.d, t); c.dot(t); c.w(t, (0, -60)); c.port((0, -60), 'I_in', 'r')
    c.w(m2.d, (140, -60)); c.port((140, -60), 'I_out', 'r')
    gy = m1.g[1]
    c.w(m1.g, m2.g); mx = (m1.g[0] + m2.g[0]) / 2
    c.dot((mx, gy)); c.w((mx, gy), (mx, t[1]), t)
    yb = m1.s[1] + 20
    c.w(m1.s, (0, yb), (140, yb), m2.s)
    c.dot((70, yb)); c.w((70, yb), (70, yb + 10)); c.gnd((70, yb + 10))


def current_divider(c):
    """- | Current divider"""
    (m2, m1), (j,) = c.chain([('n', 'M_2'), ('n', 'M_1')], 0, 0)
    xin = -70
    c.gate_bus([m2, m1], xin)
    yg = j[1]; c.dot((xin, yg)); c.w((xin, yg), (xin - 30, yg)); c.port((xin - 30, yg), 'V_bias', 'l')
    c.w(m2.d, (0, -50)); c.port((0, -50), 'I_out1', 'r')
    c.w(m1.s, (0, m1.s[1] + 40)); c.port((0, m1.s[1] + 40), 'I_out2', 'b')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), 'I_in', 'r')


# ================================================================ 2. gain stages
def active_load(c):
    """Common emitter with active load | Common source with active load"""
    (m2, m1), (j,) = c.chain([('p', 'M_2'), ('n', 'M_1')], 0, 0)
    c.vdd((0, -20)); c.w((0, -20), m2.s)
    g = (0, m1.s[1] + 20); c.gnd(g); c.w(m1.s, g)
    c.w(m2.g, (m2.g[0] - 30, m2.g[1])); c.port((m2.g[0] - 30, m2.g[1]), 'V_bias', 'l')
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), IN, 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')


def inverter(c):
    """Inverter"""
    (p, n), (j,) = c.chain([('p', 'M_2'), ('n', 'M_1')], 0, 0)
    c.vdd((0, -20)); c.w((0, -20), c.top(p))
    g = (0, n.s[1] + 20); c.gnd(g); c.w(n.s, g)
    xin = -70
    c.gate_bus([p, n], xin)
    c.dot((xin, j[1])); c.w((xin, j[1]), (xin - 30, j[1])); c.port((xin - 30, j[1]), IN, 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')


def diode_load(c):
    """Common emitter with diode load | Common source with diode load"""
    (m2, m1), (j,) = c.chain([('n', 'M_2'), ('n', 'M_1')], 0, 0)
    size_label(c, m2, '(W/L)_2'); size_label(c, m1, '(W/L)_1')
    top = (0, -20); c.vdd((0, -40)); c.w((0, -40), top, m2.d); c.dot(top)
    c.w(m2.g, (m2.g[0] - 20, m2.g[1]), (m2.g[0] - 20, top[1]), top)
    g = (0, m1.s[1] + 20); c.gnd(g); c.w(m1.s, g)
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), IN, 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')


def folded_diode_load(c):
    """Folded diode load"""
    m1 = c.Q('n', (0, 0), 'd', gate='left', name='M_1')
    m2 = c.Q('n', (100, 0), 'd', gate='left', name='M_2')
    for d in (m1, m2):
        g = (d.s[0], d.s[1] + 20); c.gnd(g); c.w(d.s, g)
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), IN, 'l')
    node = (0, -20)
    c.w(m1.d, (0, -60)); c.dot(node); c.port((0, -60), 'I_bias', 'r')
    c.w(node, (150, -20)); c.port((150, -20), OUT, 'r')
    c.w(m2.d, (100, -20)); c.dot((100, -20))                         # M_2 is diode-connected
    c.w((40, -20), (40, m2.g[1]), m2.g); c.dot((40, -20))


def degeneration(c):
    """- | Degenerated common source"""
    (m1, m2), (j,) = c.chain([('n', 'M_1'), ('n', 'M_2')], 0, 0)
    g = (0, m2.s[1] + 20); c.gnd(g); c.w(m2.s, g)
    c.w(m1.d, (0, -50)); c.port((0, -50), 'I_out', 'r')
    xin = -70
    c.gate_bus([m1, m2], xin)
    yi = m1.g[1]; c.dot((xin, yi)); c.w((xin, yi), (xin - 30, yi)); c.port((xin - 30, yi), IN, 'l')


def tunable_degeneration(c):
    """- | Tunable degeneration"""
    (m1, m2), (j,) = c.chain([('n', 'M_1'), ('n', 'M_2')], 0, 0)
    g = (0, m2.s[1] + 20); c.gnd(g); c.w(m2.s, g)
    c.w(m1.d, (0, -50)); c.port((0, -50), 'I_out', 'r')
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), IN, 'l')
    c.w(m2.g, (m2.g[0] - 30, m2.g[1])); c.port((m2.g[0] - 30, m2.g[1]), 'V_bias', 'l')


# ================================================================ 3. buffers
def follower(c):
    """Emitter follower | Source follower"""
    (m1, m2), (j,) = c.chain([('n', 'M_1'), ('n', 'M_2')], 0, 0)
    c.vdd((0, -20)); c.w((0, -20), m1.d)
    g = (0, m2.s[1] + 20); c.gnd(g); c.w(m2.s, g)
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), IN, 'l')
    c.w(m2.g, (m2.g[0] - 30, m2.g[1])); c.port((m2.g[0] - 30, m2.g[1]), 'V_bias', 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')


def push_pull(c):
    """Push-pull emitter follower | Push-pull source follower"""
    (m1, m2), (j,) = c.chain([('n', 'M_1'), ('p', 'M_2')], 0, 0)
    c.vdd((0, -20)); c.w((0, -20), m1.d)
    g = (0, m2.d[1] + 20); c.gnd(g); c.w(m2.d, g)
    xin = -70
    c.gate_bus([m1, m2], xin)
    yi = j[1]; c.dot((xin, yi)); c.w((xin, yi), (xin - 30, yi)); c.port((xin - 30, yi), IN, 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')


def buffer_stage(c):
    """Common-base stage | Common-gate stage"""
    (m1, m2), (j,) = c.chain([('n', 'M_1'), ('n', 'M_2')], 0, 0)
    g = (0, m2.s[1] + 20); c.gnd(g); c.w(m2.s, g)
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), 'V_bias2', 'l')
    c.w(m2.g, (m2.g[0] - 30, m2.g[1])); c.port((m2.g[0] - 30, m2.g[1]), 'V_bias1', 'l')
    xl = m1.g[0] - 30
    c.dot(j); c.w(j, (xl, j[1])); c.port((xl, j[1]), 'I_in', 'l')
    c.w(m1.d, (0, -40)); c.port((0, -40), 'I_out', 'r')


def stacked_buffers(c):
    """Cascoded common base | Cascoded common gate"""
    (m2, m1), (j,) = c.chain([('n', 'M_2'), ('n', 'M_1')], 0, 0)
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), 'V_bias1', 'l')
    c.w(m2.g, (m2.g[0] - 30, m2.g[1])); c.port((m2.g[0] - 30, m2.g[1]), 'V_bias2', 'l')
    c.w(m2.d, (0, -40)); c.port((0, -40), 'I_out', 'r')
    c.w(m1.s, (0, m1.s[1] + 40)); c.port((0, m1.s[1] + 40), 'I_in', 'r')


def two_outputs(c):
    """Common base + common emitter | Common gate + common source"""
    m1 = c.Q('n', (0, 0), 'd', gate='left', name='M_1')
    m2 = c.Q('n', (-100, 0), 's', gate='left', name='M_2')
    yt = -110
    c.w(m1.d, (0, yt)); c.port((0, yt), 'I_out,n', 'r')
    c.w(m2.d, (-100, yt)); c.port((-100, yt), 'I_out,p', 'r')
    g = (0, m1.s[1] + 20); c.gnd(g); c.w(m1.s, g)
    c.w(m2.g, (m2.g[0] - 30, m2.g[1])); c.port((m2.g[0] - 30, m2.g[1]), 'V_bias', 'l')
    nd = (-100, m1.g[1])
    c.w(m2.s, nd); c.dot(nd); c.w(nd, m1.g)
    c.w(nd, (-150, nd[1])); c.port((-150, nd[1]), IN, 'l')
    c.w(nd, (-100, nd[1] + 40)); c.port((-100, nd[1] + 40), 'I_bias', 'b')


def two_paths(c):
    """Emitter follower + common base | Source follower + common gate"""
    (m1, m2), (j,) = c.chain([('n', 'M_1'), ('n', 'M_2')], 0, 0)
    c.vdd((0, -20)); c.w((0, -20), m1.d)
    inp = (0, m2.s[1] + 20); c.w(m2.s, inp); c.dot(inp); c.w(inp, (0, inp[1] + 40)); c.port((0, inp[1] + 40), 'I_bias', 'b')
    c.w(m2.g, (m2.g[0] - 20, m2.g[1])); c.port((m2.g[0] - 20, m2.g[1]), 'V_bias', 't', dx=8)
    xl = -130
    c.w(inp, (xl, inp[1])); c.dot((xl, inp[1])); c.w((xl, inp[1]), (xl - 30, inp[1])); c.port((xl - 30, inp[1]), IN, 'l')
    c.w((xl, inp[1]), (xl, m1.g[1]), m1.g)
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')


# ================================================================ 4. feedback pairs
def cascode(c):
    """Cascode"""
    (m2, m1), (j,) = c.chain([('n', 'M_2'), ('n', 'M_1')], 0, 0)
    g = (0, m1.s[1] + 20); c.gnd(g); c.w(m1.s, g)
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), IN, 'l')
    c.w(m2.g, (m2.g[0] - 30, m2.g[1])); c.port((m2.g[0] - 30, m2.g[1]), 'V_bias', 'l')
    c.w(m2.d, (0, -40)); c.port((0, -40), 'I_out', 'r')


def regulated_cascode(c):
    """Regulated cascode"""
    m1 = c.Q('n', (0, 0), 'd', gate='left', name='M_1')
    c.w(m1.d, (0, -60)); c.port((0, -60), 'I_out', 'r')
    c.w(m1.s, (0, 160)); c.port((0, 160), 'I_in', 'b')
    m2 = c.Q('n', (-110, 60), 'd', gate='right', name='M_2', lbl='l')
    gy = m1.g[1]
    c.w(m2.d, (-110, gy)); c.dot((-110, gy)); c.w((-110, gy), m1.g)
    c.w((-110, gy), (-110, -60)); c.port((-110, -60), 'I_bias', 'r')
    c.w(m2.g, (0, m2.g[1])); c.dot((0, m2.g[1]))
    c.w(m2.s, (-110, 160)); c.port((-110, 160), 'V_bias', 'b')


def flipped_follower(c):
    """Flipped voltage follower"""
    (m1, m2), (j,) = c.chain([('n', 'M_1'), ('n', 'M_2')], 0, 0)
    g = (0, m2.s[1] + 20); c.gnd(g); c.w(m2.s, g)
    top = (0, -20); c.w(m1.d, top); c.dot(top); c.w(top, (0, -60)); c.port((0, -60), 'I_bias', 'r')
    c.w(m1.g, (m1.g[0] - 40, m1.g[1])); c.port((m1.g[0] - 40, m1.g[1]), IN, 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')
    xb = -130
    c.w(top, (xb, top[1]), (xb, m2.g[1]), m2.g)


def active_inductor(c):
    """Active inductor"""
    (m1, m2), (j,) = c.chain([('n', 'M_1'), ('n', 'M_2')], 0, 0)
    g = (0, m2.s[1] + 20); c.gnd(g); c.w(m2.s, g)
    top = (0, -20); c.w(m1.d, top); c.dot(top); c.w(top, (0, -60)); c.port((0, -60), 'I_bias', 'r')
    c.w(m1.g, (m1.g[0] - 40, m1.g[1])); c.port((m1.g[0] - 40, m1.g[1]), 'V_bias', 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), 'Z_in', 'r')
    xb = -130
    c.w(top, (xb, top[1]), (xb, m2.g[1]), m2.g)


def differential_pair(c):
    """Emitter-coupled pair | Source-coupled pair"""
    m1 = c.Q('n', (0, 0), 'd', gate='left', name='M_1')
    m2 = c.Q('n', (140, 0), 'd', gate='right', name='M_2')
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), 'V_in,p', 'l')
    c.w(m2.g, (m2.g[0] + 30, m2.g[1])); c.port((m2.g[0] + 30, m2.g[1]), 'V_in,n', 'r')
    c.w(m1.d, (0, -40)); c.port((0, -40), 'I_out,n', 'r')
    c.w(m2.d, (140, -40)); c.port((140, -40), 'I_out,p', 'l')
    yb = m1.s[1] + 20
    c.w(m1.s, (0, yb), (140, yb), m2.s); c.dot((70, yb))
    c.w((70, yb), (70, yb + 40)); c.port((70, yb + 40), 'I_bias', 'b')


def cross_coupled_pair(c):
    """Cross-coupled pair"""
    m1 = c.Q('n', (0, 0), 'd', gate='right', name='M_1')
    m2 = c.Q('n', (150, 0), 'd', gate='left', name='M_2')
    ta = (0, -20); tb = (150, -40)
    c.w(m1.d, ta); c.dot(ta); c.w(ta, (0, -60)); c.port((0, -60), 'V_p', 'r')
    c.w(m2.d, (150, -60)); c.dot(tb); c.port((150, -60), 'V_n', 'r')
    c.w(ta, (95, -20), (95, m2.g[1]), m2.g)
    c.w(tb, (55, -40), (55, m1.g[1]), m1.g)
    ys = 100
    c.w(m1.s, (0, ys), (150, ys), m2.s); c.dot((75, ys)); c.w((75, ys), (75, ys + 40)); c.port((75, ys + 40), 'I_bias', 'b')


# ================================================================ 5. circuit elements
def controlled_resistor(c):
    """- | Controlled resistor"""
    m1 = c.Q('n', (0, 0), 's', gate='left', rot=1, name='M_1')
    tap = (-40, 0)
    c.w(m1.s, tap); c.dot(tap); c.w(tap, (-90, 0)); c.port((-90, 0), 'V_p', 'l')
    c.w(m1.d, (130, 0)); c.port((130, 0), 'V_n', 'r')
    m2 = c.Q('n', (-40, -40), 's', gate='right', name='M_2')
    c.w(tap, m2.s)
    gn = (40, -80)
    c.w(m2.g, gn); c.dot(gn); c.w(gn, m1.g)
    tp = (-40, -140)
    c.w(m2.d, tp); c.dot(tp); c.w(tp, (-40, -180)); c.port((-40, -180), 'I_bias', 'r')
    c.w(tp, (40, -140), gn)


def large_resistor(c):
    """- | Large resistor"""
    m1 = c.Q('p', (0, 0), 's', gate='right', rot=3, name='M_1')
    m2 = c.Q('p', (120, 0), 'd', gate='left', rot=1, name='M_2')
    c.w(m1.d, (100, 0), m2.d); c.dot((100, 0))
    c.w(m1.s, (-30, 0)); c.port((-30, 0), 'V_p', 'l')
    c.w(m2.s, (230, 0)); c.port((230, 0), 'V_n', 'r')
    c.w((100, 0), (100, -60)); c.dot((100, -60))
    c.w(m1.g, (m1.g[0], -60), (100, -60))
    c.w(m2.g, (m2.g[0], -60), (100, -60))


def varactor(c):
    """Varactor"""
    m1 = c.Q('n', (0, 0), 'd', gate='left', name='M_1')
    m2 = c.Q('n', (125, 0), 'd', gate='right', name='M_2')
    mx = 62.5
    c.w(m1.d, (0, -20), (125, -20), m2.d); c.dot((mx, -20))
    c.w(m1.s, (0, 100), (125, 100), m2.s); c.dot((mx, 100))
    c.w((mx, -20), (mx, 100))
    c.w((mx, -20), (mx, -60)); c.port((mx, -60), 'V_bias', 'r')
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), 'V_p', 'l')
    c.w(m2.g, (m2.g[0] + 30, m2.g[1])); c.port((m2.g[0] + 30, m2.g[1]), 'V_n', 'r')


def antiparallel_capacitors(c):
    """- | Anti-parallel capacitors"""
    m1 = c.Q('n', (0, 0), 'd', gate='right')
    c.name(m1, 'M_1', 'r', dy=-14)
    m2 = c.Q('n', (90, -130), 'd', gate='left')
    c.name(m2, 'M_2', 'l', dy=-14)
    xl, xr = -30, 120
    c.w(m1.d, (xl, 0)); c.w(m1.s, (xl, 80)); c.w(m2.g, (xl, m2.g[1])); c.w((xl, m2.g[1]), (xl, 80))
    c.dot((xl, -20)); c.w((xl, -20), (xl - 30, -20)); c.port((xl - 30, -20), 'V_p', 'l')
    c.w(m2.d, (xr, m2.d[1])); c.w(m2.s, (xr, m2.s[1])); c.w(m1.g, (xr, m1.g[1])); c.w((xr, m1.g[1]), (xr, m2.d[1]))
    c.dot((xr, -90)); c.w((xr, -90), (xr + 30, -90)); c.port((xr + 30, -90), 'V_n', 'r')


def antiparallel_diodes(c):
    """Anti-parallel diodes"""
    m1 = c.Q('n', (0, 0), 'd', rot=3, name='M_1')
    m2 = c.Q('n', (0, -60), 's', rot=1, name='M_2')
    xl, xr = -20, 100
    c.w(m1.d, (xl, 0), (xl, -60), m2.s); c.dot((xl, -30)); c.w((xl, -30), (xl - 30, -30)); c.port((xl - 30, -30), 'V_p', 'l')
    c.w(m1.s, (xr, 0), (xr, -60), m2.d); c.dot((xr, -30)); c.w((xr, -30), (xr + 30, -30)); c.port((xr + 30, -30), 'V_n', 'r')
    c.w(m1.g, (m1.g[0], m1.g[1] + 20), (xl, m1.g[1] + 20), (xl, 0)); c.dot((xl, 0))
    c.w(m2.g, (m2.g[0], m2.g[1] - 20), (xr, m2.g[1] - 20), (xr, -60)); c.dot((xr, -60))


def low_leakage_diode(c):
    """- | Ultra-low-power diode"""
    m1 = c.Q('p', (0, 0), 's', gate='right', rot=3, name='M_1')
    m2 = c.Q('n', (100, 0), 'd', gate='right', rot=3, name='M_2')
    c.w(m1.d, m2.d)
    c.w(m1.s, (-20, 0)); c.dot((-20, 0)); c.w((-20, 0), (-50, 0)); c.port((-50, 0), 'V_p', 'l')
    c.w(m2.s, (200, 0)); c.dot((200, 0)); c.w((200, 0), (230, 0)); c.port((230, 0), 'V_n', 'r')
    c.w(m1.g, (m1.g[0], -60), (200, -60), (200, 0))
    c.w(m2.g, (m2.g[0], -80), (-20, -80), (-20, 0))


def level_shift(c):
    """Level shift"""
    m2 = c.Q('n', (0, 0), 'd', gate='left', name='M_2')
    c.vdd((0, -20)); c.w((0, -20), m2.d)
    node = (0, 100)
    c.w(m2.s, node); c.dot(node); c.w(node, (0, 160)); c.port((0, 160), 'I_bias', 'b')
    m1 = c.Q('n', (-50, 100), 'g', gate='right', name='M_1')
    c.w(m1.g, node)
    c.w(m1.d, (m1.d[0], 40)); c.dot((m1.d[0], 40)); c.w((m1.d[0], 40), (m1.d[0] - 40, 40)); c.port((m1.d[0] - 40, 40), 'V_p', 'l')
    c.w((m1.d[0], 40), m2.g)
    c.w(m1.s, (m1.s[0], 160), (m1.s[0] - 40, 160)); c.port((m1.s[0] - 40, 160), 'V_n', 'l')


def linear_i_to_v(c):
    """- | Linear I-to-V converter"""
    (m1, m2), (j,) = c.chain([('n', 'M_1'), ('n', 'M_2')], 0, 0)
    c.vdd((0, -20)); c.w((0, -20), m1.d)
    g = (0, m2.s[1] + 20); c.gnd(g); c.w(m2.s, g)
    term, tx = -120, -60
    c.w(m1.g, (term, m1.g[1])); c.port((term, m1.g[1]), 'V_bias', 'l')
    c.dot(j); c.w(j, (50, j[1])); c.port((50, j[1]), OUT, 'r')
    c.w(j, (term, j[1])); c.dot((tx, j[1])); c.port((term, j[1]), 'I_in', 'l')
    c.w((tx, j[1]), (tx, m2.g[1]), m2.g)


# ================================================================ 6. switched circuits
def transmission_gate(c):
    """- | Transmission gate"""
    m1 = c.Q('n', (0, 0), 'd', rot=3, name='M_1')
    m2 = c.Q('p', (0, -60), 'd', rot=1, name='M_2')
    xl, xr = -20, 100
    c.w(m1.d, (xl, 0), (xl, -60), m2.d); c.dot((xl, -30)); c.w((xl, -30), (xl - 30, -30)); c.port((xl - 30, -30), 'V_1', 'l')
    c.w(m1.s, (xr, 0), (xr, -60), m2.s); c.dot((xr, -30)); c.w((xr, -30), (xr + 30, -30)); c.port((xr + 30, -30), 'V_2', 'r')
    c.w(m1.g, (m1.g[0], m1.g[1] + 30)); c.port((m1.g[0], m1.g[1] + 30), 'V_on', 'b')
    c.w(m2.g, (m2.g[0], m2.g[1] - 30)); c.port((m2.g[0], m2.g[1] - 30), '!V_on', 't')


def multiplexer(c):
    """- | Multiplexer"""
    m1 = c.Q('n', (0, 0), 'd', gate='right', rot=3, name='M_1')
    m2 = c.Q('n', (0, -110), 'd', gate='right', rot=3, name='M_2')
    for d, nm in ((m1, 'V_1'), (m2, 'V_2')):
        c.w(d.d, (d.d[0] - 40, d.d[1])); c.port((d.d[0] - 40, d.d[1]), nm, 'l')
    c.w(m1.g, (m1.g[0], m1.g[1] - 20)); c.port((m1.g[0], m1.g[1] - 20), 'V_sel1', 't')
    c.w(m2.g, (m2.g[0], m2.g[1] - 20)); c.port((m2.g[0], m2.g[1] - 20), 'V_sel2', 't')
    xc, yc = 130, -55
    c.w(m1.s, (xc, 0), (xc, yc)); c.dot((xc, yc)); c.w(m2.s, (xc, -110), (xc, yc))
    c.w((xc, yc), (xc + 40, yc)); c.port((xc + 40, yc), 'V_com', 'r')


def sample_and_hold(c):
    """- | Sample-and-hold"""
    pass_device_and_capacitor(c, 'V_sample')


def low_pass(c):
    """- | Low-pass filter"""
    pass_device_and_capacitor(c, 'V_bias')


def peak_detector(c):
    """Peak detector"""
    m1 = c.Q('n', (0, 0), 'd', gate='left', name='M_1')
    top = (0, -20)
    c.w(m1.d, top); c.dot(top); c.w(top, (0, -60)); c.port((0, -60), IN, 'r')
    c.w(top, (-60, -20), (-60, m1.g[1]), m1.g)
    pk = (0, 100)
    c.w(m1.s, pk); c.dot(pk); c.w(pk, (60, 100)); c.port((60, 100), 'V_peak', 'r')
    capacitor_to_ground(c, pk)


def voltage_doubler(c):
    """- | Voltage doubler"""
    m1 = c.Q('p', (0, 0), 's', gate='left')
    c.name(m1, 'M_1', 'l', dy=-14)
    bar = 30
    c.w(m1.s, (bar, 0), (bar, 80), m1.d)
    c.dot((bar, 40)); c.w((bar, 40), (150, 40)); c.port((150, 40), OUT, 'r')
    c.w(m1.g, (-70, 40)); c.port((-70, 40), IN, 'l')
    m2 = c.Q('n', (70, 100), 's', gate='right', rot=2, name='M_2', lbl='r')
    c.w(m2.s, (70, 40)); c.dot((70, 40))
    gy = m2.d[1] + 20
    c.w(m2.d, (70, gy)); c.dot((70, gy)); c.gnd((70, gy))
    c.w(m2.g, (m2.g[0] - 20, m2.g[1]), (m2.g[0] - 20, gy), (70, gy))


# ================================================================ 7. logic gates
def nand(c):
    """NAND"""
    (m2, m1), (j,) = c.chain([('n', 'M_2'), ('n', 'M_1')], 0, 0)
    g = (0, m1.s[1] + 20); c.gnd(g); c.w(m1.s, g)
    for d, nm in ((m1, 'V_a'), (m2, 'V_b')):
        c.w(d.g, (d.g[0] - 30, d.g[1])); c.port((d.g[0] - 30, d.g[1]), nm, 'l')
    t = (0, -20); c.w(m2.d, t); c.dot(t); c.w(t, (50, t[1])); c.port((50, t[1]), OUT, 'r')
    c.w(t, (0, -50)); c.port((0, -50), 'I_bias', 'r')


def nor(c):
    """NOR"""
    m1 = c.Q('n', (0, 0), 'd', gate='left', name='M_1')
    m2 = c.Q('n', (130, 0), 'd', gate='right', name='M_2')
    for d in (m1, m2):
        g = (d.s[0], d.s[1] + 20); c.gnd(g); c.w(d.s, g)
    mx = 65
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), 'V_a', 'l')
    c.w(m2.g, (m2.g[0] + 30, m2.g[1])); c.port((m2.g[0] + 30, m2.g[1]), 'V_b', 'r')
    yt = -20
    c.w(m1.d, (0, yt), (130, yt)); c.w(m2.d, (130, yt)); c.dot((mx, yt))
    t = (mx, -40); c.w((mx, yt), t); c.dot(t); c.w(t, (170, t[1])); c.port((170, t[1]), OUT, 'r')
    c.w(t, (mx, -70)); c.port((mx, -70), 'I_bias', 'r')


def xnor(c):
    """XNOR"""
    m1 = c.Q('n', (0, 0), 'd', gate='left', name='M_1')
    m2 = c.Q('n', (140, 0), 'd', gate='right', name='M_2')
    mx = 70
    ga = (m1.g[0] - 20, m1.g[1]); gb = (m2.g[0] + 20, m2.g[1])
    c.w(m1.g, ga); c.dot(ga); c.w(ga, (ga[0] - 30, ga[1])); c.port((ga[0] - 30, ga[1]), 'V_a', 'l')
    c.w(m2.g, gb); c.dot(gb); c.w(gb, (gb[0] + 30, gb[1])); c.port((gb[0] + 30, gb[1]), 'V_b', 'r')
    yt = -20
    c.w(m1.d, (0, yt), (mx, yt)); c.w(m2.d, (140, yt), (mx, yt)); c.dot((mx, yt))
    c.w((mx, yt), (mx, -40)); c.dot((mx, -40)); c.w((mx, -40), (gb[0] + 30, -40)); c.port((gb[0] + 30, -40), OUT, 'r')
    c.w((mx, -40), (mx, -70)); c.port((mx, -70), 'I_bias', 'r')
    y1, y2 = m1.s[1] + 20, m1.s[1] + 40
    c.w(m1.s, (0, y1), (mx - 15, y1), (mx + 15, y2), (gb[0], y2), gb)
    c.w(m2.s, (140, y1), (mx + 15, y1), (mx - 15, y2), (ga[0], y2), ga)


def mixer(c):
    """- | Mixer"""
    (m2, m1), (j,) = c.chain([('n', 'M_2'), ('n', 'M_1')], 0, 0)
    g = (0, m1.s[1] + 20); c.gnd(g); c.w(m1.s, g)
    c.w(m2.d, (0, -50)); c.port((0, -50), 'I_out', 'r')
    c.w(m2.g, (m2.g[0] - 30, m2.g[1])); c.port((m2.g[0] - 30, m2.g[1]), 'V_lo', 'l')
    c.w(m1.g, (m1.g[0] - 30, m1.g[1])); c.port((m1.g[0] - 30, m1.g[1]), 'V_rf', 'l')


# ================================================================ registry
def names(f):
    """(BJT name, MOSFET name) from the first docstring line; '' where the circuit has no such version"""
    parts = [p.strip() for p in (f.__doc__ or f.__name__).strip().splitlines()[0].split('|')]
    return tuple('' if p == '-' else p for p in (parts[0], parts[-1]))


TYPES = (n_type, p_type)
HELPERS = (size_label, capacitor_to_ground, pass_device_and_capacitor)

# one transistor: (heading, first connection, second connection)
ROLES = [
    ('Amplifier and Follower', role_amplifier, role_follower),
    ('Current Buffer and Current Source', role_current_buffer, role_current_source),
    ('Diode and Switch', role_diode, role_switch),
    ('Resistor and Capacitor', role_resistor, role_capacitor),
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
