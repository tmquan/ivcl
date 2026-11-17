"""Scene graph: symbols + wires + dots + labels, with continuous connectors.

`Scene.connectors()` merges every wire piece and every symbol lead into maximal unbroken
polylines: a connector runs from inside one symbol body (its anchor) to the next body,
terminal or junction, turning corners as needed.  T-junctions end the side branch only.
Every corner of a connector is rounded; the corners inside a symbol stay as drawn.
"""
import json
import math
import base64
from xml.sax.saxutils import escape
from . import symbols as S
from . import tex as T

WIRE_W = S.WIRE * S.S                                   # connector stroke (px)
RING_R, RING_W = S.RING['r'] * S.S, S.RING['w'] * S.S   # open terminal: centre-line radius, stroke
DOT_R = S.DOT['r'] * S.S                                # node dot radius
LABEL = 16                                              # default label size (px)
CORNER = 8.0                                            # radius of the rounded connector corners (px)
BASE = 0.33                                             # baseline offset below the label's y, in font sizes


class Inst:
    def __init__(s, key, x, y, rot, mirror, sym):
        s.key, s.x, s.y, s.rot, s.mirror = key, x, y, rot, mirror
        s.sym = sym
        s.w, s.h = S.oriented(sym, rot, mirror)
        s.t = {k: (x + p[0], y + p[1]) for k, p in S.terminals(sym, rot, mirror).items()}
        s.a = {k: (x + p[0], y + p[1]) for k, p in S.anchors(sym, rot, mirror).items()}


def _key(p):
    return (round(p[0], 3), round(p[1], 3))


def _on_seg(p, a, b):
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if abs(cross) > 1e-6:
        return False
    d = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
    L2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    return 1e-6 < d < L2 - 1e-6


def _straight(a, n, b):
    """a - n - b collinear with n between"""
    v1 = (a[0] - n[0], a[1] - n[1]); v2 = (b[0] - n[0], b[1] - n[1])
    return abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-6 and (v1[0] * v2[0] + v1[1] * v2[1]) < 0


def _fillet(path, r, steps=8):
    """polyline with its corners replaced by circular arcs of radius r (each given as `steps` chords).
    Two neighbouring corners share the segment between them half and half; the first and the last
    segment, which run into a symbol or end at a terminal, are available to their corner almost whole."""
    last = len(path) - 2
    out = [path[0]]
    for k in range(1, len(path) - 1):
        a, p, b = path[k - 1], path[k], path[k + 1]
        ux, uy, la = p[0] - a[0], p[1] - a[1], math.hypot(p[0] - a[0], p[1] - a[1])
        vx, vy, lb = b[0] - p[0], b[1] - p[1], math.hypot(b[0] - p[0], b[1] - p[1])
        if la < 1e-9 or lb < 1e-9:
            out.append(p); continue
        ux, uy, vx, vy = ux / la, uy / la, vx / lb, vy / lb
        cross, dot = ux * vy - uy * vx, ux * vx + uy * vy
        turn = math.atan2(abs(cross), dot)                     # 0 = straight on
        if turn < 1e-3 or turn > math.pi - 1e-3:
            out.append(p); continue
        room_a = la - 1.5 if k == 1 else la / 2
        room_b = lb - 1.5 if k == last else lb / 2
        d = min(r * math.tan(turn / 2), room_a, room_b)        # distance from the corner to the arc ends
        rr = d / math.tan(turn / 2)
        sgn = 1.0 if cross > 0 else -1.0
        t1 = (p[0] - ux * d, p[1] - uy * d)
        cx, cy = t1[0] - sgn * uy * rr, t1[1] + sgn * ux * rr  # centre: off the incoming segment, on the inside
        a0 = math.atan2(t1[1] - cy, t1[0] - cx)
        for j in range(steps + 1):
            ang = a0 + sgn * turn * j / steps
            out.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    out.append(path[-1])
    return out


class Scene:
    def __init__(s, name='', title='', note=''):
        s.name, s.title, s.note = name, title, note
        s.insts, s.wires, s.dots, s.texts = [], [], [], []

    # -- building ---------------------------------------------------
    def add(s, key, x, y, rot=0, mirror=False):
        i = Inst(key, x, y, rot, mirror, S.SYM[key])
        s.insts.append(i)
        return i

    def put(s, key, term, at, rot=0, mirror=False):
        """place symbol so that terminal `term` is at point `at`"""
        i0 = Inst(key, 0, 0, rot, mirror, S.SYM[key])
        tx, ty = i0.t[term]
        return s.add(key, at[0] - tx, at[1] - ty, rot, mirror)

    def wire(s, *pts):
        out = []
        for p in pts:
            p = (round(p[0], 3), round(p[1], 3))
            if not out or out[-1] != p:
                out.append(p)
        if len(out) > 1:
            s.wires.append(out)

    def dot(s, p, open_=False):
        s.dots.append((p[0], p[1], open_))

    def text(s, txt, x, y, anchor='middle', size=None, mode='math'):
        """mode 'math': label mini-markup (see tex.py); 'text'/'bold': plain Computer Modern"""
        s.texts.append((txt, x, y, anchor, size or LABEL, mode))

    # -- continuous connectors ---------------------------------------
    def connectors(s, corner=None):
        """continuous connectors as polylines.  Every corner of a connector is rounded with radius
        `corner` px (default CORNER; 0 = sharp).  The corners inside a symbol belong to the symbol
        and stay as drawn."""
        corner = CORNER if corner is None else corner
        segs, ends = [], set()
        for w in s.wires:
            w = [_key(p) for p in w]
            ends.add(w[0]); ends.add(w[-1])
            segs += [(a, b) for a, b in zip(w[:-1], w[1:]) if a != b]
        for i in s.insts:
            for k, tp in i.t.items():
                a, b = _key(tp), _key(i.a[k])
                ends.add(a); ends.add(b)
                if a != b:
                    segs.append((a, b))
        atoms = set()
        for a, b in segs:
            cut = sorted((p for p in ends if _on_seg(p, a, b)), key=lambda p: (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2)
            ch = [a] + cut + [b]
            for u, v in zip(ch[:-1], ch[1:]):
                if u != v:
                    atoms.add((u, v) if u < v else (v, u))
        adj = {}
        for u, v in atoms:
            adj.setdefault(u, []).append(v); adj.setdefault(v, []).append(u)
        cont = {}                                   # (node, came_from) -> continue_to
        for n, nb in adj.items():
            if len(nb) == 2:
                cont[(n, nb[0])] = nb[1]; cont[(n, nb[1])] = nb[0]
            elif len(nb) > 2:
                used = set()
                for i in range(len(nb)):
                    for j in range(i + 1, len(nb)):
                        if i in used or j in used:
                            continue
                        if _straight(nb[i], n, nb[j]):
                            cont[(n, nb[i])] = nb[j]; cont[(n, nb[j])] = nb[i]; used.add(i); used.add(j)
        seen = set()
        edge = lambda u, v: (u, v) if u < v else (v, u)

        def walk(u, v):
            path = [u, v]; seen.add(edge(u, v))
            prev, cur = u, v
            while (cur, prev) in cont:
                nx = cont[(cur, prev)]
                if edge(cur, nx) in seen:
                    break
                path.append(nx); seen.add(edge(cur, nx)); prev, cur = cur, nx
            return path
        paths = []
        for u in sorted(adj):
            for v in sorted(adj[u]):
                if edge(u, v) not in seen and (u, v) not in cont:      # u is a true end for this edge
                    paths.append(walk(u, v))
        for u in sorted(adj):                                           # closed loops
            for v in sorted(adj[u]):
                if edge(u, v) not in seen:
                    paths.append(walk(u, v))
        out = []
        for p in paths:
            q = [p[0]]
            for k in range(1, len(p) - 1):
                if not _straight(q[-1], p[k], p[k + 1]):
                    q.append(p[k])
            q.append(p[-1])
            out.append(q)
        if corner:
            out = [_fillet(q, corner) for q in out]
        return out

    # -- bounds -----------------------------------------------------
    def text_box(s, t):
        """(x0, y0, x1, y1) of a label in scene px (y down)"""
        txt, x, y, a, sz, mode = t
        try:
            src = T.tex(txt) if mode == 'math' else txt
            xa, ya, xb, yb = T.extent(src, sz, mode)
            wd = xb - xa
            top, bot = y + BASE * sz - yb, y + BASE * sz - ya
        except Exception:                             # matplotlib not available: rough estimate
            wd = 0.55 * sz * len(txt.replace('_', '').replace('{', '').replace('}', ''))
            top, bot = y - 0.6 * sz, y + 0.6 * sz
        x0 = {'start': x, 'middle': x - wd / 2, 'end': x - wd}[a]
        return x0, top, x0 + wd, bot

    def bounds(s, texts=True):
        xs, ys = [], []
        for i in s.insts:
            xs += [i.x, i.x + i.w]; ys += [i.y, i.y + i.h]
        for w in s.wires:
            for p in w:
                xs.append(p[0]); ys.append(p[1])
        for d in s.dots:
            r = RING_R + RING_W / 2
            xs += [d[0] - r, d[0] + r]; ys += [d[1] - r, d[1] + r]
        if texts:
            for t in s.texts:
                x0, y0, x1, y1 = s.text_box(t)
                xs += [x0, x1]; ys += [y0, y1]
        return min(xs), min(ys), max(xs), max(ys)

    # -- SVG --------------------------------------------------------
    def svg_parts(s):
        g = []
        for w in s.connectors():
            g.append('<polyline points="%s" fill="none"/>' % ' '.join('%.2f,%.2f' % p for p in w))
        for i in s.insts:
            cw, ch, body = S.svg_body(i.sym, i.rot, i.mirror, 'body')
            g.append('<g transform="translate(%.2f,%.2f)">%s</g>' % (i.x, i.y, body))
        for x, y, op in s.dots:
            if op:
                g.append('<circle cx="%.2f" cy="%.2f" r="%.2f" fill="#fff" stroke-width="%.2f"/>' % (x, y, RING_R, RING_W))
            else:
                g.append('<circle cx="%.2f" cy="%.2f" r="%.2f" fill="#000" stroke="none"/>' % (x, y, DOT_R))
        for txt, x, y, a, sz, mode in s.texts:
            src = T.tex(txt) if mode == 'math' else txt
            g.append('<path d="%s" fill="#000" stroke="none"/>' % T.svg_d(src, sz, x, y + BASE * sz, a, mode))
        return ''.join(g)

    def to_svg(s, margin=30, caption=True):
        x0, y0, x1, y1 = s.bounds()
        capH = 50 if caption else 0
        need = 0
        if caption:
            need = 24 + max(T.extent(s.title, 16, 'bold')[2], T.extent(s.note, 11, 'text')[2] if s.note else 0)
        W = max((x1 - x0) + 2 * margin, need)
        H = (y1 - y0) + 2 * margin + capH
        ox = margin - x0 + (W - ((x1 - x0) + 2 * margin)) / 2
        oy = margin - y0 + capH
        cap = ''
        if caption:
            cap = '<path d="%s" fill="#000" stroke="none"/>' % T.svg_d(s.title, 16, 12, 24, 'start', 'bold')
            if s.note:
                cap += '<path d="%s" fill="#555" stroke="none"/>' % T.svg_d(s.note, 11, 12, 42, 'start', 'text')
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" viewBox="0 0 %.0f %.0f" '
                'stroke="#000" stroke-width="%.3f" stroke-linecap="butt" stroke-linejoin="miter" stroke-miterlimit="10" fill="none">'
                '<rect width="100%%" height="100%%" fill="#fff" stroke="none"/>%s'
                '<g transform="translate(%.2f,%.2f)">%s</g></svg>'
                % (W, H, W, H, WIRE_W, cap, ox, oy, s.svg_parts()))

    # -- draw.io ----------------------------------------------------
    def to_drawio_cells(s, margin=40, with_title=True):
        x0, y0, x1, y1 = s.bounds()
        ox = margin - x0
        oy = margin - y0 + (40 if with_title else 0)
        cells = []
        nid = [2]

        def new():
            nid[0] += 1
            return 'c%d' % nid[0]

        q = lambda v: escape(v, {'"': '&quot;'})
        if with_title:
            cells.append('<mxCell id="%s" value="%s" style="text;html=1;strokeColor=none;fillColor=none;align=left;'
                         'verticalAlign=middle;fontStyle=1;fontSize=16;fontFamily=Georgia;" vertex="1" parent="1">'
                         '<mxGeometry x="10" y="8" width="620" height="24" as="geometry"/></mxCell>' % (new(), q(s.title)))
        for w in s.connectors():
            pts = [(p[0] + ox, p[1] + oy) for p in w]
            mid = ''.join('<mxPoint x="%.2f" y="%.2f"/>' % p for p in pts[1:-1])
            arr = '<Array as="points">%s</Array>' % mid if mid else ''
            cells.append('<mxCell id="%s" value="" style="endArrow=none;html=1;rounded=0;strokeWidth=%.2f;edgeStyle=none;" '
                         'edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                         '<mxPoint x="%.2f" y="%.2f" as="sourcePoint"/><mxPoint x="%.2f" y="%.2f" as="targetPoint"/>%s'
                         '</mxGeometry></mxCell>' % (new(), WIRE_W, pts[0][0], pts[0][1], pts[-1][0], pts[-1][1], arr))
        pad = S.PAD
        for i in s.insts:
            svg = S.svg_doc(i.sym, i.rot, i.mirror, which='body')
            b = base64.b64encode(svg.encode()).decode()
            pts = json.dumps(S.conn_points(i.sym, i.rot, i.mirror, which='anchor')).replace(' ', '')
            cells.append('<mxCell id="%s" value="" style="shape=image;aspect=fixed;imageAspect=0;verticalLabelPosition=bottom;'
                         'points=%s;image=data:image/svg+xml,%s;" vertex="1" parent="1"><mxGeometry x="%.2f" y="%.2f" width="%d" height="%d" as="geometry"/></mxCell>'
                         % (new(), pts, b, i.x + ox - pad, i.y + oy - pad, i.w + 2 * pad, i.h + 2 * pad))
        for x, y, op in s.dots:
            r = RING_R if op else DOT_R
            st = ('strokeWidth=%.2f;fillColor=#ffffff;strokeColor=#000000;' % RING_W) if op else 'strokeColor=none;fillColor=#000000;'
            cells.append('<mxCell id="%s" value="" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;%s" '
                         'vertex="1" parent="1"><mxGeometry x="%.2f" y="%.2f" width="%.2f" height="%.2f" as="geometry"/></mxCell>'
                         % (new(), st, x + ox - r, y + oy - r, 2 * r, 2 * r))
        for txt, x, y, a, sz, mode in s.texts:
            al = {'start': 'left', 'middle': 'center', 'end': 'right'}[a]
            wd = 160
            px = {'start': x, 'middle': x - wd / 2, 'end': x - wd}[a]
            val = '\\(%s\\)' % T.tex(txt) if mode == 'math' else txt
            cells.append('<mxCell id="%s" value="%s" style="text;html=1;strokeColor=none;fillColor=none;align=%s;verticalAlign=middle;'
                         'fontSize=%d;spacing=0;whiteSpace=nowrap;" vertex="1" parent="1">'
                         '<mxGeometry x="%.2f" y="%.2f" width="%d" height="26" as="geometry"/></mxCell>'
                         % (new(), q(val), al, sz, px + ox, y + oy - 13, wd))
        return cells

    def to_drawio_page(s, idx):
        cells = ''.join(s.to_drawio_cells())
        return ('<diagram name="%s" id="p%d"><mxGraphModel dx="1000" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" '
                'connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="1100" pageHeight="850" math="1" shadow="0">'
                '<root><mxCell id="0"/><mxCell id="1" parent="0"/>%s</root></mxGraphModel></diagram>'
                % (escape(s.name, {'"': '&quot;'}), idx, cells))
