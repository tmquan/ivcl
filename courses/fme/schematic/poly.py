"""Small polygon helpers (pure Python): simplification, nesting of rings and hole bridging."""
import math


def area(r):
    s = 0.0
    n = len(r)
    for i in range(n):
        x0, y0 = r[i]; x1, y1 = r[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def inside(pt, ring):
    x, y = pt; n = len(ring); c = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]; xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            c = not c
        j = i
    return c


def bridge(outer, hole):
    """join `hole` into `outer` with a zero-width horizontal bridge from the hole's right-most vertex"""
    mi = max(range(len(hole)), key=lambda i: hole[i][0])
    mx, my = hole[mi]
    best = None
    n = len(outer)
    for i in range(n):
        ax, ay = outer[i]; bx, by = outer[(i + 1) % n]
        if ay == by or (ay - my) * (by - my) > 0:
            continue
        xi = ax + (my - ay) * (bx - ax) / (by - ay)
        if xi >= mx - 1e-9 and (best is None or xi < best[0]):
            best = (xi, i)
    if best is None:
        return None
    xi, i = best
    I = (xi, my)
    loop = hole[mi:] + hole[:mi] + [hole[mi]]
    return outer[:i + 1] + [I] + loop + [I] + outer[i + 1:]


def dedupe(r, eps=1e-6):
    out = []
    for p in r:
        if not out or abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) < eps and abs(out[0][1] - out[-1][1]) < eps:
        out.pop()
    return out


def solid_rings(rings):
    """rings (outer boundaries and holes in any order/orientation) -> hole-free closed polygons.
    Holes are merged into their outer ring by zero-width bridges, so a plain non-zero or
    even-odd fill of each returned polygon reproduces the shape."""
    rings = [dedupe(r) for r in rings]
    rings = [r for r in rings if len(r) >= 3 and abs(area(r)) > 1e-9]
    depth = [sum(1 for j, o in enumerate(rings) if j != i and inside(r[0], o)) for i, r in enumerate(rings)]
    result = []
    for oi in range(len(rings)):
        if depth[oi] % 2:
            continue
        outer = rings[oi]
        if area(outer) < 0:
            outer = outer[::-1]
        holes = [rings[h] for h in range(len(rings)) if depth[h] == depth[oi] + 1 and inside(rings[h][0], rings[oi])]
        holes = [h if area(h) < 0 else h[::-1] for h in holes]
        for h in sorted(holes, key=lambda h: -max(p[0] for p in h)):
            merged = bridge(outer, h)
            if merged is not None:
                outer = merged
        result.append(dedupe(outer))
    return result


def simplify(ring, tol):
    """Douglas-Peucker simplification of a closed ring (max deviation `tol`)"""
    n = len(ring)
    if n < 5:
        return ring
    i1 = max(range(n), key=lambda i: (ring[i][0] - ring[0][0]) ** 2 + (ring[i][1] - ring[0][1]) ** 2)

    def rdp(pts):
        keep = [False] * len(pts); keep[0] = keep[-1] = True
        stack = [(0, len(pts) - 1)]
        while stack:
            a, b = stack.pop()
            if b <= a + 1:
                continue
            ax, ay = pts[a]; bx, by = pts[b]; dx, dy = bx - ax, by - ay; L = math.hypot(dx, dy)
            dmax, idx = -1.0, -1
            for i in range(a + 1, b):
                px, py = pts[i]
                d = abs(dx * (py - ay) - dy * (px - ax)) / L if L > 0 else math.hypot(px - ax, py - ay)
                if d > dmax:
                    dmax, idx = d, i
            if dmax > tol:
                keep[idx] = True; stack += [(a, idx), (idx, b)]
        return [p for p, k in zip(pts, keep) if k]
    a = rdp(ring[:i1 + 1]); b = rdp(ring[i1:] + [ring[0]])
    return a[:-1] + b[:-1]
