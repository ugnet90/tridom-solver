#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import itertools
import time
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# -----------------------------
# Tile set (0–5), 76 tiles:
# - rotations equivalent
# - ALL_DIFFERENT tiles have CW and CCW variants (NOT mirror-equivalent)
# Corner labels are numbers on vertices; placing a tile assigns numbers to triangle vertices in CCW order.
# -----------------------------

DIGITS = list(range(6))

def rot3(t):
    a, b, c = t
    return [(a,b,c),(b,c,a),(c,a,b)]

# For identifying a tile up to rotation (NOT reflection)
def canon_cyclic(t):
    return min(rot3(t))

def build_tiles_76():
    tiles = []
    tile_rots = []
    tile_kind = []

    # triples
    for x in DIGITS:
        t = (x,x,x)
        tiles.append(canon_cyclic(t))
        tile_rots.append([(x,x,x)])
        tile_kind.append("TRIPLE")

    # doubles xxy (y != x)
    for x in DIGITS:
        for y in DIGITS:
            if y == x:
                continue
            rots = list(dict.fromkeys([(x,x,y),(x,y,x),(y,x,x)]))
            tiles.append(canon_cyclic((x,x,y)))
            tile_rots.append(rots)
            tile_kind.append("DOUBLE")

    # all-different: two chiral variants per {a,b,c}
    for a,b,c in itertools.combinations(DIGITS,3):
        for t in [(a,b,c),(a,c,b)]:  # CW vs CCW representative
            rots = list(dict.fromkeys(rot3(t)))
            tiles.append(canon_cyclic(t))
            tile_rots.append(rots)
            tile_kind.append("ALL_DIFF")

    assert len(tiles) == 76
    assert len(set(tiles)) == 76
    return tiles, tile_rots, tile_kind

TILES, TILE_ROTS, TILE_KIND = build_tiles_76()

# Map oriented triple -> tile index (unique in this 76-set definition)
ORIENT_TO_TILE = {}
for i in range(76):
    for ot in TILE_ROTS[i]:
        ORIENT_TO_TILE[ot] = i

# -----------------------------
# Geometry: triangular lattice and shapes
# -----------------------------

SQRT3 = math.sqrt(3)
E1 = (1.0, 0.0)
E2 = (0.5, SQRT3/2)

def pt(i,j):
    return (i*E1[0] + j*E2[0], i*E1[1] + j*E2[1])

DIRS = [
    (1.0, 0.0),
    (0.5, SQRT3/2),
    (-0.5, SQRT3/2),
    (-1.0, 0.0),
    (-0.5, -SQRT3/2),
    (0.5, -SQRT3/2),
]

def add(a,b,scale=1.0):
    return (a[0]+scale*b[0], a[1]+scale*b[1])

def hex_polygon(side=4, start=(0.0,0.0), start_dir_index=0):
    v=[start]
    cur=start
    for k in range(6):
        d=DIRS[(start_dir_index+k)%6]
        cur=add(cur,d,side)
        v.append(cur)
    return v[:-1]

def point_in_poly(x,y,poly):
    inside=False
    n=len(poly)
    for i in range(n):
        x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
        if ((y1>y) != (y2>y)):
            xint = (x2-x1)*(y-y1)/(y2-y1+1e-18) + x1
            if x < xint:
                inside = not inside
    return inside

def tri_centroid(tri):
    (x1,y1),(x2,y2),(x3,y3)=tri
    return ((x1+x2+x3)/3, (y1+y2+y3)/3)

def signed_area(a,b,c):
    return (a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))/2

def triangle_area(a,b,c):
    return abs(signed_area(a,b,c))

def point_in_tri(p,a,b,c):
    A = triangle_area(a,b,c)
    A1 = triangle_area(p,b,c)
    A2 = triangle_area(a,p,c)
    A3 = triangle_area(a,b,p)
    return abs((A1+A2+A3)-A) < 1e-6

def build_shape_A_fast_hex_76():
    """
    Base hex side 4 has 96 unit triangles.
    Cut: 3 corners with cut_side=2 (6 each) and 1 corner with cut_side=1 (2) -> remove 20 => 76 kept.
    Returns: tri_vid (triangle vertices in CCW, each vertex is an (i,j) lattice index),
             tri_geom (triangle coordinates)
    """
    S = 4
    hex_poly = hex_polygon(side=S, start=(0.0,0.0), start_dir_index=0)

    xs=[p[0] for p in hex_poly]; ys=[p[1] for p in hex_poly]
    minx,maxx=min(xs)-2, max(xs)+2
    miny,maxy=min(ys)-2, max(ys)+2

    triangles=[]
    tri_meta=[]
    for i in range(-10, 40):
        for j in range(-10, 40):
            p00=pt(i,j); p10=pt(i+1,j); p01=pt(i,j+1); p11=pt(i+1,j+1)

            up=(p00,p10,p01)
            c=tri_centroid(up)
            if (minx<=c[0]<=maxx) and (miny<=c[1]<=maxy) and point_in_poly(c[0],c[1],hex_poly):
                triangles.append(up)
                tri_meta.append(((i,j),(i+1,j),(i,j+1)))

            down=(p11,p01,p10)
            c=tri_centroid(down)
            if (minx<=c[0]<=maxx) and (miny<=c[1]<=maxy) and point_in_poly(c[0],c[1],hex_poly):
                triangles.append(down)
                tri_meta.append(((i+1,j+1),(i,j+1),(i+1,j)))

    assert len(triangles)==96

    # corner cuts
    cuts = {1:2, 2:2, 3:2, 4:1}
    hex_v = hex_poly
    cut_polys=[]
    for k,cs in cuts.items():
        V=hex_v[k]
        prev=hex_v[(k-1)%6]; nxt=hex_v[(k+1)%6]
        d1=((prev[0]-V[0])/S, (prev[1]-V[1])/S)
        d2=((nxt[0]-V[0])/S, (nxt[1]-V[1])/S)
        cut_polys.append((V, add(V,d1,cs), add(V,d2,cs)))

    keep_idx=[]
    for idx,tri in enumerate(triangles):
        c=tri_centroid(tri)
        if any(point_in_tri(c,A,B,C) for A,B,C in cut_polys):
            continue
        keep_idx.append(idx)

    assert len(keep_idx)==76

    tri_vid=[]
    tri_geom=[]
    for idx in keep_idx:
        a,b,c = tri_meta[idx]
        pts = [pt(*a), pt(*b), pt(*c)]
        if signed_area(*pts) < 0:
            b,c = c,b
            pts = [pt(*a), pt(*b), pt(*c)]
        tri_vid.append((a,b,c))
        tri_geom.append(tuple(pts))

    return tri_vid, tri_geom

def build_shape_B_parallelogram_76(w=19, h=2):
    """
    Simple straight-edged parallelogram on the triangular lattice using rhombi (2 triangles each):
    Build w*h 'rhombi' => 2*w*h triangles.
    Default w=19, h=2 gives 76 triangles.
    """
    assert 2*w*h == 76
    tri_vid=[]
    tri_geom=[]
    # Use the unit cell defined by points (i,j),(i+1,j),(i,j+1),(i+1,j+1)
    # For each (i,j) in rectangle, add two triangles.
    for i in range(w):
        for j in range(h):
            # vertices
            a=(i,j); b=(i+1,j); c=(i,j+1); d=(i+1,j+1)
            # up triangle (a,b,c) CCW?
            pts=[pt(*a),pt(*b),pt(*c)]
            if signed_area(*pts)<0: b,c=c,b; pts=[pt(*a),pt(*b),pt(*c)]
            tri_vid.append((a,b,c)); tri_geom.append(tuple(pts))
            # down triangle (d,c,b)
            pts=[pt(*d),pt(*c),pt(*b)]
            if signed_area(*pts)<0: c,b=b,c; pts=[pt(*d),pt(*c),pt(*b)]
            tri_vid.append((d,c,b)); tri_geom.append(tuple(pts))
    return tri_vid, tri_geom

# -----------------------------
# Solver: backtracking with propagation
# -----------------------------

def solve(tri_vid, time_limit=0, seed=1, verbose=True):
    """
    tri_vid: list of triangles, each triangle is 3 lattice-vertex IDs (i,j) in CCW order.
    Returns assignment list: for each triangle index -> (tile_index, oriented_triple)
    where oriented_triple = (val_at_v0, val_at_v1, val_at_v2) in CCW order.
    """
    random.seed(seed)

    # compress vertex ids
    vertices=sorted({v for tri in tri_vid for v in tri})
    vid={v:i for i,v in enumerate(vertices)}
    triV=[(vid[a],vid[b],vid[c]) for a,b,c in tri_vid]
    V=len(vertices); T=len(triV)

    tris_by_vertex=[[] for _ in range(V)]
    for t,(a,b,c) in enumerate(triV):
        tris_by_vertex[a].append(t); tris_by_vertex[b].append(t); tris_by_vertex[c].append(t)

    domains=[set(DIGITS) for _ in range(V)]
    tri_assign=[None]*T
    used=[False]*76

    # Symmetry breaking: fix first vertex to 0 and second to 1 if possible.
    domains[0]={0}
    if V>1:
        domains[1]={1}

    def tri_candidates(t):
        a,b,c=triV[t]
        A,B,C=domains[a],domains[b],domains[c]
        out=[]
        for va in A:
            for vb in B:
                for vc in C:
                    ot=(va,vb,vc)
                    ti=ORIENT_TO_TILE.get(ot)
                    if ti is None or used[ti]:
                        continue
                    out.append((ti,ot))
        return out

    def choose_next():
        best=None; bestc=None; bestkey=None
        for t in range(T):
            if tri_assign[t] is not None:
                continue
            a,b,c=triV[t]
            fixed = (len(domains[a])==1)+(len(domains[b])==1)+(len(domains[c])==1)
            cands=tri_candidates(t)
            key=(len(cands), -fixed)
            if best is None or key<bestkey:
                best=t; bestc=cands; bestkey=key
                if len(cands)==0:
                    return best,[]
        return best,bestc

    def apply(t, ti, ot):
        tri_assign[t]=(ti,ot)
        used[ti]=True
        changes=[]
        for v,val in zip(triV[t], ot):
            if domains[v] != {val}:
                changes.append((v, domains[v]))
                domains[v]={val}
        return changes

    def undo(t, ti, changes):
        tri_assign[t]=None
        used[ti]=False
        for v,old in changes:
            domains[v]=old

    def local_ok(changes):
        affected=set()
        for v,_ in changes:
            for t2 in tris_by_vertex[v]:
                if tri_assign[t2] is None:
                    affected.add(t2)
        for t2 in affected:
            if len(tri_candidates(t2))==0:
                return False
        return True

    start=time.time()
    best_progress=0
    nodes=0

    def bt():
        nonlocal best_progress, nodes
        if time_limit and (time.time()-start) > time_limit:
            return False
        filled = T - sum(1 for x in tri_assign if x is None)
        if filled > best_progress:
            best_progress = filled
            if verbose:
                print(f"[progress] filled {filled}/{T}, nodes={nodes}")
        if filled == T:
            return True

        t, cands = choose_next()
        if cands == []:
            return False

        # heuristic: place tiles that fix more new vertices first; also try TRIPLE earlier.
        a,b,c = triV[t]
        def score(item):
            ti,ot=item
            new = sum(1 for v in (a,b,c) if len(domains[v])>1)
            kind = TILE_KIND[ti]
            kind_rank = 2 if kind=="TRIPLE" else (1 if kind=="DOUBLE" else 0)
            return (-new, -kind_rank, random.random())

        for ti,ot in sorted(cands, key=score):
            nodes += 1
            ch=apply(t,ti,ot)
            if local_ok(ch):
                if bt():
                    return True
            undo(t,ti,ch)
        return False

    ok = bt()
    if not ok and verbose:
        print(f"[result] no solution found within constraints/time_limit={time_limit}. best filled={best_progress}/{T}")
    if ok and verbose:
        print("[result] SOLVED")
    return ok, tri_assign, vertices

# -----------------------------
# Rendering
# -----------------------------

def render_solution(tri_geom, tri_vid, vertices, tri_assign, out_png, out_pdf, title):
    # Build mapping vertex->value from any triangle assignment
    vval={}
    for (ti,ot), tri in zip(tri_assign, tri_vid):
        if ti is None:
            continue
        for v,val in zip(tri, ot):
            vval[v]=val

    fig, ax = plt.subplots(figsize=(9,8))
    patches=[]
    for tri in tri_geom:
        patches.append(Polygon(tri, closed=True))
    pc=PatchCollection(patches, facecolor="0.95", edgecolor="0.35", linewidth=0.6)
    ax.add_collection(pc)

    # annotate vertex values
    for v in vertices:
        x,y = pt(*v)
        val = vval.get(v, None)
        if val is not None:
            ax.text(x, y, str(val), ha="center", va="center", fontsize=8)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Main: try A then B
# -----------------------------

def main():
    print("Tridom 0–5 (76 tiles, CW/CCW distinct for xyz): Try A (fast-hex 76) then B (parallelogram 19x2).")

    tri_vid_A, tri_geom_A = build_shape_A_fast_hex_76()
    okA, assignA, verticesA = solve(tri_vid_A, time_limit=0, seed=1, verbose=True)
    if okA:
        render_solution(tri_geom_A, tri_vid_A, verticesA, assignA,
                        out_png="solution_A.png", out_pdf="solution_A.pdf",
                        title="Lösung A: fast-Hexagon (76 Dreiecke) – Eckenzahlen an den Gitterpunkten")
        print("Wrote solution_A.png / solution_A.pdf")
        return

    # If A not solved (or takes too long), try B
    tri_vid_B, tri_geom_B = build_shape_B_parallelogram_76(w=19,h=2)
    okB, assignB, verticesB = solve(tri_vid_B, time_limit=0, seed=2, verbose=True)
    if okB:
        render_solution(tri_geom_B, tri_vid_B, verticesB, assignB,
                        out_png="solution_B.png", out_pdf="solution_B.pdf",
                        title="Lösung B: Parallelogramm 19×2 Rhomben (76 Dreiecke)")
        print("Wrote solution_B.png / solution_B.pdf")
        return

    print("No solution found (with current search). Consider changing B-shape (w,h) or adding time_limit/restraints.")

if __name__ == "__main__":
    main()
