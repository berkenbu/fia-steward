#!/usr/bin/env python3
"""
FIA STEWARD SIMULATOR  -  you are the race steward.
Watch a 3D race, spot incidents, and file penalty reports.

Requirements:  Python 3.9+   and   pip install panda3d
Run:           python fia_steward.py
(Panda3D = real 3D engine with shadows + Bullet rigid-body physics. Not HTML.)
"""
import math, random, sys, os, bisect
from collections import deque

from panda3d.core import (loadPrcFileData, Vec3, Point3, Point2, TextNode, CardMaker, LineSegs,
    AmbientLight, DirectionalLight, Fog, TransparencyAttrib, GeomVertexFormat,
    GeomVertexData, GeomVertexWriter, Geom, GeomTriangles, GeomNode, AntialiasAttrib,
    PNMImage, Texture, SamplerState, GeomVertexArrayFormat, NodePath, BitMask32)
loadPrcFileData("", "window-title FIA Steward Simulator\nwin-size 1600 900\n"
                    "sync-video 1\nmultisamples 4\nframebuffer-multisample 1\n")
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletBoxShape
try:                                    # optional visual upgrades: pip install panda3d-simplepbr panda3d-gltf
    import simplepbr
except ImportError:
    simplepbr = None
try:
    import gltf
except ImportError:
    gltf = None
NO_SHADOW = BitMask32.bit(5)           # the sun's shadow camera only renders things not hidden from this bit
HERE = os.path.dirname(os.path.abspath(__file__))
QUALITY = os.environ.get("FIA_QUALITY", "high").lower()     # high | low (low = no PBR/bloom)

# ------------------------------------------------------------------ constants
LAPS = 6
G = 9.81
MASS = 800.0
HALF_W = 7.5                    # half track width (m) - white line
PIT_LIMIT = 80.0 / 3.6          # pit lane limit (m/s)
PIT_ZONE = 190.0                # |dist from line| inside which the limit applies
PIT_C, PIT_FAST, PIT_WORK = -14.5, -12.0, -17.0
LOOP = 100.0                    # timing loop spacing (m)
GRIP0, GRIPD = 14.0, 0.003      # lateral grip: a = GRIP0 + GRIPD*v^2  (aero)
# Incident-frequency presets. REALISTIC is tuned to roughly an F1 2025 grand prix:
# a handful of noted incidents per race, 1-3 of them penalty-worthy.
LEVELS = {
    "REALISTIC": dict(mistake=0.0006, aggr=0.32, pit=0.06, dive=0.20,
                      desc="About a real 2025 GP: a handful of noted incidents, 1-3 penalties."),
    "BUSY": dict(mistake=0.0025, aggr=0.70, pit=0.18, dive=0.60,
                 desc="A feisty race: roughly double the incidents."),
    "CHAOS": dict(mistake=0.0080, aggr=1.00, pit=0.35, dive=1.20,
                  desc="Lap-one carnage and constant mistakes. Good luck."),
}
CFG = dict(LEVELS["REALISTIC"])


def set_laps(n):
    global LAPS
    LAPS = n


PENALTIES = [("NO FURTHER ACTION", 0), ("WARNING", 0), ("5s TIME PENALTY", 5),
             ("10s TIME PENALTY", 10), ("DRIVE-THROUGH (+20s)", 20)]
SHORT = ["NFA", "WARNING", "+5s", "+10s", "DT(+20s)"]
DRIVERS = [(2, "MAR", (0.90, 0.12, 0.12)), (5, "DUB", (0.15, 0.40, 0.95)),
           (8, "OKA", (1.00, 0.55, 0.05)), (12, "BRU", (0.05, 0.75, 0.55)),
           (17, "SOL", (0.78, 0.78, 0.82)), (21, "FIN", (0.60, 0.25, 0.90)),
           (27, "HAR", (0.15, 0.65, 0.20)), (33, "VEG", (1.00, 0.88, 0.10)),
           (44, "WIL", (0.05, 0.78, 0.92)), (63, "CAS", (0.97, 0.35, 0.65))]


def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v
def wrap(a, L): return (a + L / 2.0) % L - L / 2.0
def ang_wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi
def crossed(a, b, p, N):
    d = (b - a) % N
    return 0 < d < 40 and 0 < (p - a) % N <= d
def fmt_lap(t):
    if t is None: return "--"
    return "%d:%06.3f" % (int(t // 60), t - 60 * int(t // 60))
def alat_max(v): return GRIP0 + GRIPD * v * v
def pitw(sr):
    a = abs(sr)
    if a >= 260: return 0.0
    if a <= 190: return 1.0
    t = (260 - a) / 70.0
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ mesh builder
class Mesh:
    FACES = [((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
             ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
             ((0, 1, 0), [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)]),
             ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
             ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
             ((0, 0, -1), [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)])]

    _fmt = None

    @staticmethod
    def fmt():
        if Mesh._fmt is None:
            a = GeomVertexArrayFormat()
            a.addColumn("vertex", 3, Geom.NTFloat32, Geom.CPoint)
            a.addColumn("normal", 3, Geom.NTFloat32, Geom.CNormal)
            a.addColumn("texcoord", 2, Geom.NTFloat32, Geom.CTexcoord)
            a.addColumn("color", 4, Geom.NTUint8, Geom.CColor)
            a.addColumn("tangent", 4, Geom.NTFloat32, Geom.CVector)   # simplepbr builds its TBN basis from this; zero = NaN lighting
            f = GeomVertexFormat(); f.addArray(a)
            Mesh._fmt = GeomVertexFormat.registerFormat(f)
        return Mesh._fmt

    def __init__(s, uv=6.0):
        s.uv = uv                                   # metres per texture tile
        s.vd = GeomVertexData("m", Mesh.fmt(), Geom.UHStatic)
        s.vw = GeomVertexWriter(s.vd, "vertex")
        s.nw = GeomVertexWriter(s.vd, "normal")
        s.cw = GeomVertexWriter(s.vd, "color")
        s.tw = GeomVertexWriter(s.vd, "texcoord")
        s.gw = GeomVertexWriter(s.vd, "tangent")
        s.prim = GeomTriangles(Geom.UHStatic)
        s.n = 0

    def tri(s, a, b, c, col, nrm=None):
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        l = math.sqrt(nx * nx + ny * ny + nz * nz)
        if l < 1e-9: return
        nx, ny, nz = nx / l, ny / l, nz / l
        if nrm and nx * nrm[0] + ny * nrm[1] + nz * nrm[2] < 0:
            b, c = c, b
            nx, ny, nz = -nx, -ny, -nz
        if abs(nz) > 0.5: tx, ty = 1.0, 0.0                       # tangent = texture U direction
        else:
            tl = math.hypot(nx, ny) or 1.0; tx, ty = -ny / tl, nx / tl
        for p in (a, b, c):
            s.vw.addData3(p[0], p[1], p[2]); s.nw.addData3(nx, ny, nz); s.gw.addData4(tx, ty, 0.0, 1.0)
            # world-planar UVs (s.uv m tiles): floors use x/y, walls use along-wall/z
            if abs(nz) > 0.5: s.tw.addData2(p[0] / s.uv, p[1] / s.uv)
            else: s.tw.addData2((p[0] + p[1]) / s.uv, p[2] / s.uv)
            s.cw.addData4(col[0], col[1], col[2], col[3] if len(col) > 3 else 1.0)
        s.prim.addVertices(s.n, s.n + 1, s.n + 2)
        s.n += 3

    def quad(s, a, b, c, d, col, nrm=None):
        s.tri(a, b, c, col, nrm); s.tri(a, c, d, col, nrm)

    def box(s, c, size, col, ang=0.0):
        ca, sa = math.cos(ang), math.sin(ang)
        hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
        def P(a, b, z):
            lx, ly = a * hx, b * hy
            return (c[0] + lx * ca - ly * sa, c[1] + lx * sa + ly * ca, c[2] + z * hz)
        for n, pts in Mesh.FACES:
            nn = (n[0] * ca - n[1] * sa, n[0] * sa + n[1] * ca, n[2])
            s.quad(P(*pts[0]), P(*pts[1]), P(*pts[2]), P(*pts[3]), col, nn)

    def frustum(s, c, z0, z1, r0, r1, col, segs=10):
        for i in range(segs):
            a0, a1 = 2 * math.pi * i / segs, 2 * math.pi * (i + 1) / segs
            am = (a0 + a1) / 2
            p0 = (c[0] + r0 * math.cos(a0), c[1] + r0 * math.sin(a0), z0)
            p1 = (c[0] + r0 * math.cos(a1), c[1] + r0 * math.sin(a1), z0)
            p2 = (c[0] + r1 * math.cos(a1), c[1] + r1 * math.sin(a1), z1)
            p3 = (c[0] + r1 * math.cos(a0), c[1] + r1 * math.sin(a0), z1)
            s.quad(p0, p1, p2, p3, col, (math.cos(am), math.sin(am), 0.2))
            s.tri((c[0], c[1], z1), p3, p2, col, (0, 0, 1))
            s.tri((c[0], c[1], z0), p1, p0, col, (0, 0, -1))

    def build(s, parent, name="mesh"):
        geom = Geom(s.vd); geom.addPrimitive(s.prim)
        gn = GeomNode(name); gn.addGeom(geom)
        return parent.attachNewNode(gn)


_TEAM_TEX = {}


def tint_car(model, col):
    """Recolour the red-dominant pixels of the car's livery texture to the team colour (needs numpy)."""
    try:
        import numpy as np
        key = tuple(round(c, 2) for c in col)
        if key not in _TEAM_TEX:
            tex = model.findAllTextures()[0]
            w, h = tex.getXSize(), tex.getYSize()
            a = np.frombuffer(bytes(tex.getRamImageAs("RGB")), dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            red = (r > 0.25) & (r > g * 1.6) & (r > b * 1.6)
            k = np.clip(r / 0.8, 0, 1.2)[..., None] * np.array(col[:3], dtype=np.float32)
            out = np.where(red[..., None], k, a)
            nt = Texture("team"); nt.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
            nt.setRamImageAs((np.clip(out, 0, 1) * 255).astype(np.uint8).tobytes(), "RGB")
            nt.setMinfilter(SamplerState.FT_linear_mipmap_linear); nt.setAnisotropicDegree(8)
            _TEAM_TEX[key] = nt
        model.setTexture(model.findAllTextureStages()[0], _TEAM_TEX[key], 10)
    except Exception as e:
        print("team tint skipped:", e)


def build_car(parent, col, helmet, num):
    """Low-poly F1 car: monocoque, nose, sidepods, wings, halo, 4 wheels, DRS flap."""
    root = parent.attachNewNode("car")
    mp = os.path.join(HERE, "models", "car.glb")
    if gltf is not None and os.path.exists(mp):
        # glTF car at models/car.glb (nose toward -Y in the shipped f2002, so it is turned 180 deg).
        # The red livery is recoloured per team; an optional node named "flap" is animated for DRS.
        model = NodePath(gltf.load_model(mp)); model.setScale(0.95); model.setH(180)
        mn, mx = model.getTightBounds()
        model.setPos(-(mn.x + mx.x) / 2, -(mn.y + mx.y) / 2 + 0.1, -mn.z)
        model.reparentTo(root)
        tint_car(model, col)
        flap = model.find("**/flap")
        if flap.isEmpty(): flap = root.attachNewNode("flap")
        tn = TextNode("num"); tn.setText(str(num)); tn.setAlign(TextNode.ACenter)
        tn.setTextColor(1, 1, 1, 1); tn.setShadow(0.06, 0.06); tn.setShadowColor(0, 0, 0, 1)
        t = root.attachNewNode(tn); t.setBillboardPointEye(); t.setScale(2.2); t.setPos(0, 0, 3.0); t.setLightOff(); t.hide(NO_SHADOW)
        return root, flap
    c = (col[0], col[1], col[2], 1); c2 = (col[0] * .6, col[1] * .6, col[2] * .6, 1)
    dk = (0.07, 0.07, 0.08, 1)
    m = Mesh()
    m.box((0, 0.2, 0.42), (0.80, 3.6, 0.40), c)
    m.box((0, 2.1, 0.30), (0.42, 1.8, 0.26), c)
    m.box((0, 0.05, 0.38), (1.55, 1.8, 0.36), c2)
    m.box((0, -0.8, 0.80), (0.45, 1.7, 0.46), c)
    m.box((0, 0.35, 0.80), (0.55, 0.5, 0.12), dk)
    m.box((0, 0.35, 0.88), (0.28, 0.30, 0.26), helmet)
    m.box((0, 2.45, 0.10), (2.0, 0.45, 0.06), dk)
    m.box((0, 0.0, 0.12), (1.2, 4.0, 0.06), dk)
    for sx in (-1, 1):
        m.box((sx * 1.0, 2.45, 0.18), (0.05, 0.5, 0.22), c)
        m.box((sx * 0.52, -2.05, 0.80), (0.05, 0.55, 0.50), c)
    m.box((0, -2.05, 0.95), (0.95, 0.45, 0.05), dk)
    m.box((0, -1.95, 0.60), (0.10, 0.3, 0.7), dk)
    m.build(root)
    flap = root.attachNewNode("flap"); flap.setPos(0, -1.85, 1.05)
    fm = Mesh(); fm.box((0, -0.15, 0.0), (0.95, 0.32, 0.04), c); fm.build(flap)
    for x, y, r, w in ((-0.9, 1.55, 0.34, 0.38), (0.9, 1.55, 0.34, 0.38),
                       (-0.9, -1.4, 0.38, 0.50), (0.9, -1.4, 0.38, 0.50)):
        wm = Mesh(); wm.frustum((0, 0), -w / 2, w / 2, r, r, (0.04, 0.04, 0.04, 1), 12)
        wn = wm.build(root); wn.setR(90); wn.setPos(x, y, r)
    tn = TextNode("num"); tn.setText(str(num)); tn.setAlign(TextNode.ACenter)
    tn.setTextColor(1, 1, 1, 1); tn.setShadow(0.06, 0.06); tn.setShadowColor(0, 0, 0, 1)
    t = root.attachNewNode(tn); t.setBillboardPointEye(); t.setScale(2.2)
    t.setPos(0, 0, 3.0); t.setLightOff(); t.hide(NO_SHADOW)
    return root, flap


def soft_tex(rgb, peak, power=2.0, name="soft"):
    """64x64 radial-falloff RGBA sprite: blob shadows, smoke puffs, sparks."""
    n = 64; img = PNMImage(n, n, 4)
    for y in range(n):
        for x in range(n):
            r = math.hypot((x + 0.5) / n * 2 - 1, (y + 0.5) / n * 2 - 1)
            img.setXel(x, y, rgb[0], rgb[1], rgb[2]); img.setAlpha(x, y, peak * max(0.0, 1.0 - r) ** power)
    tex = Texture(name); tex.load(img)
    tex.setWrapU(SamplerState.WM_clamp); tex.setWrapV(SamplerState.WM_clamp)
    return tex


def vignette_tex():
    n = 128; img = PNMImage(n, n, 4)
    for y in range(n):
        for x in range(n):
            r = math.hypot((x + 0.5) / n * 2 - 1, (y + 0.5) / n * 2 - 1)
            img.setXel(x, y, 0, 0, 0); img.setAlpha(x, y, 0.5 * clamp((r - 0.6) / 0.8, 0, 1) ** 1.6)
    tex = Texture("vig"); tex.load(img)
    tex.setWrapU(SamplerState.WM_clamp); tex.setWrapV(SamplerState.WM_clamp)
    return tex


class Particles:
    """Pooled billboard sprites (tyre smoke, grass dust, collision sparks)."""
    def __init__(s, parent, tex, n=360):
        s.root = parent.attachNewNode("fx")
        s.root.setLightOff(); s.root.setShaderOff(); s.root.setDepthWrite(False)
        s.root.setTransparency(TransparencyAttrib.MAlpha); s.root.setTexture(tex)
        cm = CardMaker("p"); cm.setFrame(-0.5, 0.5, -0.5, 0.5)
        s.free, s.live = [], []
        for _ in range(n):
            p = s.root.attachNewNode(cm.generate()); p.setBillboardPointEye(); p.hide(); s.free.append(p)

    def emit(s, pos, vel, life, s0, s1, col, a0=0.5, drag=1.2, grav=0.0):
        if not s.free: return
        p = s.free.pop(); p.setPos(pos); p.show()
        s.live.append([p, Vec3(vel), life, 0.0, s0, s1, col, a0, drag, grav])

    def update(s, dt):
        keep = []
        for q in s.live:
            p, v, life, age, s0, s1, col, a0, drag, grav = q
            age += dt
            if age >= life: p.hide(); s.free.append(p); continue
            q[3] = age
            v *= max(0.0, 1.0 - drag * dt); v.z += grav * dt
            p.setPos(p.getPos() + v * dt)
            f = age / life
            p.setScale(s0 + (s1 - s0) * f)
            p.setColorScale(col[0], col[1], col[2], a0 * (1 - f) ** 1.2)
            keep.append(q)
        s.live = keep

    def clear(s):
        for q in s.live: q[0].hide(); s.free.append(q[0])
        s.live = []


# ------------------------------------------------------------------ track
class Track:
    S = 1.5

    def __init__(s):
        raw = [(-350, 0), (-175, 0), (0, 0), (175, 0), (350, 0),                       # main straight
               (430, 0), (474, -18), (492, -62), (474, -106), (430, -124),               # turn 1 hairpin
               (350, -124), (250, -124), (190, -124), (155, -134), (125, -114), (95, -124),  # chicane
               (30, -124), (-60, -124), (-150, -124), (-190, -134), (-230, -114), (-270, -124),  # esses
               (-350, -124), (-394, -106), (-412, -62), (-394, -18)]                     # last hairpin
        P = [(x * s.S, y * s.S) for x, y in raw]
        n = len(P); dense = []
        for i in range(n):
            p0, p1, p2, p3 = P[i - 1], P[i], P[(i + 1) % n], P[(i + 2) % n]
            for k in range(40):
                t = k / 40.0; t2 = t * t; t3 = t2 * t
                dense.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t +
                                (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 +
                                (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in (0, 1)))
        cum = [0.0]
        for k in range(1, len(dense) + 1):
            a, b = dense[k - 1], dense[k % len(dense)]
            cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        s.L = cum[-1]; s.N = int(s.L / 2.5); s.ds = s.L / s.N
        s.x, s.y = [], []; k = 0
        for i in range(s.N):
            tg = i * s.ds
            while cum[k + 1] < tg: k += 1
            seg = cum[k + 1] - cum[k]; f = (tg - cum[k]) / seg if seg > 0 else 0
            a, b = dense[k], dense[(k + 1) % len(dense)]
            s.x.append(a[0] + (b[0] - a[0]) * f); s.y.append(a[1] + (b[1] - a[1]) * f)
        N = s.N
        s.th, s.tx, s.ty, s.nx, s.ny = [], [], [], [], []
        for i in range(N):
            dx = s.x[(i + 1) % N] - s.x[i - 1]; dy = s.y[(i + 1) % N] - s.y[i - 1]
            l = math.hypot(dx, dy)
            s.tx.append(dx / l); s.ty.append(dy / l); s.th.append(math.atan2(dy, dx))
            s.nx.append(-dy / l); s.ny.append(dx / l)
        raw_k = [ang_wrap(s.th[(i + 1) % N] - s.th[i - 1]) / (2 * s.ds) for i in range(N)]
        s.k = [sum(raw_k[(i + j) % N] for j in range(-4, 5)) / 9.0 for i in range(N)]
        s.vc = []
        for kk in map(abs, s.k):
            v = math.sqrt(GRIP0 / (kk - GRIPD)) if kk > GRIPD + 0.0002 else 130.0
            s.vc.append(max(min(v, 130.0), 22.0) * 0.94)
        s.i0 = s.nearest(0, 0)
        # numbered corners
        segs = []; cur = None
        for o in range(N + 1):
            i = (s.i0 + o) % N
            if o < N and abs(s.k[i]) > 0.0035:
                if cur is None: cur = [o, o]
                else: cur[1] = o
            elif cur is not None:
                if segs and cur[0] - segs[-1][1] < 25: segs[-1][1] = cur[1]
                else: segs.append(cur)
                cur = None
        s.turn_of = [0] * N
        for n_, (a, b) in enumerate(segs, 1):
            for o in range(a - 8, b + 9): s.turn_of[(s.i0 + o) % N] = n_
        S = s.S
        def at(x, y): return s.nearest(x * S, y * S)
        s.drs = [(at(-300, 0), at(-250, 0), at(330, 0)), (at(50, -124), at(0, -124), at(-140, -124))]
        s.cams = []
        step = int(320 / s.ds)
        for c, i in enumerate(range(5, N, step)):
            side = (-1 if s.k[i] > 0 else 1) if abs(s.k[i]) > 0.003 else (1 if c % 2 else -1)
            d = side * (HALF_W + 24)
            s.cams.append(Vec3(s.x[i] + s.nx[i] * d, s.y[i] + s.ny[i] * d, 6.0))

    def nearest(s, x, y, hint=None):
        best, bi = 1e18, 0
        rng = range(s.N) if hint is None else ((hint + o) % s.N for o in range(-70, 71))
        for i in rng:
            dx, dy = x - s.x[i], y - s.y[i]
            d = dx * dx + dy * dy
            if d < best: best, bi = d, i
        return bi

    def rel(s, i): return ((i - s.i0) * s.ds + s.L / 2) % s.L - s.L / 2
    def idx_at(s, sr): return (s.i0 + int(round(sr / s.ds))) % s.N

    def is_grass(s, d, sr):
        if abs(d) <= HALF_W + 1.2: return False
        w = pitw(sr)
        return not (w > 0.3 and abs(d - PIT_C * w) <= 5.2 * w)

    def where(s, i):
        if s.turn_of[i]: return "Turn %d" % s.turn_of[i]
        return "Main straight" if abs(s.rel(i)) < 520 else "Back straight" if s.y[i] < -s.S * 100 else "Straight"


# ------------------------------------------------------------------ incident
class Incident:
    def __init__(s, n, t, kind, cars, fault, ruling, lap, loc, ev, why):
        s.n, s.t, s.kind, s.cars, s.fault, s.ruling = n, t, kind, cars, fault, ruling
        s.lap, s.loc, s.ev, s.why, s.status = lap, loc, ev, why, "open"


# ------------------------------------------------------------------ car
class Car:
    def __init__(s, g, i, drv, slot):
        s.g, s.i = g, i
        s.num, s.code, s.col = drv
        tr = g.track
        idx = (tr.i0 - int((9 + slot * 8.5) / tr.ds)) % tr.N
        lat = 2.3 if slot % 2 == 0 else -2.3
        x, y = tr.x[idx] + tr.nx[idx] * lat, tr.y[idx] + tr.ny[idx] * lat
        s.node = BulletRigidBodyNode("car%d" % s.num)
        s.node.setMass(MASS)
        s.node.addShape(BulletBoxShape(Vec3(1.0, 2.7, 0.35)))
        s.node.setLinearFactor(Vec3(1, 1, 0)); s.node.setAngularFactor(Vec3(0, 0, 1))
        s.node.setRestitution(0.28); s.node.setFriction(0.15)
        s.node.setDeactivationEnabled(False)
        s.np = g.render.attachNewNode(s.node)
        s.np.setPos(x, y, 0.4); s.np.setH(math.degrees(tr.th[idx]) - 90)
        g.world.attachRigidBody(s.node)
        helmet = (random.random(), random.random(), random.random(), 1)
        s.vis, s.flap = build_car(g.render, s.col, helmet, s.num)
        s.vis.setPos(x, y, 0.0); s.vis.setH(math.degrees(tr.th[idx]) - 90)
        cm = CardMaker("blob"); cm.setFrame(-1.9, 1.9, -3.5, 3.5)       # soft contact shadow under the car
        blob = s.vis.attachNewNode(cm.generate()); blob.setP(-90); blob.setZ(0.075)
        blob.setTexture(g.blob_tex); blob.setTransparency(TransparencyAttrib.MAlpha)
        blob.setLightOff(); blob.setShaderOff(); blob.setDepthWrite(False)
        s.vl = 0.0
        s.pace = random.uniform(0.955, 1.0); s.aggr = random.uniform(0.1, CFG['aggr'])
        s.err = random.uniform(0.2, 1.0); s.line_off = random.uniform(-1.2, 1.2)
        s.pit_lap = random.choice([2, 3]); s.pit_state = None; s.pitted = False
        r, p = random.random(), CFG['pit']
        s.pit_target = (PIT_LIMIT - 1.5 if r >= p else PIT_LIMIT + random.uniform(0.3, 1.1)
                        if r >= p * 0.4 else PIT_LIMIT + random.uniform(2.0, 5.5))
        s.box_sr = -150.0 + [d[0] for d in DRIVERS].index(s.num) * 33.0; s.stop_t = 0.0
        s.pass_t = 0.0; s.pass_dir = 1; s.mistake_t = 0.0; s.mistake_dir = 1; s.dive_t = -5.0
        s.idx = idx; s.pidx = idx; s.d = 0.0; s.prog = 0.0; s.s_last = 0.0
        s.v = Vec3(0, 0, 0); s.pv = Vec3(0, 0, 0); s.vf = 0.0; s.spd = 0.0
        s.thr = 0.0; s.brk = 0.0; s.glat = 0.0; s.glon = 0.0
        s.laps_done = 0; s.started = False; s.lap_t0 = 0.0; s.last_lap = None; s.best_lap = None
        s.finished = False; s.finish_time = None
        s.loop_key = -1; s.times = {}
        s.drs_open = False; s.elig = [False, False]; s.det_t = [-99.0, -99.0]
        s.tl = False; s.tl_t0 = 0.0; s.tl_max = 0.0; s.strikes = 0
        s.in_zone = False; s.zone_max = 0.0
        s.last_hit_t = -99.0; s.pen_s = 0; s.warnings = 0
        s.trace = deque(maxlen=150); s.flap_a = 0.0
        s.update_track(True)

    # ---- track-relative position
    def update_track(s, first=False):
        tr = s.g.track; p = s.np.getPos()
        s.pidx = s.idx
        s.idx = tr.nearest(p.x, p.y, None if first else s.idx)
        i = s.idx; dx, dy = p.x - tr.x[i], p.y - tr.y[i]
        s.d = dx * tr.nx[i] + dy * tr.ny[i]
        cur = i * tr.ds + dx * tr.tx[i] + dy * tr.ty[i]
        if first: s.prog = tr.rel(i) + dx * tr.tx[i] + dy * tr.ty[i]
        else: s.prog += wrap(cur - s.s_last, tr.L)
        s.s_last = cur

    # ---- AI driver + tyre/aero force model (called every frame before physics)
    def control(s, dt, hold):
        g = s.g; tr = g.track; node = s.node; N = tr.N; ds = tr.ds
        vel = node.getLinearVelocity()
        q = s.np.getQuat(g.render); fwd = q.getForward(); right = q.getRight()
        vf = vel.dot(fwd); vl = vel.dot(right); spd = vel.length()
        s.vf, s.spd, s.vl = vf, spd, vl
        if hold:
            node.setLinearVelocity(Vec3(0, 0, 0)); node.setAngularVelocity(Vec3(0, 0, 0))
            s.thr, s.brk = 0.0, 1.0
            return
        idx = s.idx; sr = tr.rel(idx); p = s.np.getPos()
        ahead, agap, alongside = None, 1e9, []
        for o in g.cars:
            if o is s: continue
            dp = wrap(o.prog - s.prog, tr.L); dd = o.d - s.d
            if 0 < dp < 150 and abs(dd) < 3.3 and dp < agap: ahead, agap = o, dp
            if abs(dp) < 7.0 and abs(dd) < 3.4: alongside.append(dd)
        if (not s.pitted and s.pit_state is None and not s.finished
                and s.laps_done == s.pit_lap - 1 and -430 <= sr < -300):
            s.pit_state = "in"
        pit = s.pit_state
        # ---------------- lateral target
        lat = s.line_off; mistake = False
        if pit:
            w = pitw(sr)
            work = pit == "stopped" or (pit == "in" and 0 <= s.box_sr - sr < 24)
            lat = (PIT_WORK if work else PIT_FAST) * w
        else:
            if ahead and agap < 45 and not (g.t < 20 and s.aggr < 0.8):
                if s.pass_t <= 0:
                    side = -1 if ahead.d > s.d else 1
                    if side > 0 and s.d > HALF_W - 4: side = -1
                    if side < 0 and s.d < -(HALF_W - 4): side = 1
                    s.pass_dir = side
                s.pass_t = 2.5
                lat = ahead.d + s.pass_dir * 3.4
            else:
                s.pass_t -= dt
            for dd in alongside:
                lat += (-1 if dd > 0 else 1) * (3.3 - abs(dd)) * (1.0 - 0.9 * s.aggr)
            if s.mistake_t > 0:
                s.mistake_t -= dt; mistake = True; lat = s.mistake_dir * (HALF_W + 2.6)
            elif spd > 40 and not s.finished and random.random() < CFG['mistake'] * s.err * dt:
                s.mistake_t = 2.2
                kk = tr.k[(idx + int(70 / ds)) % N]
                s.mistake_dir = (-1 if kk > 0 else 1) if abs(kk) > 0.002 else random.choice([-1, 1])
            if not mistake: lat = clamp(lat, -(HALF_W - 1.2), HALF_W - 1.2)
        # ---------------- speed target (look-ahead braking)
        s.dive_t -= dt
        if (ahead and agap < 30 and s.dive_t < -3 and s.aggr > 0.45 and
                abs(tr.k[(idx + int(120 / ds)) % N]) > 0.006 and random.random() < CFG['dive'] * dt * s.aggr):
            s.dive_t = 3.0
        f = s.pace * (1.10 if s.dive_t > 0 else 1.0)
        v_t = 125.0
        for m_ in range(0, 84, 3):
            v_t = min(v_t, math.sqrt((tr.vc[(idx + m_) % N] * f) ** 2 + 2 * 36.0 * m_ * ds))
        if s.finished: v_t = min(v_t, 30.0)
        if not pit and abs(s.d) > HALF_W + 2.5: v_t = min(v_t, 22.0 if abs(s.d) < 30 else 14.0)
        if ahead and abs(ahead.d - s.d) < 2.6 and not (s.aggr > 0.85 and agap > 6):
            # time-gap following: hold ~0.45 s behind the car in front, close only as fast as the gap allows
            des = 7.0 + 0.45 * max(vf, 0.0)
            v_follow = ahead.vf + max(agap - des, 0.0) * 0.7 - (3.0 if agap < des * 0.6 else 0.0)
            v_t = min(v_t, max(v_follow, 4.0))
        if pit:
            vlim = s.pit_target
            if sr < -PIT_ZONE: v_t = min(v_t, math.sqrt(max(vlim - 1.0, 5.0) ** 2 + 2 * 22 * max(-PIT_ZONE - sr - 12, 0)))
            elif sr <= PIT_ZONE + 5: v_t = min(v_t, vlim)
            if pit == "in":
                dist = s.box_sr - sr
                if dist < 80: v_t = min(v_t, math.sqrt(2 * 6.0 * max(dist, 0)) + 0.3)
                if dist < -15: s.pit_state = "out"
                if dist < 1.0 and spd < 1.5:
                    s.pit_state = "stopped"; s.stop_t = 2.8 + random.random()
            elif pit == "stopped":
                v_t = 0.0; s.stop_t -= dt
                if s.stop_t <= 0: s.pit_state = "out"
            elif pit == "out" and sr > 262:
                s.pit_state = None; s.pitted = True
        # ---------------- steering (pure pursuit) + yaw response
        Ld = 9 + 0.40 * max(vf, 0)
        j = (idx + int(Ld / ds)) % N
        tx, ty = tr.x[j] + tr.nx[j] * lat, tr.y[j] + tr.ny[j] * lat
        tvx, tvy = tx - p.x, ty - p.y
        alpha = math.atan2(fwd.x * tvy - fwd.y * tvx, fwd.x * tvx + fwd.y * tvy)
        grass = tr.is_grass(s.d, sr)
        grip = 0.55 if grass else 1.0
        am = alat_max(spd) * grip
        hd = math.atan2(fwd.y, fwd.x)
        recover = abs(ang_wrap(hd - tr.th[idx])) > 1.3 and spd < 20 and not pit
        if recover:
            yaw_t = clamp(alpha * 2.0, -1.6, 1.6); auth = 1.0; v_t = 10.0
        else:
            vv = max(vf, 0.0)
            yaw_t = clamp(vv * 2 * math.sin(alpha) / max(math.hypot(tvx, tvy), 3.0),
                          -am / max(vv, 8.0), am / max(vv, 8.0))
            slip = math.atan2(abs(vl), max(abs(vf), 1.0))
            auth = clamp(1.2 - slip * 2.2, 0.08, 1.0)
        w = node.getAngularVelocity().z
        w += (yaw_t - w) * min(1.0, 9.0 * auth * dt)
        node.setAngularVelocity(Vec3(0, 0, w))
        # ---------------- throttle / brake
        err = v_t - vf
        if err > 0: s.thr, s.brk = clamp(err / 4.0, 0, 1), 0.0
        else: s.thr, s.brk = 0.0, clamp(-err / (1.5 if pit else 3.0), 0, 1)
        # ---------------- forces
        tow = bool(ahead and agap < 30 and abs(ahead.d - s.d) < 2.5)
        cda = 1.3 * (0.72 if s.drs_open else 1.0) * (0.86 if tow else 1.0)
        drag = 0.5 * 1.2 * cda * spd * spd
        roll = MASS * G * 0.015 + (MASS * 5.0 if grass else 0.0)
        fe = min(600e3 / max(vf, 4.0), MASS * (16 + 0.003 * max(vf, 0) ** 2) * grip) * s.thr
        fb = MASS * (30 + 0.003 * spd * spd) * (0.6 if grass else 1.0) * s.brk
        F = fwd * fe
        if spd > 0.05:
            vd = vel / spd
            F -= vd * (drag + roll)
            F -= vd * min(fb, MASS * spd / max(dt, 1 / 240.0))
        a_need = -vl / max(dt, 1 / 120.0)
        a_lat = clamp(a_need, -am, am)
        F += right * (MASS * a_lat)
        node.applyCentralForce(F)
        s.glat = a_lat / G; s.glon = (fe - fb - drag) / (MASS * G)

    def post(s, dt):
        p = s.np.getPos(); s.vis.setPos(p.x, p.y, 0.0); s.vis.setH(s.np.getH())
        s.v = Vec3(s.node.getLinearVelocity())
        s.update_track()

    # ---- rule monitoring (laps, timing loops, DRS, track limits, pit speed)
    def monitor(s, dt):
        g = s.g; tr = g.track; t = g.t; N = tr.N
        if not s.started and s.prog >= 0: s.started, s.lap_t0 = True, t
        while not s.finished and s.prog >= (s.laps_done + 1) * tr.L:
            s.laps_done += 1; lt = t - s.lap_t0; s.last_lap, s.lap_t0 = lt, t
            if s.best_lap is None or lt < s.best_lap: s.best_lap = lt
            if s.laps_done == 1 and g.order and g.order[0] is s:
                g.banner("DRS ENABLED", 3, (0.3, 1, 0.4, 1)); g.say("DRS ENABLED", (0.3, 1, 0.4, 1))
            if s.laps_done >= LAPS:
                s.finished, s.finish_time = True, t; g.on_finish(s)
        k = int(s.prog // LOOP) if s.prog >= 0 else -1
        while s.loop_key < k:
            s.loop_key += 1; s.times[s.loop_key] = t
        if s.finished: return
        # DRS: detect <1.0s behind at detection line -> may open in the activation zone
        want = False
        for z, (det, a0, a1) in enumerate(tr.drs):
            if crossed(s.pidx, s.idx, det, N):
                s.det_t[z] = t; s.elig[z] = False
                pos = g.order.index(s)
                if pos > 0 and s.laps_done >= 1 and s.pit_state is None:
                    o = g.order[pos - 1]; gap = t - o.det_t[z]
                    if 0 < gap <= 1.0 and o.pit_state is None: s.elig[z] = True
            inz = (s.idx - a0) % N <= (a1 - a0) % N
            if s.elig[z] and inz: want = True
            if s.elig[z] and not inz and (s.idx - a1) % N < 60: s.elig[z] = False
        s.drs_open = want and s.brk < 0.2 and s.pit_state is None
        # track limits: all four wheels beyond the white line
        if not s.pit_state:
            q = s.np.getQuat(g.render); f = q.getForward(); r = q.getRight()
            nx, ny = tr.nx[s.idx], tr.ny[s.idx]
            ws = [s.d + (r.x * sx + f.x * sy) * nx + (r.y * sx + f.y * sy) * ny
                  for sx, sy in ((-1, 1.7), (1, 1.7), (-1, -1.6), (1, -1.6))]
            lo, hi = min(ws), max(ws)
            beyond = lo - HALF_W if lo > HALF_W else (-HALF_W - hi if hi < -HALF_W else 0.0)
            if beyond > 0:
                if not s.tl: s.tl, s.tl_t0, s.tl_max, s.tl_loc = True, t, 0.0, tr.where(s.idx)
                s.tl_max = max(s.tl_max, beyond)
            elif s.tl:
                s.tl = False
                if t - s.tl_t0 > 0.5 and s.tl_max >= 1.5: g.track_limit_incident(s, t - s.tl_t0, s.tl_max)
        else:
            s.tl = False
        # pit lane speed
        if s.pit_state and abs(tr.rel(s.idx)) <= PIT_ZONE:
            if not s.in_zone: s.in_zone, s.zone_max = True, 0.0
            s.zone_max = max(s.zone_max, s.spd * 3.6)
        elif s.in_zone:
            s.in_zone = False; g.pit_incident(s, s.zone_max)



# ------------------------------------------------------------------ game
class Game(ShowBase):
    # =====================================================================
    #  setup
    # =====================================================================
    def __init__(s):
        ShowBase.__init__(s)
        s.disableMouse()
        s.setBackgroundColor(0.55, 0.72, 0.93)
        s.camLens.setNearFar(0.8, 4000); s.camLens.setFov(72)
        s.render.setAntialias(AntialiasAttrib.MAuto)
        try: s.mono = s.loader.loadFont("cmtt12")
        except Exception: s.mono = None
        s.track = Track()
        s.world = BulletWorld(); s.world.setGravity(Vec3(0, 0, 0))
        s.setup_lights(); s.setup_fx(); s.build_scene(); s.build_sky()
        s.cars, s.dots = [], []
        s.laps_opt, s.laps_i = [3, 4, 6, 8, 10], 2
        s.levels, s.lvl_i = list(LEVELS.keys()), 0
        s.state, s.first_run, s.pmenu = "menu", True, False
        s.buttons, s.menu_a, s.pin_cam, s.pin_pt = [], 0.0, None, None
        s.setup_ui(); s.setup_map(); s.setup_panels(); s.setup_menu(); s.setup_input()
        s.spawn_cars(); s.show_state(); s.setup_post()
        s.taskMgr.add(s.update, "update")

    def spawn_cars(s):
        for c in s.cars:
            s.world.removeRigidBody(c.node); c.np.removeNode(); c.vis.removeNode()
        CFG.update(LEVELS[s.levels[s.lvl_i]]); set_laps(s.laps_opt[s.laps_i])
        drv = DRIVERS[:]; random.shuffle(drv)
        s.cars = [Car(s, i, d, i) for i, d in enumerate(drv)]
        s.order = s.cars[:]
        s.t, s.grid_t, s.paused, s.pmenu = 0.0, 7.0, False, False
        s.fx.clear()
        s.replay, s.panel = None, False
        s.sel_inc = s.sel_car = s.sel_pen = 0
        s.score = s.reports = s.good = 0
        s.cam_mode, s.follow, s.cam_spot = 1, 0, None
        s.hist_t, s.hist, s.hist_acc = [], [], 0.0
        s.incidents, s.inc_n, s.contact, s.last_contact = [], 0, {}, {}
        s.log, s.banner_t, s.first_fin = [], 0.0, None
        s.banner_txt.setText(""); s.rep.hide()
        for ch in s.res_content.getChildren(): ch.removeNode()
        for c, d in zip(s.cars, s.dots): d.setColor(c.col[0], c.col[1], c.col[2], 1)

    def show_state(s):
        menu = s.state == "menu"
        for r in s.hud_roots: (r.hide if menu else r.show)()
        (s.menu.show if menu else s.menu.hide)()
        (s.res.show if s.state == "over" else s.res.hide)()
        (s.pause_ov.show if s.pmenu else s.pause_ov.hide)()

    # =====================================================================
    #  buttons / mouse
    # =====================================================================
    def button(s, parent, x0, z0, x1, z1, label, fn, scale=0.034, col=(0.10, 0.14, 0.28, 0.92),
               fg=(1, 1, 1, 1), xy=False, mono=False, left=False, hover=True):
        root = parent.attachNewNode("btn")
        bg = s.rect(root, x0, z0, x1, z1, col)
        txt = None
        if label is not None:
            zc = (z0 + z1) / 2 - scale * 0.33
            if left: txt = s.T(root, (x0 + 0.015, zc), scale, fg, TextNode.ALeft, mono, label)
            else: txt = s.T(root, ((x0 + x1) / 2, zc), scale, fg, TextNode.ACenter, mono, label)
        b = dict(root=root, parent=parent, bg=bg, r=(x0, z0, x1, z1), fn=fn, col=col, txt=txt,
                 xy=xy, hover=hover and fn is not None)
        s.buttons.append(b)
        return b

    def mouse_a2d(s):
        if not s.mouseWatcherNode.hasMouse(): return None
        m = s.mouseWatcherNode.getMouse()
        return m.x * s.getAspectRatio(), m.y

    def btn_rect(s, b):
        p = b["parent"].getPos(s.aspect2d); x0, z0, x1, z1 = b["r"]
        return x0 + p.x, z0 + p.z, x1 + p.x, z1 + p.z

    def update_hover(s):
        m = s.mouse_a2d()
        for b in s.buttons:
            c = b["col"]
            if m and b["hover"] and not b["bg"].isHidden():
                x0, z0, x1, z1 = s.btn_rect(b)
                if x0 <= m[0] <= x1 and z0 <= m[1] <= z1:
                    c = (min(1, c[0] + .18), min(1, c[1] + .18), min(1, c[2] + .18), min(1, c[3] + .25))
            b["bg"].setColor(*c)

    def on_click(s):
        m = s.mouse_a2d()
        if not m: return
        for b in reversed(s.buttons):
            if b["bg"].isHidden(): continue
            x0, z0, x1, z1 = s.btn_rect(b)
            if x0 <= m[0] <= x1 and z0 <= m[1] <= z1:
                if b["xy"]: b["fn"](m[0], m[1])
                elif b["fn"]: b["fn"]()
                return
        if s.state in ("grid", "race", "over") and not s.panel and not s.pmenu:
            c = s.pick_car(*m)
            if c: s.follow_car(c)

    def to_a2d(s, pos):
        p3 = s.cam.getRelativePoint(s.render, Point3(pos[0], pos[1], pos[2])); p2 = Point2()
        if s.camLens.project(p3, p2): return p2.x * s.getAspectRatio(), p2.y
        return None

    def pick_car(s, mx, mz):
        best, bd = None, 0.075
        for c in s.cars:
            sp = s.to_a2d(Vec3(c.vis.getX(), c.vis.getY(), 1.0))
            if sp:
                d = math.hypot(sp[0] - mx, sp[1] - mz)
                if d < bd: best, bd = c, d
        return best

    def follow_car(s, c):
        s.follow = c.i; s.cam_spot = None
        if s.cam_mode == 5: s.cam_mode = 1

    def set_cam(s, n):
        if n == 5: s.set_pin(s.cars[s.follow].idx)
        else: s.cam_mode = n; s.cam_spot = None

    def set_pin(s, idx):
        tr = s.track; i = idx % tr.N
        side = (-1 if tr.k[i] > 0 else 1) if abs(tr.k[i]) > 0.003 else 1
        d = side * (HALF_W + 26)
        s.pin_cam = Vec3(tr.x[i] + tr.nx[i] * d, tr.y[i] + tr.ny[i] * d, 7.0)
        s.pin_pt = Vec3(tr.x[i], tr.y[i], 0.5); s.cam_mode = 5

    # =====================================================================
    #  HUD
    # =====================================================================
    def setup_ui(s):
        s.hud_tl = s.a2dTopLeft.attachNewNode("hud_tl"); s.hud_tr = s.a2dTopRight.attachNewNode("hud_tr")
        s.hud_tc = s.a2dTopCenter.attachNewNode("hud_tc"); s.hud_bl = s.a2dBottomLeft.attachNewNode("hud_bl")
        s.hud_br = s.a2dBottomRight.attachNewNode("hud_br")
        s.hud_roots = [s.hud_tl, s.hud_tr, s.hud_tc, s.hud_bl, s.hud_br]
        TL, TR, TC, BL, BR = s.hud_tl, s.hud_tr, s.hud_tc, s.hud_bl, s.hud_br
        # ---- interval timing board (rows are clickable)
        s.rect(TL, 0.01, -0.70, 0.95, -0.005, (0, 0, 0, 0.62))
        s.rect(TL, 0.01, -0.095, 0.95, -0.005, (0.75, 0.05, 0.1, 0.85))
        for r in range(len(DRIVERS)):
            z0 = -0.202 - r * 0.050
            s.button(TL, 0.012, z0, 0.948, z0 + 0.05, None, lambda r=r: s.follow_car(s.order[r]), col=(1, 1, 1, 0.0))
        s.T(TL, (0.03, -0.065), 0.042, text="INTERVAL TIMING")
        s.T(TL, (0.58, -0.062), 0.026, (1, 1, 1, 0.85), text="click a row to spectate")
        s.T(TL, (0.03, -0.14), 0.034, (0.7, 0.8, 1, 1),
            text="%2s %-3s %2s %8s %8s %8s %-3s %4s" % ("P", "DRV", "#", "INT", "GAP", "LAST", "ST", "PEN"))
        s.rows = [s.T(TL, (0.03, -0.19 - r * 0.050), 0.034) for r in range(len(DRIVERS))]
        # ---- top centre: lap counter, camera + action buttons, grid lights, banner
        s.lap_txt = s.T(TC, (0, -0.075), 0.06, align=TextNode.ACenter, mono=False)
        s.cam_btns = []
        for k, name in enumerate(("1 CHASE", "2 HELI", "3 TV", "4 ONBOARD", "5 PINNED")):
            x0 = -0.55 + k * 0.22
            s.cam_btns.append(s.button(TC, x0, -0.215, x0 + 0.21, -0.145, name, lambda n=k + 1: s.set_cam(n), 0.027))
        s.act_btns = []
        for k, (name, fn) in enumerate((("R REPORT", s.k_report), ("V REPLAY", s.k_replay),
                                        ("SPACE PAUSE", s.k_pause), ("H HELP", s.k_help))):
            x0 = -0.55 + k * 0.28
            s.act_btns.append(s.button(TC, x0, -0.295, x0 + 0.27, -0.225, name, fn, 0.027, col=(0.45, 0.08, 0.12, 0.92)))
        s.lights = [s.rect(TC, -0.28 + k * 0.14, -0.43, -0.18 + k * 0.14, -0.33, (0.2, 0, 0, 1)) for k in range(5)]
        s.banner_txt = s.T(TC, (0, -0.58), 0.075, align=TextNode.ACenter, mono=False)
        s.status_txt = s.T(TC, (0, -0.68), 0.045, (1, 0.9, 0.3, 1), TextNode.ACenter, False)
        # ---- score
        s.rect(TR, -0.62, -0.19, -0.01, -0.005, (0, 0, 0, 0.62))
        s.score_txt = s.T(TR, (-0.60, -0.07), 0.045)
        s.score_txt2 = s.T(TR, (-0.60, -0.14), 0.034, (0.8, 0.9, 1, 1))
        # ---- race-control log
        s.rect(BL, 0.01, 0.01, 1.62, 0.39, (0, 0, 0, 0.55))
        s.logs = [s.T(BL, (0.03, 0.035 + k * 0.05), 0.031, mono=False) for k in range(7)]
        s.T(BL, (0.03, 0.41), 0.028, (1, 1, 1, 0.85), mono=False,
            text="Click a car, a timing row or the track map to spectate")
        # ---- telemetry
        s.rect(BR, -0.90, 0.01, -0.01, 0.66, (0, 0, 0, 0.62))
        s.rect(BR, -0.90, 0.60, -0.01, 0.66, (0.1, 0.3, 0.8, 0.85))
        s.tel_h = s.T(BR, (-0.88, 0.615), 0.036, text="TELEMETRY")
        s.tel_spd = s.T(BR, (-0.88, 0.47), 0.11, text="0")
        s.T(BR, (-0.60, 0.48), 0.035, (0.7, 0.8, 1, 1), text="km/h")
        s.tel_gear = s.T(BR, (-0.22, 0.47), 0.11, (1, 0.85, 0.2, 1), TextNode.ACenter, text="N")
        s.T(BR, (-0.88, 0.375), 0.032, (0.8, 1, 0.8, 1), text="THR")
        s.T(BR, (-0.88, 0.315), 0.032, (1, 0.8, 0.8, 1), text="BRK")
        s.bar_t = s.rect(BR, 0, 0, 1, 1, (0.2, 1, 0.3, 1)); s.bar_b = s.rect(BR, 0, 0, 1, 1, (1, 0.2, 0.2, 1))
        s.tel_info = s.T(BR, (-0.88, 0.235), 0.031)
        s.rect(BR, -0.88, 0.03, -0.02, 0.20, (1, 1, 1, 0.08))
        s.T(BR, (-0.87, 0.205), 0.024, (0.8, 0.8, 0.8, 1), text="speed(white) throttle(green) brake(red) - last 5s")
        s.trace_np = None

    def setup_map(s):
        tr, P = s.track, s.hud_br
        s.rect(P, -0.90, 0.69, -0.01, 1.03, (0, 0, 0, 0.62))
        s.rect(P, -0.90, 0.99, -0.01, 1.03, (0.1, 0.3, 0.8, 0.85))
        s.T(P, (-0.88, 1.0), 0.028, text="TRACK MAP - click to spectate")
        s.map_btn = s.button(P, -0.90, 0.69, -0.01, 0.99, None, s.map_click, col=(0, 0, 0, 0.0), xy=True, hover=False)
        minx, maxx, miny, maxy = min(tr.x), max(tr.x), min(tr.y), max(tr.y)
        s.map_c = ((minx + maxx) / 2, (miny + maxy) / 2)
        s.map_sc = min(0.82 / (maxx - minx), 0.23 / (maxy - miny))
        def line(idxs, thick, col):
            ls = LineSegs(); ls.setThickness(thick); ls.setColor(*col)
            for k, i in enumerate(idxs):
                x, z = s.map_xy(tr.x[i % tr.N], tr.y[i % tr.N])
                (ls.moveTo if k == 0 else ls.drawTo)(x, 0, z)
            P.attachNewNode(ls.create())
        allidx = list(range(0, tr.N, 3)) + [0]
        line(allidx, 8, (0.12, 0.12, 0.16, 1)); line(allidx, 4, (0.85, 0.85, 0.9, 1))
        for det, a0, a1 in tr.drs:
            n = (a1 - a0) % tr.N; line([a0 + k for k in range(0, n, 3)] + [a1], 4, (0.1, 0.9, 0.2, 1))
        pit = [tr.i0 + o for o in range(int(-190 / tr.ds), int(190 / tr.ds), 3)]
        ls = LineSegs(); ls.setThickness(2); ls.setColor(1, 0.85, 0.1, 1)
        for k, i in enumerate(pit):
            i %= tr.N; w = pitw(tr.rel(i)); d = PIT_C * w
            x, z = s.map_xy(tr.x[i] + tr.nx[i] * d, tr.y[i] + tr.ny[i] * d)
            (ls.moveTo if k == 0 else ls.drawTo)(x, 0, z)
        P.attachNewNode(ls.create())
        x, z = s.map_xy(tr.x[tr.i0], tr.y[tr.i0])
        ls = LineSegs(); ls.setThickness(3); ls.setColor(1, 0.1, 0.1, 1)
        ls.moveTo(x, 0, z - 0.025); ls.drawTo(x, 0, z + 0.025); P.attachNewNode(ls.create())
        for n in range(1, max(tr.turn_of) + 1):
            ids = [i for i in range(tr.N) if tr.turn_of[i] == n]
            i = ids[len(ids) // 2]; x, z = s.map_xy(tr.x[i], tr.y[i])
            s.T(P, (x, z + (0.034 if tr.y[i] > s.map_c[1] else -0.05)), 0.024, (1, 0.9, 0.3, 1),
                TextNode.ACenter, False, "T%d" % n)
        s.ring = s.rect(P, -1, -1, 1, 1, (1, 1, 1, 1)); s.ring.setScale(0.019)
        s.dots = []
        for _ in DRIVERS:
            d = s.rect(P, -1, -1, 1, 1, (1, 0, 0, 1)); d.setScale(0.011); s.dots.append(d)

    def map_xy(s, wx, wy):
        return -0.455 + (wx - s.map_c[0]) * s.map_sc, 0.84 + (wy - s.map_c[1]) * s.map_sc

    def map_click(s, mx, mz):
        p = s.hud_br.getPos(s.aspect2d); lx, lz = mx - p.x, mz - p.z
        best, bd = None, 0.03
        for c in s.cars:
            x, z = s.map_xy(c.vis.getX(), c.vis.getY()); d = math.hypot(x - lx, z - lz)
            if d < bd: best, bd = c, d
        if best: s.follow_car(best); return
        tr = s.track; bi, bd = None, 0.06
        for i in range(0, tr.N, 3):
            x, z = s.map_xy(tr.x[i], tr.y[i]); d = math.hypot(x - lx, z - lz)
            if d < bd: bi, bd = i, d
        if bi is not None: s.set_pin(bi)

    # =====================================================================
    #  panels: report form, pause, results, help, main menu
    # =====================================================================
    def setup_panels(s):
        A = s.aspect2d; SEL, BASE = (0.18, 0.40, 0.85, 0.95), (0.10, 0.14, 0.28, 0.92)
        s.SEL, s.BASE = SEL, BASE
        # ---- report form
        s.rep = A.attachNewNode("rep"); s.rep.hide()
        s.button(s.rep, -1.30, -0.80, 1.30, 0.80, None, None, col=(0.02, 0.03, 0.08, 0.96), hover=False)
        s.rect(s.rep, -1.30, 0.70, 1.30, 0.80, (0.75, 0.05, 0.1, 1))
        s.T(s.rep, (-1.27, 0.725), 0.05, text="STEWARD REPORT FORM - FIA RACE CONTROL")
        s.T(s.rep, (-1.27, 0.665), 0.030, (0.7, 0.8, 1, 1),
            text="   ID  LAP TYPE          CARS               LOCATION        AGE    (click a row)")
        s.inc_btns = [s.button(s.rep, -1.27, 0.585 - r * 0.062, 1.27, 0.641 - r * 0.062, "", lambda r=r: s.sel_row(r),
                               0.032, mono=True, left=True) for r in range(7)]
        s.rep_ev = s.T(s.rep, (-1.27, 0.10), 0.033, (0.8, 1, 0.8, 1))
        s.lbl_drv = s.T(s.rep, (-1.27, -0.07), 0.034, (1, 0.9, 0.3, 1), mono=False, text="DRIVER TO PENALISE")
        s.drv_btns = [s.button(s.rep, -1.27 + k * 0.75, -0.16, -0.57 + k * 0.75, -0.09, "", lambda k=k: s.sel_driver(k), 0.036)
                      for k in range(2)]
        s.lbl_pen = s.T(s.rep, (-1.27, -0.225), 0.034, (1, 0.9, 0.3, 1), mono=False, text="PENALTY")
        s.pen_btns = [s.button(s.rep, -1.27, -0.305 - r * 0.075, 0.05, -0.239 - r * 0.075,
                               "%d   %s" % (r + 1, PENALTIES[r][0]), lambda r=r: s.sel_penalty(r), 0.034, left=True)
                      for r in range(5)]
        s.sub_btn = s.button(s.rep, 0.30, -0.42, 1.27, -0.28, "SUBMIT REPORT  (ENTER)", s.submit, 0.040, col=(0.10, 0.45, 0.15, 0.95))
        s.rep_replay = s.button(s.rep, 0.30, -0.55, 1.27, -0.45, "WATCH REPLAY  (V)", s.k_replay, 0.034)
        s.button(s.rep, 0.30, -0.68, 1.27, -0.58, "CLOSE  (ESC)", s.close_report, 0.034, col=(0.45, 0.08, 0.12, 0.92))
        s.T(s.rep, (-1.27, -0.77), 0.028, (0.7, 0.8, 1, 1),
            text="Keys: UP/DOWN incident  LEFT/RIGHT driver  1-5 penalty  ENTER submit")
        # ---- pause overlay
        s.pause_ov = A.attachNewNode("pause"); s.pause_ov.hide()
        s.button(s.pause_ov, -4, -1.1, 4, 1.1, None, None, col=(0, 0, 0, 0.55), hover=False)
        s.T(s.pause_ov, (0, 0.30), 0.12, (1, 1, 1, 1), TextNode.ACenter, False, "PAUSED")
        s.button(s.pause_ov, -0.45, 0.02, 0.45, 0.16, "RESUME", s.resume, 0.05)
        s.button(s.pause_ov, -0.45, -0.18, 0.45, -0.04, "QUIT TO MAIN MENU", s.to_menu, 0.045, col=(0.45, 0.08, 0.12, 0.92))
        # ---- results
        s.res = A.attachNewNode("res"); s.res.hide()
        s.button(s.res, -1.25, -0.9, 1.25, 0.9, None, None, col=(0, 0, 0, 0.92), hover=False)
        s.res_content = s.res.attachNewNode("rc")
        s.button(s.res, -1.15, -0.86, -0.05, -0.74, "RACE AGAIN", s.start_race, 0.045, col=(0.10, 0.45, 0.15, 0.95))
        s.button(s.res, 0.05, -0.86, 1.15, -0.74, "MAIN MENU", s.to_menu, 0.045)

    def setup_menu(s):
        M = s.menu = s.aspect2d.attachNewNode("menu")
        s.button(M, -4, -1.1, 4, 1.1, None, None, col=(0.01, 0.02, 0.06, 0.45), hover=False)
        s.rect(M, -4, 0.50, 4, 0.78, (0, 0, 0, 0.55)); s.rect(M, -4, 0.485, 4, 0.50, (0.85, 0.05, 0.1, 1))
        s.T(M, (0, 0.575), 0.15, (1, 1, 1, 1), TextNode.ACenter, False, "FIA STEWARD")
        s.T(M, (0, 0.445), 0.05, (1, 0.45, 0.45, 1), TextNode.ACenter, False, "RACE CONTROL SIMULATOR")
        s.button(M, -0.55, 0.20, 0.55, 0.34, "START RACE", s.start_race, 0.06, col=(0.10, 0.45, 0.15, 0.95))
        s.laps_btn = s.button(M, -0.55, 0.02, 0.55, 0.15, "", s.cycle_laps, 0.040)
        s.lvl_btn = s.button(M, -0.55, -0.16, 0.55, -0.03, "", s.cycle_level, 0.040)
        s.button(M, -0.55, -0.34, 0.55, -0.21, "HOW TO PLAY", s.k_help, 0.040)
        s.button(M, -0.55, -0.52, 0.55, -0.39, "QUIT", s.userExit, 0.040, col=(0.45, 0.08, 0.12, 0.92))
        s.lvl_desc = s.T(M, (0, -0.64), 0.034, (0.85, 0.9, 1, 1), TextNode.ACenter, False)
        s.T(M, (0, -0.90), 0.030, (0.7, 0.75, 0.9, 0.9), TextNode.ACenter, False,
            text="Click a button or press ENTER to start. You are the steward - the drivers are on their own.")
        s.refresh_menu()
        # ---- help overlay (topmost: swallows the next click)
        s.help = s.aspect2d.attachNewNode("help"); s.help.hide()
        s.button(s.help, -1.35, -0.75, 1.35, 0.75, None, s.k_help, col=(0, 0, 0, 0.90), hover=False)
        s.T(s.help, (-1.30, 0.64), 0.05, (1, 0.85, 0.2, 1), text="HOW TO STEWARD")
        s.T(s.help, (-1.30, 0.55), 0.031, text=(
            "The cars race on their own. Race Control NOTES incidents; you decide the penalties.\n\n"
            "COLLISION    : who caused it? Rear contact, or a car moving into another. Judge speed & movement.\n"
            "TRACK LIMITS : all 4 wheels beyond the white line. Strike 3 = warning, strike 4+ = 5s.\n"
            "               No penalty if the car was forced off by contact.\n"
            "PIT SPEEDING : limit is 80 km/h between the yellow lines. Tiny excess = warning, more = time penalty.\n"
            "DRS          : within 1.0s at the orange detection line -> flap opens in the green zone.\n\n"
            "MOUSE   click a car, a timing-board row, or the track map to spectate. Click the map's track to\n"
            "        pin a trackside camera at that spot. All on-screen buttons are clickable.\n"
            "KEYS    R report form | V replay (0.5x) | I jump to latest incident | 1-5 cameras | TAB next car\n"
            "        SPACE pause | ESC menu / close | H help\n\n"
            "SCORE   Match the stewards' verdict for points. Wrong driver or too harsh a penalty loses points.\n\n"
            "                                        (click anywhere to close)"))

    def refresh_menu(s):
        s.laps_btn["txt"].setText("RACE LENGTH:  %d LAPS" % s.laps_opt[s.laps_i])
        lv = s.levels[s.lvl_i]
        s.lvl_btn["txt"].setText("INCIDENT FREQUENCY:  %s" % lv)
        s.lvl_desc.setText(LEVELS[lv]["desc"])

    def cycle_laps(s): s.laps_i = (s.laps_i + 1) % len(s.laps_opt); s.refresh_menu()
    def cycle_level(s): s.lvl_i = (s.lvl_i + 1) % len(s.levels); s.refresh_menu()

    # =====================================================================
    #  flow: menu <-> race
    # =====================================================================
    def start_race(s):
        s.spawn_cars(); s.state = "grid"; s.show_state()
        s.say("RACE CONTROL: Welcome, Steward. Click cars to spectate, press R to file a report.", (0.8, 0.9, 1, 1))
        s.say("%d laps | incident frequency: %s" % (LAPS, s.levels[s.lvl_i]), (0.8, 0.9, 1, 1))
        if s.first_run: s.help.show(); s.first_run = False
        else: s.help.hide()

    def to_menu(s):
        s.state = "menu"; s.help.hide(); s.spawn_cars(); s.show_state()

    def resume(s): s.pmenu = False; s.show_state()

    def finish_race(s):
        s.state = "over"; s.panel = False; s.rep.hide()
        for inc in s.incidents:
            if inc.status == "open":
                inc.status = "expired"
                if inc.ruling > 0: s.score -= 40
        fin = sorted([c for c in s.cars if c.finished], key=lambda c: c.finish_time + c.pen_s)
        rest = sorted([c for c in s.cars if not c.finished], key=lambda c: -c.prog)
        R = s.res_content
        s.T(R, (-1.20, 0.80), 0.06, (1, 0.85, 0.2, 1), text="FINAL CLASSIFICATION (penalties applied)")
        win = fin[0].finish_time + fin[0].pen_s if fin else 0
        rows = ["POS DRV  #   TIME/GAP      PEN  WARN"]
        for k, c in enumerate(fin + rest, 1):
            if c.finished:
                tot = c.finish_time + c.pen_s; tm = fmt_lap(tot) if k == 1 else "+%.3f" % (tot - win)
            else:
                tm = "+%d LAP" % (int((fin[0].prog - c.prog) // s.track.L + 1) if fin else 0)
            rows.append("%2d  %-3s %2d  %-12s %4s  %s" % (k, c.code, c.num, tm, ("+%ds" % c.pen_s) if c.pen_s else "-", c.warnings or "-"))
        s.T(R, (-1.20, 0.68), 0.040, text="\n".join(rows))
        nrep = len([i for i in s.incidents])
        rank = "Chief Steward" if s.score >= 300 else "Race Steward" if s.score >= 150 else \
               "Trainee Steward" if s.score >= 0 else "Rookie Marshal"
        s.T(R, (-1.20, -0.50), 0.05, (0.5, 1, 0.6, 1), text="STEWARD SCORE %d   |   Rank: %s" % (s.score, rank))
        s.T(R, (-1.20, -0.60), 0.034, (0.8, 0.9, 1, 1),
            text="Incidents noted: %d   Reports filed: %d   Good calls: %d" % (nrep, s.reports, s.good))
        s.show_state()

    # =====================================================================
    #  report form
    # =====================================================================
    def close_report(s): s.panel = False; s.refresh_report()

    def sel_row(s, r): s.sel_inc = r; s.sel_car = 0; s.refresh_report()
    def sel_driver(s, k): s.sel_car = k; s.refresh_report()
    def sel_penalty(s, p): s.sel_pen = p; s.refresh_report()

    def refresh_report(s):
        if not s.panel: s.rep.hide(); return
        s.rep.show()
        ol = s.open_list()[:7]
        s.sel_inc = min(s.sel_inc, max(len(ol) - 1, 0))
        for r, b in enumerate(s.inc_btns):
            if r < len(ol):
                inc = ol[r]; names = "/".join("#%d" % c.num for c in inc.cars)
                b["root"].show()
                b["txt"].setText("%3d  %-3d %-13s %-18s %-15s %3ds" % (inc.n, inc.lap, inc.kind, names, inc.loc, s.t - inc.t))
                b["col"] = s.SEL if r == s.sel_inc else s.BASE
            else: b["root"].hide()
        form = [s.lbl_drv, s.lbl_pen]
        if not ol:
            s.rep_ev.setText("No open incidents.\nKeep watching - Race Control will note anything suspicious.")
            for b in s.drv_btns + s.pen_btns + [s.sub_btn, s.rep_replay]: b["root"].hide()
            for t in form: t.hide()
            return
        for t in form: t.show()
        inc = ol[s.sel_inc]; s.sel_car = min(s.sel_car, len(inc.cars) - 1)
        s.rep_ev.setText("EVIDENCE - INC %d - %s (%s)\n" % (inc.n, inc.kind, inc.loc) + "\n".join("  - " + e for e in inc.ev))
        for k, b in enumerate(s.drv_btns):
            if k < len(inc.cars):
                c = inc.cars[k]; b["root"].show(); b["txt"].setText("#%d  %s" % (c.num, c.code))
                b["col"] = s.SEL if k == s.sel_car else s.BASE
            else: b["root"].hide()
        for p, b in enumerate(s.pen_btns):
            b["root"].show(); b["col"] = s.SEL if p == s.sel_pen else s.BASE
        s.sub_btn["root"].show(); s.rep_replay["root"].show()

    # =====================================================================
    #  keys
    # =====================================================================
    def setup_input(s):
        for key, fn in (("r", s.k_report), ("v", s.k_replay), ("i", s.k_jump), ("tab", s.k_tab),
                        ("space", s.k_pause), ("h", s.k_help), ("escape", s.k_esc), ("enter", s.k_enter),
                        ("mouse1", s.on_click),
                        ("arrow_up", lambda: s.k_nav(-1, 0)), ("arrow_down", lambda: s.k_nav(1, 0)),
                        ("arrow_left", lambda: s.k_nav(0, -1)), ("arrow_right", lambda: s.k_nav(0, 1))):
            s.accept(key, fn)
        for n in range(1, 6): s.accept(str(n), s.k_num, [n])

    def live(s): return s.state in ("grid", "race", "over")

    def k_report(s):
        if s.state not in ("grid", "race") or s.pmenu: return
        s.panel = not s.panel; s.sel_inc = s.sel_car = 0
        if s.panel: s.help.hide()
        s.refresh_report()

    def k_num(s, n):
        if not s.live(): return
        if s.panel: s.sel_pen = n - 1; s.refresh_report()
        else: s.set_cam(n)

    def k_nav(s, dv, dh):
        if not s.panel: return
        ol = s.open_list()[:7]
        if dv and ol: s.sel_inc = (s.sel_inc + dv) % len(ol); s.sel_car = 0
        if dh and ol: s.sel_car = (s.sel_car + dh) % len(ol[s.sel_inc].cars)
        s.refresh_report()

    def k_tab(s):
        if s.live(): s.follow_car(s.cars[(s.follow + 1) % len(s.cars)])

    def k_pause(s):
        if s.state == "race" or s.state == "grid": s.paused = not s.paused

    def k_help(s):
        (s.help.show if s.help.isHidden() else s.help.hide)()

    def k_esc(s):
        if not s.help.isHidden(): s.help.hide()
        elif s.state == "menu": return
        elif s.replay: s.end_replay()
        elif s.panel: s.close_report()
        elif s.state in ("grid", "race"): s.pmenu = not s.pmenu; s.show_state()

    def k_enter(s):
        if s.state in ("menu", "over"): s.start_race()
        elif s.panel: s.submit()

    # =====================================================================
    #  frame loop
    # =====================================================================
    def update(s, task):
        dt = min(globalClock.getDt(), 1 / 30.0)
        if s.state != "menu":
            if s.replay: s.step_replay(dt)
            elif not (s.paused or s.pmenu): s.step_sim(dt); s.fx.update(dt)
        s.update_camera(dt); s.update_hover()
        if s.state != "menu": s.update_ui(dt)
        return task.cont

    def update_camera(s, dt):
        cam = s.camera
        if s.state == "menu":
            s.menu_a += dt * 0.07; a = s.menu_a
            cam.setPos(-45 + 260 * math.cos(a), -10 + 260 * math.sin(a), 70 + 20 * math.sin(a * 2.3))
            cam.lookAt(-45, -10, 4); s.camLens.setFov(62); s.sky.setPos(cam.getPos()); s.set_film(240)
            s.sun_np.setPos(35, -70, 150); s.sun_np.lookAt(-45, -10, 0)
            return
        car = s.cars[s.follow]
        pos = Vec3(car.vis.getPos(s.render)); fwd = car.vis.getQuat(s.render).getForward()
        k = 1 - math.exp(-7 * dt); m = s.cam_mode; fov = 72
        cp = Vec3(cam.getPos())
        if m == 1:
            want = pos - fwd * 14 + Vec3(0, 0, 5.2); cam.setPos(cp + (want - cp) * k)
            lk = pos + fwd * 10 + Vec3(0, 0, 1.0); cam.lookAt(lk[0], lk[1], lk[2])
        elif m == 2:
            want = pos + Vec3(0, -25, 95); cam.setPos(cp + (want - cp) * k); cam.lookAt(pos[0], pos[1], pos[2]); fov = 55
        elif m == 3:
            if s.cam_spot is None or (s.cam_spot - pos).length() > 300:
                s.cam_spot = min(s.track.cams, key=lambda v: (v - pos).length())
            cam.setPos(s.cam_spot); cam.lookAt(pos[0], pos[1], pos[2] + 1)
            fov = clamp(math.degrees(2 * math.atan(16 / max((s.cam_spot - pos).length(), 1))), 9, 60)
        elif m == 5 and s.pin_cam is not None:
            def dist(c): return (Vec3(c.vis.getX(), c.vis.getY(), 0) - s.pin_pt).length()
            near = min(s.cars, key=dist)
            if dist(s.cars[s.follow]) > dist(near) + 25: s.follow = near.i
            tgt = s.cars[s.follow]; d = dist(tgt)
            cam.setPos(s.pin_cam)
            if d < 260:
                tp = Vec3(tgt.vis.getX(), tgt.vis.getY(), 1.0); cam.lookAt(tp[0], tp[1], tp[2])
                fov = clamp(math.degrees(2 * math.atan(16 / max((s.pin_cam - tp).length(), 1))), 9, 60)
            else:
                cam.lookAt(s.pin_pt[0], s.pin_pt[1], s.pin_pt[2]); fov = 50
        else:
            cam.setPos(pos + fwd * 0.4 + Vec3(0, 0, 1.15)); cam.setHpr(car.vis.getH(s.render), -1.5, 0); fov = 88
        # speed sensation: FOV widens with speed; chase/onboard get vibration (stronger on kerbs and grass)
        sp = clamp(car.spd / 95.0, 0, 1)
        if m in (1, 4) and s.state != "grid":
            fov += (7 if m == 1 else 5) * sp
            s.shake_t += dt
            rough = 3.0 if abs(car.d) > HALF_W - 0.5 else 1.0
            amp = (0.22 if m == 1 else 0.45) * sp * rough
            cam.setR(cam.getR() + amp * math.sin(s.shake_t * 37.0))
            cam.setP(cam.getP() + amp * 0.7 * math.sin(s.shake_t * 53.3 + 1.0))
        s.set_film(240 if m == 2 else 100 if m == 4 else 150)
        s.camLens.setFov(fov)
        s.sky.setPos(cam.getPos())
        s.sun_np.setPos(pos + Vec3(80, -60, 150)); s.sun_np.lookAt(pos[0], pos[1], pos[2])

    def update_ui(s, dt):
        order = s.order
        for r, row in enumerate(s.rows):
            c = order[r]
            if r == 0: intv, gap = "LEADER", ""
            else: intv, gap = s.gap(c, order[r - 1]), s.gap(c, order[0])
            flag = "PIT" if c.pit_state else "DRS" if c.drs_open else "FIN" if c.finished else ""
            pen = ("+%ds" % c.pen_s) if c.pen_s else ""
            row.setText("%2d %-3s %2d %8s %8s %8s %-3s %4s" % (r + 1, c.code, c.num, intv, gap, fmt_lap(c.last_lap), flag, pen))
            row.setFg((1, 1, 1, 1) if c is s.cars[s.follow] else tuple(min(1, 0.45 + v * 0.65) for v in c.col) + (1,))
        lead = order[0]
        s.lap_txt.setText("LAP %d/%d     %s" % (min(LAPS, lead.laps_done + 1), LAPS, fmt_lap(s.t)))
        for k, r in enumerate(s.lights):
            if s.state != "grid": r.hide()
            else:
                r.show(); r.setColor(*((1, 0.05, 0.05, 1) if s.grid_t < 6.5 - k else (0.2, 0, 0, 1)))
        if s.banner_t > 0:
            s.banner_t -= dt
            if s.banner_t <= 0: s.banner_txt.setText("")
        s.status_txt.setText("REPLAY 0.5x - V/ESC to exit" if s.replay else "PAUSED" if s.paused else "")
        s.score_txt.setText("STEWARD SCORE  %d" % s.score)
        nopen = len(s.open_list())
        s.score_txt2.setText("Reports %d | Good calls %d | Open %d" % (s.reports, s.good, nopen))
        for k, b in enumerate(s.cam_btns):
            b["col"] = s.SEL if s.cam_mode == k + 1 else s.BASE
        s.act_btns[0]["txt"].setText("R REPORT (%d)" % nopen)
        s.act_btns[2]["txt"].setText("SPACE RESUME" if s.paused else "SPACE PAUSE")
        for k, t in enumerate(s.logs):
            idx = len(s.log) - 1 - k
            if idx >= 0: t.setText(s.log[idx][0][:92]); t.setFg(s.log[idx][1])
            else: t.setText("")
        for c in s.cars:
            op = c.drs_open if not s.replay else getattr(c, "drs_open_vis", False)
            c.flap_a += ((-32.0 if op else 0.0) - c.flap_a) * min(1, 12 * dt)
            c.flap.setP(c.flap_a)
        for c, d in zip(s.cars, s.dots):
            x, z = s.map_xy(c.vis.getX(), c.vis.getY()); d.setPos(x, 0, z)
        fc = s.cars[s.follow]; x, z = s.map_xy(fc.vis.getX(), fc.vis.getY()); s.ring.setPos(x, 0, z)
        if s.panel and int(s.t * 2) != getattr(s, "_rr", -1): s._rr = int(s.t * 2); s.refresh_report()
        s.update_tel()

    def update_tel(s):
        c = s.cars[s.follow]; kph = c.spd * 3.6
        gt = [0, 60, 95, 135, 175, 215, 255, 290, 345]
        gear = 1
        while gear < 8 and kph >= gt[gear]: gear += 1
        fr = clamp((kph - gt[gear - 1]) / (gt[gear] - gt[gear - 1]), 0, 1)
        rpm = 4500 if kph < 5 else int(6500 + 5500 * fr)
        s.tel_h.setText("TELEMETRY   #%d %s   P%d   %s" % (c.num, c.code, s.order.index(c) + 1, "PIT" if c.pit_state else ""))
        s.tel_spd.setText("%3d" % kph); s.tel_gear.setText(str(gear) if kph > 3 else "N")
        s.bar_t.setPos(-0.75, 0, 0.372); s.bar_t.setScale(max(0.001, 0.7 * c.thr), 1, 0.03)
        s.bar_b.setPos(-0.75, 0, 0.312); s.bar_b.setScale(max(0.001, 0.7 * c.brk), 1, 0.03)
        drs = "OPEN" if c.drs_open else "READY" if any(c.elig) else "off"
        s.tel_info.setText("RPM %5d   DRS %-5s   LAT %+.1fG   LON %+.1fG\nBEST %s   LAPS %d/%d   STRIKES %d" % (
            rpm, drs, c.glat, c.glon, fmt_lap(c.best_lap), c.laps_done, LAPS, c.strikes))
        if s.trace_np: s.trace_np.removeNode()
        if len(c.trace) > 2:
            x0, z0, wd, ht = -0.88, 0.03, 0.86, 0.165
            s.trace_np = holder = s.hud_br.attachNewNode("trace")
            for ix, col, sc in ((0, (1, 1, 1, 1), 100.0), (1, (0.3, 1, 0.3, 1), 1.0), (2, (1, 0.25, 0.25, 1), 1.0)):
                ls = LineSegs(); ls.setThickness(2); ls.setColor(*col)
                for k, smp in enumerate(c.trace):
                    (ls.moveTo if k == 0 else ls.drawTo)(x0 + wd * k / 149.0, 0, z0 + ht * clamp(smp[ix] / sc, 0, 1))
                holder.attachNewNode(ls.create())

    # ---- carried over from v1 (incident logic, physics loop, replay)
    def setup_lights(s):
        amb = AmbientLight("a"); amb.setColor((0.50, 0.53, 0.60, 1))
        s.render.setLight(s.render.attachNewNode(amb))
        sun = DirectionalLight("sun"); sun.setColor((1.0, 0.95, 0.85, 1))
        sun.setShadowCaster(True, 4096 if QUALITY == "high" else 2048, 4096 if QUALITY == "high" else 2048)
        # tight depth range: simplepbr's shadow bias is in normalised depth
        ln = s.sun_lens = sun.getLens(); ln.setFilmSize(240, 240); ln.setNearFar(100, 270); s._film = 240
        sun.setCameraMask(BitMask32.bit(5))
        s.sun_np = s.render.attachNewNode(sun)
        s.sun_np.setPos(100, -80, 200); s.sun_np.lookAt(0, 0, 0)
        s.render.setLight(s.sun_np)
        if simplepbr is not None and QUALITY == "high":
            s.pbr = simplepbr.init(msaa_samples=4, max_lights=2, use_normal_maps=False,
                                   enable_shadows=True, shadow_bias=0.0008, exposure=1.0)
        else:
            s.render.setShaderAuto()
        fog = Fog("fog"); fog.setColor(0.55, 0.72, 0.93); fog.setExpDensity(0.0007)
        s.render.setFog(fog)

    def set_film(s, v):
        """Sun shadow-frustum size (m): tight around the action = sharper shadows from the same 4k map."""
        if abs(v - s._film) > 0.5: s._film = v; s.sun_lens.setFilmSize(v, v)

    def setup_fx(s):
        s.blob_tex = soft_tex((0, 0, 0), 0.55, 1.0, "blob")
        s.fx = Particles(s.render, soft_tex((1, 1, 1), 1.0, 1.6, "puff"))
        s.shake_t = 0.0

    def setup_post(s):
        """Vignette always; HDR bloom (high quality only), chained onto simplepbr's offscreen buffer so its
        4x MSAA and tonemapping are kept. (SSAO is not possible here: simplepbr's shaders write no normals.)"""
        cm = CardMaker("vig"); cm.setFrame(-1, 1, -1, 1)
        s.vig = s.render2d.attachNewNode(cm.generate()); s.vig.setTexture(vignette_tex())
        s.vig.setTransparency(TransparencyAttrib.MAlpha); s.vig.setBin("background", -1)
        s.filters = None
        if QUALITY != "high": return
        try:
            from direct.filter.CommonFilters import CommonFilters
            pbr = getattr(s, "pbr", None)
            target = pbr._filtermgr.buffers[0] if pbr is not None else s.win
            f = CommonFilters(target, s.cam)
            f.setHighDynamicRange(); f.setMSAA(4)
            f.setBloom(blend=(0.3, 0.4, 0.3, 0.0), mintrigger=0.7, maxtrigger=1.2, desat=0.6,
                       intensity=0.7, size="medium")
            s.filters = f
        except Exception as e:
            print("post-processing skipped:", e)

    def emit_fx(s, dt):
        """Tyre smoke when sliding or locking, dust when on the grass."""
        tr = s.track
        for c in s.cars:
            if c.spd < 8: continue
            grass = tr.is_grass(c.d, tr.rel(c.idx))
            kinds = ((c.spd * 1.2 if grass else 0.0, (0.55, 0.45, 0.30), 0.45, 2.2),
                     (70.0 if abs(c.vl) > 3.0 else 0.0, (0.9, 0.9, 0.9), 0.5, 3.0),
                     (14.0 if (c.brk > 0.92 and c.spd > 55) else 0.0, (0.85, 0.85, 0.85), 0.35, 2.4))
            q = c.np.getQuat(s.render); f, r = q.getForward(), q.getRight(); p = c.np.getPos()
            for rate, col, a0, sz in kinds:
                if rate and random.random() < rate * dt:
                    pos = p - f * 1.5 + r * random.choice((-0.95, 0.95)) + Vec3(0, 0, 0.25)
                    vel = c.v * 0.15 + Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.5, 1.8))
                    s.fx.emit(pos, vel, random.uniform(1.2, 2.2), sz * 0.5, sz * 2.2, col, a0, 1.2, 0.3)

    def sparks(s, pos, rel):
        for _ in range(int(min(40, 8 + rel * 2))):
            v = Vec3(random.uniform(-7, 7), random.uniform(-7, 7), random.uniform(1, 6))
            s.fx.emit(pos + Vec3(0, 0, 0.3), v, random.uniform(0.3, 0.8), 0.35, 0.1, (1.0, 0.8, 0.3), 1.0, 0.8, -12.0)
        for _ in range(4):
            s.fx.emit(pos + Vec3(0, 0, 0.4), Vec3(random.uniform(-2, 2), random.uniform(-2, 2), 1.0),
                      1.6, 1.5, 5.0, (0.8, 0.8, 0.8), 0.5, 1.5, 0.2)

    def build_scene(s):
        tr = s.track; N = tr.N; ds = tr.ds
        # one mesh per material, each with its own texture and tile size (m)
        m = Mesh(6.0); mg = Mesh(5.0); ma = Mesh(3.0); mv = Mesh(2.5); mp = Mesh(2.0)
        up = (0, 0, 1)
        mg.quad((-3000, -3000, -0.12), (3000, -3000, -0.12), (3000, 3000, -0.12), (-3000, 3000, -0.12),
                (0.20, 0.47, 0.20, 1), up)
        def edge(i, d, z): return (tr.x[i % N] + tr.nx[i % N] * d, tr.y[i % N] + tr.ny[i % N] * d, z)
        def strip(i, j, d0, d1, z, col, mm=None):
            (mm or m).quad(edge(i, d0, z), edge(j, d0, z), edge(j, d1, z), edge(i, d1, z), col, up)
        def across(i, d0, d1, thick, col, z=0.05, mm=None): strip(i, i + max(1, int(thick / ds)), d0, d1, z, col, mm or mp)
        rnd = random.Random(11)
        patch = [rnd.random() for _ in range(N // 12 + 2)]
        for i in range(N):
            j = i + 1
            sh = 0.135 + 0.028 * patch[i // 12]                       # tarmac patchwork, ~30 m sections
            gravel = (0.62, 0.57, 0.42, 1)
            strip(i, j, HALF_W, HALF_W + 16, -0.06, gravel, mv); strip(i, j, -HALF_W - 16, -HALF_W, -0.06, gravel, mv)
            band = (0.20, 0.47, 0.20, 1) if (i // 8) % 2 == 0 else (0.24, 0.54, 0.22, 1)   # mown stripes
            strip(i, j, HALF_W + 16, HALF_W + 70, -0.08, band, mg); strip(i, j, -HALF_W - 70, -HALF_W - 16, -0.08, band, mg)
            strip(i, j, -HALF_W, HALF_W, 0.0, (sh, sh, sh + 0.01, 1), ma)
            # rubbered-in racing line: hugs the inside of corners, graded edges
            c = clamp(tr.k[i % N] * 250.0, -3.5, 3.5)
            for d0, d1, f in ((-1.9, -0.9, 0.88), (-0.9, 0.9, 0.74), (0.9, 1.9, 0.88)):
                strip(i, j, c + d0, c + d1, 0.01, (sh * f, sh * f, (sh + 0.01) * f, 1), ma)
            strip(i, j, HALF_W - 0.35, HALF_W, 0.04, (1, 1, 1, 1), mp)
            strip(i, j, -HALF_W, -HALF_W + 0.35, 0.04, (1, 1, 1, 1), mp)
            if abs(tr.k[i]) > 0.003:
                kc = (0.85, 0.1, 0.1, 1) if (i // 2) % 2 == 0 else (0.95, 0.95, 0.95, 1)
                strip(i, j, HALF_W, HALF_W + 1.3, 0.06, kc, mp); strip(i, j, -HALF_W - 1.3, -HALF_W, 0.06, kc, mp)
        for _ in range(45):                                            # repair patches across the track
            i = rnd.randrange(N); w0 = rnd.uniform(-HALF_W, 0.0); w1 = min(HALF_W, w0 + rnd.uniform(3, 12))
            v = rnd.choice((0.122, 0.168))
            strip(i, i + rnd.randint(1, 4), w0, w1, 0.02, (v, v, v + 0.01, 1), ma)
        # start/finish chequers
        cols = 20; cw = 2 * HALF_W / cols
        for row in range(2):
            for c in range(cols):
                col = (1, 1, 1, 1) if (c + row) % 2 else (0.05, 0.05, 0.05, 1)
                strip(tr.i0 + row, tr.i0 + row + 1, -HALF_W + c * cw, -HALF_W + (c + 1) * cw, 0.05, col, mp)
        # DRS detection (orange) and activation (green) lines
        for det, a0, a1 in tr.drs:
            across(det, -HALF_W, HALF_W, 1.0, (1.0, 0.55, 0.0, 1)); across(a0, -HALF_W, HALF_W, 1.0, (0.1, 0.9, 0.2, 1))
            across(a1, -HALF_W, HALF_W, 1.0, (0.9, 0.1, 0.1, 1))
        # pit lane
        K = int(260 / ds)
        for o in range(-K, K):
            w0, w1 = pitw(o * ds), pitw((o + 1) * ds)
            if w0 < 0.02 or w1 < 0.02: continue
            i = tr.i0 + o; c0, c1, h0, h1 = PIT_C * w0, PIT_C * w1, 5 * w0, 5 * w1
            ma.quad(edge(i, c0 - h0, 0.03), edge(i + 1, c1 - h1, 0.03), edge(i + 1, c1 + h1, 0.03),
                    edge(i, c0 + h0, 0.03), (0.30, 0.30, 0.36, 1), up)
            if abs(o * ds) <= PIT_ZONE:
                strip(i, i + 1, c0 - 5, c0 - 4.75, 0.05, (1, 0.85, 0.1, 1), mp)
                strip(i, i + 1, c0 + 4.75, c0 + 5, 0.05, (1, 0.85, 0.1, 1), mp)
        for sg in (-PIT_ZONE, PIT_ZONE):
            across(tr.idx_at(sg), -19.5, -9.5, 1.5, (1, 0.85, 0.1, 1))
        for o in range(int(-185 / ds), int(185 / ds), 8):
            i = tr.i0 + o
            m.box(edge(i + 4, -9.4, 0.5), (8 * ds + 0.3, 0.4, 1.0), (0.85, 0.85, 0.85, 1), tr.th[(i + 4) % N])
        for k, (num, code, col) in enumerate(DRIVERS):
            i = tr.idx_at(-150.0 + k * 33.0)
            m.box(edge(i, -27, 2.5), (26, 9, 5), (0.82, 0.82, 0.86, 1), tr.th[i])
            m.box(edge(i, -27, 5.2), (26.6, 9.6, 0.5), (col[0], col[1], col[2], 1), tr.th[i])
        # grandstands (opposite the pits)
        pal = [(0.75, 0.2, 0.2), (0.2, 0.3, 0.75), (0.85, 0.7, 0.2)]
        for b, sr in enumerate((-170, 0, 170)):
            i = tr.idx_at(sr); cc = pal[b]
            for st in range(4):
                h = 2.2 + 2.2 * st
                m.box(edge(i, 30 + st * 6, h / 2), (70, 6, h), (cc[0], cc[1], cc[2], 1), tr.th[i])
        # starting gantry
        for sg in (-1, 1): m.box(edge(tr.i0, sg * (HALF_W + 2), 4.5), (1, 1, 9), (0.15, 0.15, 0.18, 1), tr.th[tr.i0])
        m.box(edge(tr.i0, 0, 9.5), (1.2, 2 * HALF_W + 6, 1.4), (0.15, 0.15, 0.18, 1), tr.th[tr.i0])
        # trees
        random.seed(7)
        xs, ys = [min(tr.x) - 250, max(tr.x) + 250], [min(tr.y) - 250, max(tr.y) + 250]
        sub = list(range(0, N, 8)); placed = 0
        while placed < 90:
            x, y = random.uniform(*xs), random.uniform(*ys)
            if min((x - tr.x[i]) ** 2 + (y - tr.y[i]) ** 2 for i in sub) < 60 ** 2: continue
            gch = random.uniform(0.35, 0.6)
            m.frustum((x, y), 0, 3.2, 0.4, 0.3, (0.35, 0.22, 0.12, 1), 6)
            m.frustum((x, y), 2.2, 11, 3.4, 0.1, (0.08, gch * 0.6, 0.1, 1), 8)
            placed += 1
        random.seed()
        s.make_textures()
        for mesh, name in ((m, "structures"), (mg, "grass"), (ma, "asphalt"), (mv, "gravel"), (mp, "paint")):
            mesh.build(s.render, name).setTexture(s.tex[name])

    def make_textures(s):
        s.tex = {
            "structures": s.make_tex(0.82, 0.34, ((16, 0.45),), seed=3),
            "asphalt": s.make_tex(0.78, 0.50, ((8, 0.28), (16, 0.22), (64, 0.16)), speck=0.03, seed=5),
            "grass": s.make_tex(0.76, 0.60, ((4, 0.40), (8, 0.30), (32, 0.25)), tint=(0.93, 1.0, 0.85), seed=7),
            "gravel": s.make_tex(0.72, 0.75, ((8, 0.30), (16, 0.30), (64, 0.30)), speck=0.08, seed=9),
            "paint": s.make_tex(0.92, 0.20, ((8, 0.16),), speck=0.03, seed=13),
        }

    def make_tex(s, base, fine, octs, speck=0.0, tint=(1, 1, 1), seed=1, n=256):
        """Mean-grey tileable multi-octave value noise (+ speckle), modulated with vertex colours."""
        rnd = random.Random(seed)
        lat = [(c, amp, [[rnd.random() for _ in range(c)] for _ in range(c)]) for c, amp in octs]
        img = PNMImage(n, n, 3)
        for y in range(n):
            for x in range(n):
                v = base + (rnd.random() - 0.5) * fine
                for c, amp, L in lat:
                    fx, fy = x * c / n, y * c / n; ix, iy = int(fx), int(fy); tx, ty = fx - ix, fy - iy
                    ix2, iy2 = (ix + 1) % c, (iy + 1) % c
                    lo = (L[iy][ix] * (1 - tx) + L[iy][ix2] * tx) * (1 - ty) + (L[iy2][ix] * (1 - tx) + L[iy2][ix2] * tx) * ty
                    v += (lo - 0.5) * amp
                if speck and rnd.random() < speck: v += rnd.choice((-0.30, 0.25))
                v = clamp(v, 0, 1); img.setXel(x, y, v * tint[0], v * tint[1], v * tint[2])
        tex = Texture("tex"); tex.load(img)
        tex.setWrapU(SamplerState.WM_repeat); tex.setWrapV(SamplerState.WM_repeat)
        tex.setMinfilter(SamplerState.FT_linear_mipmap_linear); tex.setAnisotropicDegree(8)
        return tex

    def build_sky(s):
        """Gradient sky dome that follows the camera (no lighting, no depth write)."""
        m = Mesh(); R = 3500.0; rings, segs = 8, 24
        def col(el):
            t = clamp(el / (math.pi / 2), 0, 1)
            return (0.78 + (0.20 - 0.78) * t, 0.86 + (0.45 - 0.86) * t, 0.95 + (0.85 - 0.95) * t, 1)
        def pt(el, az): return (R * math.cos(el) * math.cos(az), R * math.cos(el) * math.sin(az), R * math.sin(el))
        for r in range(rings):
            e0, e1 = -0.1 + (math.pi / 2 + 0.1) * r / rings, -0.1 + (math.pi / 2 + 0.1) * (r + 1) / rings
            for k in range(segs):
                a0, a1 = 2 * math.pi * k / segs, 2 * math.pi * (k + 1) / segs
                P = [pt(e0, a0), pt(e0, a1), pt(e1, a1), pt(e1, a0)]; C = [col(e0), col(e0), col(e1), col(e1)]
                m.tri(P[0], P[1], P[2], C[0], (0, 0, -1)); m.tri(P[0], P[2], P[3], C[2], (0, 0, -1))
        s.sky = m.build(s.render, "sky")
        s.sky.setLightOff(); s.sky.setShaderOff(); s.sky.setFogOff(); s.sky.setTwoSided(True)
        s.sky.setBin("background", 0); s.sky.setDepthWrite(False); s.sky.setDepthTest(False)

    def rect(s, parent, x0, z0, x1, z1, col):
        cm = CardMaker("r"); cm.setFrame(x0, x1, z0, z1)
        n = parent.attachNewNode(cm.generate()); n.setColor(*col)
        n.setTransparency(TransparencyAttrib.MAlpha); return n

    def T(s, parent, pos, scale=0.04, fg=(1, 1, 1, 1), align=TextNode.ALeft, mono=True, text=""):
        return OnscreenText(text=text, parent=parent, pos=pos, scale=scale, fg=fg, align=align,
                            font=s.mono if mono else None, shadow=(0, 0, 0, 0.9), mayChange=True)

    def k_jump(s):
        if s.incidents:
            inc = s.incidents[-1]; s.follow = inc.cars[0].i; s.cam_spot = None

    def k_replay(s):
        ol = s.open_list(); inc = ol[min(s.sel_inc, len(ol) - 1)] if ol else (s.incidents[-1] if s.incidents else None)
        if s.replay: s.end_replay(); return
        if not inc or not s.hist_t: return
        s.replay = {"t": max(inc.t - 7.0, s.hist_t[0]), "end": min(inc.t + 3.0, s.hist_t[-1]), "inc": inc}
        s.follow = inc.cars[0].i; s.cam_spot = None
        s.rep.hide()

    def end_replay(s):
        s.replay = None
        for c in s.cars: c.vis.setPos(c.np.getX(), c.np.getY(), 0.0); c.vis.setH(c.np.getH())
        if s.panel: s.refresh_report()

    def open_list(s): return [i for i in reversed(s.incidents) if i.status == "open"]

    def say(s, text, col=(1, 1, 1, 1)):
        s.log.append((("%s  " % fmt_lap(s.t) if s.t > 0 else "") + text, col)); s.log = s.log[-7:]

    def banner(s, text, secs, col=(1, 1, 1, 1)):
        s.banner_txt.setText(text); s.banner_txt.setFg(col); s.banner_t = secs

    def add_incident(s, kind, cars, fault, ruling, loc, ev, why, col):
        s.inc_n += 1
        lap = min(LAPS, max(c.laps_done for c in cars) + 1)
        inc = Incident(s.inc_n, s.t, kind, cars, fault, ruling, lap, loc, ev, why)
        s.incidents.append(inc)
        names = " / ".join("#%d %s" % (c.num, c.code) for c in cars)
        s.say("INCIDENT %d - LAP %d - %s - %s (%s) - NOTED" % (inc.n, lap, kind, names, loc), col)
        s.banner("INCIDENT NOTED: %s %s" % (kind, names), 3.0, col)
        s.refresh_report()

    def on_collision(s, a, b):
        tr = s.track
        rel = (a.pv - b.pv).length()
        if rel < 2.0: return
        s.sparks((a.np.getPos() + b.np.getPos()) * 0.5, rel)
        dp = wrap(b.prog - a.prog, tr.L); dd = b.d - a.d
        sg = 1 if dd >= 0 else -1
        na, nb = (tr.nx[a.idx], tr.ny[a.idx]), (tr.nx[b.idx], tr.ny[b.idx])
        la = a.pv.x * na[0] + a.pv.y * na[1]; lb = b.pv.x * nb[0] + b.pv.y * nb[1]
        ta, tb = la * sg, lb * -sg                         # lateral speed toward the other car
        rear, front = (a, b) if dp > 0 else (b, a)
        along = abs(dp); fault, why = None, "Side-by-side, no clear lateral movement: racing incident."
        if along > 2.8:
            fault = rear; why = "#%d hit the rear of #%d." % (rear.num, front.num)
            ft, rt = (tb, ta) if front is b else (ta, tb)
            if ft - rt > 2.2: fault = front; why = "#%d moved across #%d." % (front.num, rear.num)
        else:
            if ta - tb > 0.9: fault = a; why = "#%d moved into #%d." % (a.num, b.num)
            elif tb - ta > 0.9: fault = b; why = "#%d moved into #%d." % (b.num, a.num)
        ruling = 0 if not fault else (1 if rel < 4 else 2 if rel < 9 else 3 if rel < 15 else 4)
        for c in (a, b):
            if fault is None or c is not fault: c.last_hit_t = s.t
        cars = sorted([a, b], key=lambda c: c.num); n1, n2 = cars
        ev = ["Contact speed: %.0f km/h (%.1f m/s)" % (rel * 3.6, rel),
              "Geometry: %s (%.1f m apart along track)" % ("REAR CONTACT" if along > 2.8 else "SIDE BY SIDE", along)]
        if along <= 2.8:
            t1 = ta if n1 is a else tb; t2 = tb if n1 is a else ta
            ev.append("Lateral movement: #%d %+.1f m/s toward #%d | #%d %+.1f m/s toward #%d" %
                      (n1.num, t1, n2.num, n2.num, t2, n1.num))
        else:
            ev.append("#%d was behind and closing, #%d ahead on track" % (rear.num, front.num))
        s.add_incident("COLLISION", cars, fault, ruling, tr.where(a.idx), ev, why, (1, 0.4, 0.3, 1))

    def track_limit_incident(s, c, dur, mx):
        forced = c.last_hit_t > c.tl_t0 - 3.5
        if forced: return
        c.strikes += 1
        ruling = 0 if forced else (0 if c.strikes <= 2 else 1 if c.strikes == 3 else 2)
        ev = ["All four wheels beyond white line for %.1f s (max %.1f m beyond)" % (dur, mx),
              "Track-limit strike %d for #%d this race" % (c.strikes, c.num)]
        if forced: ev.append("Contact with another car in the 3.5s before leaving the track")
        why = "Forced off by contact." if forced else "Strike %d: %s" % (c.strikes, "no action yet" if ruling == 0 else "over the allowance")
        s.add_incident("TRACK LIMITS", [c], c, ruling, c.tl_loc, ev, why, (1, 0.9, 0.3, 1))

    def pit_incident(s, c, kph):
        over = kph - 80.0
        if over <= 0.3: return
        ruling = 1 if over <= 2 else 2 if over <= 10 else 3
        ev = ["Max speed in pit lane timing zone: %.1f km/h" % kph, "Pit lane limit: 80.0 km/h",
              "Over the limit by %.1f km/h" % over]
        s.add_incident("PIT SPEEDING", [c], c, ruling, "Pit lane", ev, "Pit lane speeding, %.1f km/h over." % over, (0.5, 0.8, 1, 1))

    def submit(s):
        ol = s.open_list()[:7]
        if not ol: return
        inc = ol[s.sel_inc]; car = inc.cars[s.sel_car]; pen = s.sel_pen
        corr = inc.ruling
        if corr == 0: pts = 60 if pen == 0 else -(40 + 20 * pen)
        elif car is not inc.fault: pts = -40 if pen == 0 else -100
        else: pts = -60 if pen == 0 else [100, 55, 15, -20, -40][min(abs(pen - corr), 4)]
        inc.status = "closed"; s.reports += 1; s.score += pts
        if pts >= 55: s.good += 1
        if pen == 1: car.warnings += 1
        car.pen_s += PENALTIES[pen][1]
        verdict = "Stewards agreed: %s%s" % (SHORT[corr], " on #%d" % inc.fault.num if corr and inc.fault else "")
        s.say("RULING INC %d: #%d %s | %s | %+d pts" % (inc.n, car.num, SHORT[pen], verdict, pts),
              (0.4, 1, 0.5, 1) if pts > 0 else (1, 0.4, 0.4, 1))
        s.banner("%s %+d" % (SHORT[pen] + " #%d" % car.num, pts), 3.0, (0.4, 1, 0.5, 1) if pts > 0 else (1, 0.4, 0.4, 1))
        s.say("   Reason: " + inc.why, (0.8, 0.8, 0.8, 1))
        s.sel_inc = s.sel_car = 0; s.refresh_report()

    def step_sim(s, dt):
        if s.state == "grid":
            s.grid_t -= dt
            if s.grid_t <= 0:
                s.state = "race"; s.banner("LIGHTS OUT - AND AWAY WE GO!", 3, (0.4, 1, 0.4, 1))
                s.say("LIGHTS OUT - the race is on", (0.4, 1, 0.4, 1))
        else:
            s.t += dt
        hold = s.state == "grid"
        for c in s.cars: c.control(dt, hold)
        for c in s.cars: c.pv = c.v
        s.world.doPhysics(dt, 10, 1 / 240.0)
        for c in s.cars: c.post(dt)
        s.order = sorted(s.cars, key=lambda c: (0, c.finish_time) if c.finished else (1, -c.prog))
        if s.state != "race": return
        s.check_contacts(); s.emit_fx(dt)
        for c in s.cars: c.monitor(dt)
        for inc in s.incidents:
            if inc.status == "open" and s.t - inc.t > 150:
                inc.status = "expired"
                if inc.ruling > 0:
                    s.score -= 40; s.say("INC %d closed with no report - missed penalty (-40)" % inc.n, (1, 0.5, 0.3, 1))
        s.hist_acc += dt
        if s.hist_acc >= 1 / 30.0:
            s.hist_acc = 0
            s.hist_t.append(s.t)
            s.hist.append([(c.np.getX(), c.np.getY(), c.np.getH(), c.drs_open) for c in s.cars])
            if len(s.hist_t) > 3600: del s.hist_t[:600]; del s.hist[:600]
            for c in s.cars: c.trace.append((c.spd, c.thr, c.brk))
        if s.first_fin is not None and (all(c.finished for c in s.cars) or s.t - s.first_fin > 25):
            s.finish_race()

    def check_contacts(s):
        cs = s.cars
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                A, B = cs[a], cs[b]
                if A.finished or B.finished: continue
                dx, dy = A.np.getX() - B.np.getX(), A.np.getY() - B.np.getY()
                key = (a, b)
                touching = False
                if dx * dx + dy * dy < 49:
                    touching = s.world.contactTestPair(A.node, B.node).getNumContacts() > 0
                if touching:
                    if key not in s.contact and s.t - s.last_contact.get(key, -99) > 2.0:
                        s.on_collision(A, B)
                    s.contact[key] = True; s.last_contact[key] = s.t
                elif key in s.contact and s.t - s.last_contact[key] > 0.4:
                    del s.contact[key]

    def on_finish(s, c):
        if s.first_fin is None:
            s.first_fin = s.t; s.banner("CHEQUERED FLAG - #%d %s WINS" % (c.num, c.code), 5, (1, 1, 1, 1))
            s.say("CHEQUERED FLAG - #%d %s takes the win" % (c.num, c.code), (1, 1, 1, 1))

    def step_replay(s, dt):
        r = s.replay; r["t"] += dt * 0.5
        if r["t"] > r["end"]: s.end_replay(); return
        k = clamp(bisect.bisect_right(s.hist_t, r["t"]) - 1, 0, len(s.hist_t) - 1)
        k2 = min(k + 1, len(s.hist_t) - 1)
        den = s.hist_t[k2] - s.hist_t[k]; f = (r["t"] - s.hist_t[k]) / den if den > 1e-6 else 0.0
        for c, a, b in zip(s.cars, s.hist[k], s.hist[k2]):
            c.vis.setPos(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, 0.0)
            c.vis.setH(a[2] + wrap(b[2] - a[2], 360.0) * f)
            c.drs_open_vis = a[3]

    def gap(s, c, o):
        if o.prog - c.prog >= s.track.L: return "+%d LAP" % int((o.prog - c.prog) // s.track.L)
        k = c.loop_key
        if k >= 0 and k in o.times and k in c.times: return "+%.3f" % (c.times[k] - o.times[k])
        return "--"


if __name__ == "__main__":
    Game().run()
