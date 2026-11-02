"""The pure-numpy 2.5D coral-health map engine (no torch, no Unreal).

Ported unchanged in logic from CoralUnreal `scripts/healthmap/engine.py`; only the offline-only helpers
(`HealthMap.save/load`, `mask_from_rgba`, `mask_to_rgba`) were dropped.

A `HealthMap` holds a top-down grid over the reef.  Each cell carries the reef surface height
(from the DEM, `build_dem.py`) and running per-class vote weights.  Every call to `update()`
takes ONE segmentation mask plus the camera pose that produced it, turns the (subsampled)
pixels into rays, marches each ray against the height field, and adds a range-weighted vote to
the cell the ray hits.

--------------------------------------------------------------------------------------------
COORDINATE AND POSE CONVENTIONS (all of them, explicitly)
--------------------------------------------------------------------------------------------
World frame        Unreal, LEFT-handed, centimetres:  +X forward, +Y right, +Z up.
                   The DEM stores `world_cm = (-100*x_ply, +100*y_ply, +100*z_ply)`.

Grid               `ix` indexes +X, `iy` indexes +Y.
                   `world_x = origin_cm[0] + (ix+0.5)*cell_cm`, same for Y.
                   Images are NORTH-UP: row 0 = largest X, column 0 = smallest Y.

Camera pose        `(location_cm, rotation_deg)` where `location_cm = (x, y, z)` in world cm
                   and `rotation_deg = (pitch, yaw, roll)` in degrees, i.e. exactly an
                   `FRotator` in Unreal's own field order (`FRotator(Pitch, Yaw, Roll)`).
                   `update()` also accepts a flat 6-sequence `(x, y, z, pitch, yaw, roll)`
                   or a dict with keys `location_cm` / `rotation_deg` (or `x,y,z,pitch,yaw,roll`).

                   The camera basis is Unreal's `FRotationMatrix(FRotator)`, verbatim:
                       forward = ( CP*CY,             CP*SY,             SP   )
                       right   = ( SR*SP*CY - CR*SY,  SR*SP*SY + CR*CY, -SR*CP)
                       up      = (-(CR*SP*CY + SR*SY), CY*SR - CR*SP*SY, CR*CP)
                   with C/S of Pitch/Yaw/Roll.  Sanity: pitch -90, yaw 0, roll 0 gives
                   forward (0,0,-1), right (0,+1,0), up (+1,0,0).

Pixel -> ray       Pinhole, square pixels, principal point at the image centre, top-left pixel
                   origin (the `CSEG` protocol's mask layout).  For a W x H mask,

                       su = (u + 0.5 - W/2) / (W/2)        in (-1, +1), + to the RIGHT
                       sv = (v + 0.5 - H/2) / (H/2)        in (-1, +1), + DOWNWARDS
                       th = tan(fov_h/2)                   half-width of the image plane at z=1
                       tv = th * H / W                     => vfov = 2*atan(th*H/W)
                       d  = normalize(forward + su*th*right - sv*tv*up)

                   Checks this satisfies: the continuous image centre (u=(W-1)/2, v=(H-1)/2)
                   gives su=sv=0 and d == forward exactly; the continuous image edge (u=-0.5)
                   gives su=-1 and angle(d, forward) = fov_h/2 (39.0 deg at 78 deg).
                   At 640x360 and fov_h = 78 deg the vertical FOV is
                   2*atan(tan(39 deg)*9/16) = 48.98 deg; at the 78.14 deg the demo camera
                   actually uses it is 49.09 deg.  See `vfov_deg()`.

Ray marching       Fixed step `step_cm` (default = one cell).  A ray hits at the first sample
                   where `p.z <= height(cell)`; the range is then refined by linear
                   interpolation of `p.z - height` between the last two samples, so the hit
                   RANGE is accurate to well under one cell even though the step is one cell.
                   The voted CELL is HYBRID.  On open ground the cell of the interpolated
                   crossing is the right answer: the sample that went under the surface sits
                   up to one whole step BEYOND the true crossing (2.5-4 cm horizontally at
                   pitch -60 with a 5 cm step), a systematic ~0.5-cell bias towards the far
                   side of the footprint that an oracle sharing this raycast cannot see.
                   Across a raised FACE, though, the two samples straddle a wall and the
                   interpolated crossing lands in the floor cell IN FRONT of the face, so
                   re-indexing it would make every face pixel vote the floor.  A wall is
                   exactly the case where the surface of the SAMPLE's cell stands above the
                   hit point, so: vote the crossing's cell unless the sample cell's surface
                   is above the hit z, in which case vote the sample's cell.
                   The grid is convex, so a ray that has been inside it and steps out is
                   killed rather than marched on.  If the first sample inside the grid is
                   already under the surface (the camera sits outside the grid and the ray
                   enters below the reef) there is no previous inside sample to interpolate
                   from, so the hit is placed AT that sample (frac = 1) - never behind it,
                   which would put the hit point outside the grid.
                   Rays that leave the grid, never descend to the surface, or only hit beyond
                   `max_range_m` contribute nothing.

Vote weight        `w = 1 / (1 + range_m)`, zero for `range_m > max_range_m` (default 3.5 m).
                   Bleaching detection is unreliable beyond ~1.5 m (memory note
                   `bleaching-usefulness-dinov3`), so `RELIABLE_RANGE_M = 1.5` is exported and
                   `update(..., reliable_only=True)` restricts voting to that range.

Classes            0 LIVE, 1 BLEACHED, 2 DEGRADED, 3 OTHER (CoralCapstone health mapping v2,
                   identical to the health classes in docs/PROTOCOL.md).
                   255 in a mask means IGNORE and is dropped.

Read-out gating    A cell is REPORTED (`class_map`, `surveyed`, all the area read-outs) when it
                   has at least `min_obs` OBSERVATIONS - rays, not weight.  Weight is
                   range-dependent (`w = 1/(1+range_m)` is 0.83 at 0.2 m and 0.22 at 3.5 m), so
                   a weight threshold is really a range threshold in disguise: at 3.5 m it takes
                   five times as many rays to clear the same bar as at 0.2 m.  `min_votes` is
                   kept as an OPTIONAL extra gate (default 0, i.e. off).  Both are
                   KEYWORD-ONLY on every read-out, so a caller written against the old
                   weight-first signature fails loudly instead of truncating a 0.01 weight
                   threshold into an observation count of 0 (which surveys everything).
                   Exact ties are broken towards BLEACHED (then DEGRADED, then OTHER, then
                   LIVE) - `TIE_BREAK_ORDER`.  A reef survey should over-report damage rather
                   than under-report it, and `np.argmax` alone would hand every exact tie to
                   LIVE simply because LIVE is class 0.  `class_prob()` exposes the per-cell
                   vote share so a consumer can see how close the call was (risk 3 of the
                   report: flag cells whose winning share is < 0.6).
--------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np

CLASS_NAMES = ("LIVE", "BLEACHED", "DEGRADED", "OTHER")
NUM_CLASSES = 4
LIVE, BLEACHED, DEGRADED, OTHER = 0, 1, 2, 3
IGNORE = 255

# docs/PROTOCOL.md display palette (the Unreal HUD legend)
PALETTE = np.array([
    (0, 200, 80),      # LIVE
    (255, 255, 255),   # BLEACHED
    (200, 60, 40),     # DEGRADED
    (60, 60, 200),     # OTHER
], dtype=np.uint8)

DEFAULT_FOV_H_DEG = 78.0        # AAUVPawn PerceptionFOV
DEFAULT_MAX_RANGE_M = 3.5
RELIABLE_RANGE_M = 1.5          # bleaching is unreliable past this (see module docstring)

DEFAULT_MIN_OBS = 3             # rays a cell needs before it is reported at all
DEFAULT_MIN_VOTES = 0.0         # optional EXTRA weight gate, off by default

# A hit counts as a WALL (and votes the marched sample's cell rather than the interpolated
# crossing's) when the sample cell's surface stands this far above the hit point.  Only
# float32-vs-float64 noise on a flat surface is meant to fall inside it - 1 micron.
WALL_TOL_CM = 1e-4

# Exact-tie priority, most "conservative" first: a survey should over-report damage.
# `class_map` argmaxes over the votes REORDERED by this, so an exact tie goes to the class
# that appears first here instead of to whichever class has the smallest id.
TIE_BREAK_ORDER = (BLEACHED, DEGRADED, OTHER, LIVE)
_TIE_ORDER = np.asarray(TIE_BREAK_ORDER, np.int64)


# ============================================================================ camera maths
def rotator_basis(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """Unreal FRotator -> (3,3) matrix whose ROWS are (forward, right, up) in world axes."""
    p, y, r = np.deg2rad([pitch_deg, yaw_deg, roll_deg])
    CP, SP = np.cos(p), np.sin(p)
    CY, SY = np.cos(y), np.sin(y)
    CR, SR = np.cos(r), np.sin(r)
    return np.array([
        [CP * CY, CP * SY, SP],
        [SR * SP * CY - CR * SY, SR * SP * SY + CR * CY, -SR * CP],
        [-(CR * SP * CY + SR * SY), CY * SR - CR * SP * SY, CR * CP],
    ], dtype=np.float64)


def parse_pose(pose) -> Tuple[np.ndarray, np.ndarray]:
    """Accept a dict / (loc, rot) pair / flat 6-sequence -> (loc_cm (3,), (pitch,yaw,roll))."""
    if isinstance(pose, dict):
        if "location_cm" in pose:
            loc = np.asarray(pose["location_cm"], float).reshape(3)
        else:
            loc = np.array([pose["x"], pose["y"], pose["z"]], float)
        if "rotation_deg" in pose:
            rot = np.asarray(pose["rotation_deg"], float).reshape(3)
        else:
            rot = np.array([pose.get("pitch", 0.0), pose.get("yaw", 0.0),
                            pose.get("roll", 0.0)], float)
        return loc, rot
    arr = list(pose)
    if len(arr) == 2:
        return np.asarray(arr[0], float).reshape(3), np.asarray(arr[1], float).reshape(3)
    if len(arr) == 6:
        return np.asarray(arr[:3], float), np.asarray(arr[3:], float)
    raise ValueError(f"cannot interpret camera pose {pose!r}")


def vfov_deg(fov_h_deg: float, width: int, height: int) -> float:
    """Vertical FOV implied by a horizontal FOV and square pixels."""
    return float(np.rad2deg(2.0 * np.arctan(np.tan(np.deg2rad(fov_h_deg) / 2.0) * height / width)))


def pixel_dirs(u, v, width: int, height: int, fov_h_deg: float, basis: np.ndarray) -> np.ndarray:
    """Pixel coordinates -> unit ray directions in world axes.  See module docstring.

    `u`, `v` may be floats (the continuous image plane): u in [-0.5, W-0.5],
    v in [-0.5, H-0.5], pixel centres at integers, top-left origin.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    su = (u + 0.5 - width / 2.0) / (width / 2.0)
    sv = (v + 0.5 - height / 2.0) / (height / 2.0)
    th = np.tan(np.deg2rad(fov_h_deg) / 2.0)
    tv = th * height / width
    fwd, right, up = basis[0], basis[1], basis[2]
    d = fwd + right * (su * th)[..., None] - up * (sv * tv)[..., None]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


# ============================================================================ the map
class HealthMap:
    """Running 2.5D health map over a DEM grid.  Pure numpy."""

    # ------------------------------------------------------------------ construction
    def __init__(self, dem, max_range_m: float = DEFAULT_MAX_RANGE_M,
                 step_cm: Optional[float] = None, min_range_m: float = 0.05):
        d = np.load(dem, allow_pickle=False) if isinstance(dem, (str, os.PathLike)) else dem
        self.cell_cm = float(np.asarray(d["cell_cm"]).reshape(-1)[0])
        self.origin_cm = np.asarray(d["origin_cm"], float).reshape(2)
        self.nx = int(np.asarray(d["nx"]).reshape(-1)[0])
        self.ny = int(np.asarray(d["ny"]).reshape(-1)[0])
        self.height = np.asarray(d["height"], np.float32).reshape(self.nx, self.ny)
        if "height_filled" in getattr(d, "files", d):
            self.height_filled = np.asarray(d["height_filled"], np.float32)
        else:
            self.height_filled = np.where(np.isfinite(self.height), self.height,
                                          np.nanmin(self.height)).astype(np.float32)
        self.height_filled = self.height_filled.reshape(self.nx, self.ny)
        self.colour = (np.asarray(d["colour"], np.float32).reshape(self.nx, self.ny, 3)
                       if "colour" in getattr(d, "files", d) else
                       np.full((self.nx, self.ny, 3), 0.5, np.float32))
        self.count = (np.asarray(d["count"]).reshape(self.nx, self.ny)
                      if "count" in getattr(d, "files", d) else
                      np.ones((self.nx, self.ny), np.int32))
        self.dem_meta = json.loads(str(d["meta"])) if "meta" in getattr(d, "files", d) else {}

        self.max_range_m = float(max_range_m)
        self.min_range_m = float(min_range_m)
        self.step_cm = float(step_cm) if step_cm else self.cell_cm

        self.votes = np.zeros((self.nx, self.ny, NUM_CLASSES), np.float32)
        self.obs = np.zeros((self.nx, self.ny), np.int32)
        self.frames = 0
        self.rays_cast = 0
        self.rays_hit = 0

    # ------------------------------------------------------------------ grid helpers
    @property
    def cell_m(self) -> float:
        return self.cell_cm / 100.0

    @property
    def cell_area_m2(self) -> float:
        return (self.cell_cm / 100.0) ** 2

    def index(self, wx, wy):
        ix = np.floor((np.asarray(wx) - self.origin_cm[0]) / self.cell_cm).astype(np.int64)
        iy = np.floor((np.asarray(wy) - self.origin_cm[1]) / self.cell_cm).astype(np.int64)
        return ix, iy

    def inside(self, ix, iy):
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    def cell_centre(self, ix, iy):
        return (self.origin_cm[0] + (np.asarray(ix) + 0.5) * self.cell_cm,
                self.origin_cm[1] + (np.asarray(iy) + 0.5) * self.cell_cm)

    def height_at(self, wx, wy):
        """Surface height (world cm) at world (x, y); NaN outside the grid."""
        ix, iy = self.index(wx, wy)
        ok = self.inside(ix, iy)
        out = np.full(np.shape(ix), np.nan, np.float32)
        out[ok] = self.height_filled[ix[ok], iy[ok]]
        return out

    # ------------------------------------------------------------------ ray casting
    def raycast(self, origin_cm, dirs, max_range_m: Optional[float] = None):
        """March unit rays from one origin against the height field.

        Returns (hit (N,) bool, ix (N,), iy (N,), range_m (N,), hit_xyz (N,3) world cm).
        `ix/iy` are -1 where there is no hit.

        Three details that are easy to get wrong (see the module docstring):

        * the reported CELL is the cell of the interpolated crossing, EXCEPT where the sample
          that went under stands on a surface above the hit point (a wall), in which case it
          is the sample's cell - otherwise a ray crossing a raised face would vote the floor
          cell in front of the wall, and every flat-ground ray would vote up to one step too
          far along;
        * the reported RANGE is still the interpolated one, so the range weight stays
          sub-cell accurate;
        * the grid is convex: once a ray has been inside it and steps out, it is dead.  A ray
          that starts outside and enters below the surface hits at its first inside sample
          (there is nothing to interpolate back to, and interpolating back would put the hit
          point outside the grid).
        """
        o = np.asarray(origin_cm, float).reshape(3)
        d = np.asarray(dirs, float).reshape(-1, 3)
        n = len(d)
        rmax_cm = (self.max_range_m if max_range_m is None else float(max_range_m)) * 100.0
        step = self.step_cm
        nsteps = max(int(np.ceil(rmax_cm / step)), 1)

        hit = np.zeros(n, bool)
        t_hit = np.full(n, np.nan)
        cix = np.full(n, -1, np.int64)      # cell of the sample that went under
        ciy = np.full(n, -1, np.int64)
        prev_t = np.zeros(n)
        prev_diff = np.full(n, np.nan)      # p.z - surface at the previous sample
        alive = np.ones(n, bool)
        entered = np.zeros(n, bool)         # has this ray ever been inside the grid?

        for s in range(1, nsteps + 1):
            t = min(s * step, rmax_cm)
            p = o[None, :] + d * t
            ix, iy = self.index(p[:, 0], p[:, 1])
            ins = self.inside(ix, iy)
            ok = alive & ins
            alive &= ~(~ins & entered)      # convex grid: left it once -> it never comes back
            entered |= ok
            diff = np.full(n, np.nan)
            if ok.any():
                diff[ok] = p[ok, 2] - self.height_filled[ix[ok], iy[ok]]
            new = ok & (diff <= 0.0)
            if new.any():
                pd = prev_diff[new]         # NaN unless the PREVIOUS sample was inside too
                # pd > 0 strictly wherever it is finite (a pd <= 0 would have hit on that
                # step) and diff <= 0 here, so the denominator is positive; the clamp only
                # guards the degenerate 0/0 and the clip keeps the result a step fraction.
                den = np.maximum(pd - diff[new], 1e-12)
                frac = np.where(np.isfinite(pd), pd / den,
                                1.0)        # no inside predecessor -> hit AT this sample
                frac = np.clip(frac, 0.0, 1.0)
                t_new = prev_t[new] + frac * (t - prev_t[new])
                t_hit[new] = t_new
                # ---- which CELL gets the vote (the hybrid rule, see the docstring) -------
                # The interpolated crossing is the better estimate on open ground (the sample
                # is up to a whole step beyond it); the sample is the better estimate at a
                # wall, which is exactly where the sample cell's surface is above the hit.
                hp = o[None, :] + d[new] * t_new[:, None]
                jx, jy = self.index(hp[:, 0], hp[:, 1])
                wall = self.height_filled[ix[new], iy[new]] > hp[:, 2] + WALL_TOL_CM
                use_crossing = self.inside(jx, jy) & ~wall
                cix[new] = np.where(use_crossing, jx, ix[new])
                ciy[new] = np.where(use_crossing, jy, iy[new])
                hit[new] = True
                alive[new] = False
            prev_t = np.where(alive, t, prev_t)
            prev_diff = np.where(alive, diff, prev_diff)
            if not alive.any():
                break

        hxyz = np.full((n, 3), np.nan)
        if hit.any():
            hp = o[None, :] + d[hit] * t_hit[hit][:, None]
            hxyz[hit] = hp
            # Both interpolation end-points are inside the (convex) grid, so this cannot fire;
            # it is a guard that DROPS a stray hit rather than clipping it into an edge cell.
            a, b = self.index(hp[:, 0], hp[:, 1])
            bad = ~self.inside(a, b)
            if bad.any():
                idx = np.nonzero(hit)[0][bad]
                hit[idx] = False
                t_hit[idx] = np.nan
                hxyz[idx] = np.nan
                cix[idx] = -1
                ciy[idx] = -1
        rng = t_hit / 100.0
        return hit, cix, ciy, rng, hxyz

    # ------------------------------------------------------------------ projection
    def project_mask(self, mask_hw, camera_pose_world, fov_h_deg: float = DEFAULT_FOV_H_DEG,
                     subsample: int = 4, max_range_m: Optional[float] = None):
        """Mask + pose -> the cells each sampled pixel sees.

        Returns dict with `ix`, `iy`, `cls`, `range_m`, `weight`, `hit`, `u`, `v`, `dirs`,
        `loc_cm`, `basis`.  Only entries with `hit` are meaningful.
        """
        mask = np.asarray(mask_hw)
        if mask.ndim != 2:
            raise ValueError(f"mask must be (H, W); got {mask.shape}")
        H, W = mask.shape
        loc, rot = parse_pose(camera_pose_world)
        basis = rotator_basis(rot[0], rot[1], rot[2])

        ss = max(int(subsample), 1)
        vs = np.arange(ss // 2, H, ss)
        us = np.arange(ss // 2, W, ss)
        uu, vv = np.meshgrid(us, vs, indexing="xy")
        u = uu.ravel()
        v = vv.ravel()
        dirs = pixel_dirs(u, v, W, H, fov_h_deg, basis)
        hit, ix, iy, rng, _ = self.raycast(loc, dirs, max_range_m=max_range_m)

        cls = mask[v, u]
        rmax = self.max_range_m if max_range_m is None else float(max_range_m)
        good = (hit & np.isfinite(rng) & (rng <= rmax) & (rng >= self.min_range_m)
                & (cls < NUM_CLASSES))
        w = np.where(good, 1.0 / (1.0 + np.maximum(rng, 0.0)), 0.0)
        return dict(ix=ix, iy=iy, cls=cls.astype(np.int64), range_m=rng, weight=w,
                    hit=good, u=u, v=v, dirs=dirs, loc_cm=loc, basis=basis, W=W, H=H)

    # ------------------------------------------------------------------ the update
    def update(self, mask_hw, camera_pose_world, fov_h_deg: float = DEFAULT_FOV_H_DEG,
               subsample: int = 4, reliable_only: bool = False):
        """Accumulate one segmentation frame.  Returns the projection dict."""
        rmax = RELIABLE_RANGE_M if reliable_only else None
        pr = self.project_mask(mask_hw, camera_pose_world, fov_h_deg=fov_h_deg,
                               subsample=subsample, max_range_m=rmax)
        g = pr["hit"]
        self.frames += 1
        self.rays_cast += len(g)
        self.rays_hit += int(g.sum())
        if g.any():
            flat = (pr["ix"][g] * self.ny + pr["iy"][g]).astype(np.int64)
            cls = pr["cls"][g]
            w = pr["weight"][g].astype(np.float64)
            nc = self.nx * self.ny
            acc = np.bincount(flat + cls * nc, weights=w, minlength=NUM_CLASSES * nc)
            self.votes += acc.reshape(NUM_CLASSES, self.nx, self.ny).transpose(1, 2, 0).astype(np.float32)
            self.obs += np.bincount(flat, minlength=nc).reshape(self.nx, self.ny).astype(np.int32)
        return pr

    # ------------------------------------------------------------------ readouts
    def total_votes(self) -> np.ndarray:
        return self.votes.sum(axis=2)

    def surveyed(self, *, min_obs: int = DEFAULT_MIN_OBS,
                 min_votes: float = DEFAULT_MIN_VOTES) -> np.ndarray:
        """Cells with at least `min_obs` observations (rays) - and optionally `min_votes` weight.

        The primary gate is the OBSERVATION count because vote weight falls off with range
        (`w = 1/(1+range_m)`), so a weight threshold silently becomes a range threshold.

        KEYWORD-ONLY on purpose.  `min_obs` took the first positional slot when the gate moved
        from weight to observations, so an old caller still writing `surveyed(1.0)` (or
        `class_map(0.01)`) would have had its weight threshold silently truncated to an
        observation count of 1 - or of 0, surveying the whole grid.  A `*` turns that into a
        TypeError.
        """
        m = self.obs >= int(min_obs)
        if min_votes > 0.0:
            m &= self.total_votes() >= float(min_votes)
        return m

    def class_map(self, *, min_obs: int = DEFAULT_MIN_OBS,
                  min_votes: float = DEFAULT_MIN_VOTES) -> np.ndarray:
        """(nx, ny) int8: the winning class per cell, -1 where under-observed.

        Exact ties go to BLEACHED, then DEGRADED, then OTHER, then LIVE (`TIE_BREAK_ORDER`):
        a survey should over-report damage, and a plain `argmax` would give every exact tie
        to LIVE only because LIVE is class id 0.
        """
        out = np.full((self.nx, self.ny), -1, np.int8)
        m = self.surveyed(min_obs=min_obs, min_votes=min_votes)
        if m.any():
            win = np.argmax(self.votes[m][:, _TIE_ORDER], axis=1)
            out[m] = _TIE_ORDER[win].astype(np.int8)
        return out

    def class_prob(self, *, min_obs: int = DEFAULT_MIN_OBS,
                   min_votes: float = DEFAULT_MIN_VOTES) -> np.ndarray:
        """(nx, ny, 4) vote share (votes / total votes) per class, 0 where under-observed.

        The share of the WINNING class is the confidence of `class_map`: 0.25 is a four-way
        tie, 1.0 is unanimous.  Report it next to the class, never the class alone.
        """
        tot = self.total_votes()
        out = np.zeros_like(self.votes)
        m = self.surveyed(min_obs=min_obs, min_votes=min_votes) & (tot > 0)
        out[m] = self.votes[m] / tot[m][:, None]
        return out

    def winning_prob(self, *, min_obs: int = DEFAULT_MIN_OBS,
                     min_votes: float = DEFAULT_MIN_VOTES) -> np.ndarray:
        """(nx, ny) float32 vote share of the class `class_map` reports, 0 where unobserved."""
        return self.class_prob(min_obs=min_obs, min_votes=min_votes).max(axis=2)

    def bleached_fraction(self, *, min_obs: int = DEFAULT_MIN_OBS,
                          min_votes: float = DEFAULT_MIN_VOTES) -> float:
        cm = self.class_map(min_obs=min_obs, min_votes=min_votes)
        s = cm >= 0
        return float((cm[s] == BLEACHED).mean()) if s.any() else 0.0

    def surveyed_area_m2(self, *, min_obs: int = DEFAULT_MIN_OBS,
                         min_votes: float = DEFAULT_MIN_VOTES) -> float:
        return float(self.surveyed(min_obs=min_obs,
                                   min_votes=min_votes).sum()) * self.cell_area_m2

    def bleached_area_m2(self, *, min_obs: int = DEFAULT_MIN_OBS,
                         min_votes: float = DEFAULT_MIN_VOTES) -> float:
        cm = self.class_map(min_obs=min_obs, min_votes=min_votes)
        return float((cm == BLEACHED).sum()) * self.cell_area_m2

    def coverage_fraction(self, *, min_obs: int = DEFAULT_MIN_OBS,
                          min_votes: float = DEFAULT_MIN_VOTES) -> float:
        """Surveyed cells as a fraction of the DEM cells that hold any reef."""
        filled = np.isfinite(self.height)
        return float((self.surveyed(min_obs=min_obs, min_votes=min_votes) & filled).sum()
                     / max(int(filled.sum()), 1))

    # ------------------------------------------------------------------ metrics
    def metrics_vs_gt(self, gt, *, min_obs: int = DEFAULT_MIN_OBS,
                      min_votes: float = DEFAULT_MIN_VOTES, target=BLEACHED) -> dict:
        """IoU / precision / recall for BLEACHED over SURVEYED cells only.

        `target` is a class id, or a SEQUENCE of class ids to be scored as one merged
        positive class (`target=(BLEACHED, DEGRADED)` answers "did the AUV see *damage*",
        test-plan item 0.2).  The merged run reports `class_name` as `BLEACHED|DEGRADED`
        and is otherwise identical - the ground truth raster is unchanged, only the
        prediction side is widened.
        """
        if isinstance(gt, (str, os.PathLike)):
            gd = np.load(gt, allow_pickle=False)
            G = np.asarray(gd["bleached"], bool).reshape(self.nx, self.ny)
        elif isinstance(gt, np.ndarray):
            G = np.asarray(gt, bool).reshape(self.nx, self.ny)
        else:
            G = np.asarray(gt["bleached"], bool).reshape(self.nx, self.ny)

        targets = ([int(target)] if isinstance(target, (int, np.integer))
                   else [int(t) for t in target])
        name = "|".join(CLASS_NAMES[t] for t in targets)

        cm = self.class_map(min_obs=min_obs, min_votes=min_votes)
        s = cm >= 0
        P = np.isin(cm, targets) & s
        Gs = G & s
        tp = int((P & Gs).sum())
        fp = int((P & ~Gs).sum())
        fn = int((~P & Gs).sum())
        union = tp + fp + fn
        n_s = int(s.sum())
        return dict(
            class_name=name, target=targets,
            min_obs=int(min_obs), min_votes=float(min_votes),
            surveyed_cells=n_s, surveyed_area_m2=round(n_s * self.cell_area_m2, 4),
            tp=tp, fp=fp, fn=fn,
            iou=(tp / union) if union else 0.0,
            precision=(tp / (tp + fp)) if (tp + fp) else 0.0,
            recall=(tp / (tp + fn)) if (tp + fn) else 0.0,
            f1=(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
            pred_bleached_cells=int(P.sum()),      # "cells predicted positive" (legacy key)
            pred_cells=int(P.sum()),
            gt_bleached_cells_surveyed=int(Gs.sum()),
            gt_bleached_cells_total=int(G.sum()),
            pred_percent_of_surveyed=100.0 * P.sum() / max(n_s, 1),
            gt_percent_of_surveyed=100.0 * Gs.sum() / max(n_s, 1),
            pred_area_m2=round(float(P.sum()) * self.cell_area_m2, 4),
            gt_area_m2_surveyed=round(float(Gs.sum()) * self.cell_area_m2, 4),
            gt_recall_of_whole_site=float(Gs.sum() / max(int(G.sum()), 1)),
        )

    # ------------------------------------------------------------------ minimap
    def render_minimap(self, size_px: int = 512, agent_pose=None, palette=None, *,
                       min_votes: float = DEFAULT_MIN_VOTES, alpha: float = 0.72,
                       gt=None, show_path=None, min_obs: int = DEFAULT_MIN_OBS) -> np.ndarray:
        """North-up RGB uint8 minimap: ortho colour background + health over surveyed cells.

        `agent_pose` (any form `parse_pose` accepts) draws a heading triangle.
        `show_path` is an optional (K,2) array of world (x, y) cm already travelled.
        `gt` (bool grid) is outlined in magenta when given.
        """
        pal = PALETTE if palette is None else np.asarray(palette, np.uint8)
        bg = np.clip(self.colour, 0, 1).copy()
        bg[np.asarray(self.count) == 0] = 0.10
        bg = 0.55 * bg + 0.45 * 0.25                     # darken so the overlay reads

        cm = self.class_map(min_obs=min_obs, min_votes=min_votes)
        img = (bg * 255.0).astype(np.float32)
        s = cm >= 0
        if s.any():
            col = pal[np.clip(cm, 0, NUM_CLASSES - 1)].astype(np.float32)
            img[s] = (1 - alpha) * img[s] + alpha * col[s]

        if gt is not None:
            G = np.asarray(gt["bleached"] if not isinstance(gt, np.ndarray) else gt, bool)
            edge = G & ~(np.roll(G, 1, 0) & np.roll(G, -1, 0) & np.roll(G, 1, 1) & np.roll(G, -1, 1))
            img[edge] = np.array([255, 0, 170], np.float32)

        img = np.clip(img, 0, 255).astype(np.uint8)
        from_ = _north_up(img)

        # nearest-neighbour scale to size_px (square canvas, letterboxed)
        H0, W0 = from_.shape[:2]
        scale = size_px / max(H0, W0)
        out = np.full((size_px, size_px, 3), 18, np.uint8)
        h1, w1 = max(int(round(H0 * scale)), 1), max(int(round(W0 * scale)), 1)
        ri = np.clip((np.arange(h1) / scale).astype(int), 0, H0 - 1)
        ci = np.clip((np.arange(w1) / scale).astype(int), 0, W0 - 1)
        r0 = (size_px - h1) // 2
        c0 = (size_px - w1) // 2
        out[r0:r0 + h1, c0:c0 + w1] = from_[np.ix_(ri, ci)]

        def to_px(wx, wy):
            ix = (np.asarray(wx) - self.origin_cm[0]) / self.cell_cm
            iy = (np.asarray(wy) - self.origin_cm[1]) / self.cell_cm
            row = r0 + (self.nx - ix) * scale
            col = c0 + iy * scale
            return col, row

        if show_path is not None and len(show_path):
            p = np.asarray(show_path, float).reshape(-1, 2)
            cx, cy = to_px(p[:, 0], p[:, 1])
            for x, y in zip(cx, cy):
                rr, cc = int(round(y)), int(round(x))
                if 0 <= rr < size_px and 0 <= cc < size_px:
                    out[rr, cc] = (255, 215, 0)

        if agent_pose is not None:
            loc, rot = parse_pose(agent_pose)
            _draw_agent(out, to_px(loc[0], loc[1]), rot[1], scale)

        # north arrow
        _draw_north(out)
        return out

    def summary(self, *, min_obs: int = DEFAULT_MIN_OBS,
                min_votes: float = DEFAULT_MIN_VOTES) -> dict:
        gate = dict(min_obs=min_obs, min_votes=min_votes)
        s = self.surveyed(**gate)
        return dict(frames=self.frames, rays_cast=self.rays_cast, rays_hit=self.rays_hit,
                    hit_rate=self.rays_hit / max(self.rays_cast, 1),
                    min_obs=int(min_obs), min_votes=float(min_votes),
                    surveyed_cells=int(s.sum()),
                    surveyed_area_m2=round(self.surveyed_area_m2(**gate), 4),
                    coverage_fraction=round(self.coverage_fraction(**gate), 5),
                    bleached_fraction=round(self.bleached_fraction(**gate), 5),
                    bleached_area_m2=round(self.bleached_area_m2(**gate), 4),
                    mean_obs_surveyed=round(float(self.obs[s].mean()), 2) if s.any() else 0.0,
                    mean_winning_prob=round(float(self.winning_prob(**gate)[s].mean()), 4)
                    if s.any() else 0.0,
                    cell_cm=self.cell_cm, max_range_m=self.max_range_m)

    def __repr__(self):
        return (f"HealthMap({self.nx}x{self.ny} @ {self.cell_cm:g} cm, "
                f"frames={self.frames}, surveyed={int(self.surveyed().sum())})")


# ============================================================================ helpers
def _north_up(a: np.ndarray) -> np.ndarray:
    return np.asarray(a)[::-1, ...]


def _draw_agent(img, colrow, yaw_deg: float, scale: float) -> None:
    """Filled heading triangle at (col,row); yaw 0 = world +X = image UP."""
    col, row = float(colrow[0]), float(colrow[1])
    size = max(7.0, 9.0 * max(scale, 0.25))
    a = np.deg2rad(yaw_deg)
    # world +X is image -row, world +Y is image +col
    fwd = np.array([np.sin(a), -np.cos(a)])          # (dcol, drow)
    side = np.array([np.cos(a), np.sin(a)])
    p0 = np.array([col, row]) + fwd * size
    p1 = np.array([col, row]) - fwd * size * 0.6 + side * size * 0.6
    p2 = np.array([col, row]) - fwd * size * 0.6 - side * size * 0.6
    _fill_tri(img, p0, p1, p2, np.array([255, 40, 40], np.uint8))
    _fill_tri(img, p0 * 0.72 + np.array([col, row]) * 0.28,
              p1 * 0.72 + np.array([col, row]) * 0.28,
              p2 * 0.72 + np.array([col, row]) * 0.28, np.array([255, 235, 60], np.uint8))


def _fill_tri(img, p0, p1, p2, colour) -> None:
    pts = np.stack([p0, p1, p2])
    c0, c1 = int(np.floor(pts[:, 0].min())), int(np.ceil(pts[:, 0].max())) + 1
    r0, r1 = int(np.floor(pts[:, 1].min())), int(np.ceil(pts[:, 1].max())) + 1
    H, W = img.shape[:2]
    c0, c1 = max(c0, 0), min(c1, W)
    r0, r1 = max(r0, 0), min(r1, H)
    if c1 <= c0 or r1 <= r0:
        return
    cc, rr = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5)

    def side(a, b):
        return (cc - a[0]) * (b[1] - a[1]) - (rr - a[1]) * (b[0] - a[0])

    s0, s1, s2 = side(p0, p1), side(p1, p2), side(p2, p0)
    inside = ((s0 >= 0) & (s1 >= 0) & (s2 >= 0)) | ((s0 <= 0) & (s1 <= 0) & (s2 <= 0))
    block = img[r0:r1, c0:c1]
    block[inside] = colour
    img[r0:r1, c0:c1] = block


def _draw_north(img) -> None:
    """Small 'N' tick in the top-left corner (up = +X world)."""
    H = img.shape[0]
    m = max(6, H // 60)
    img[m:m * 4, m * 2:m * 2 + 2] = (255, 255, 255)
    img[m:m + 2, m * 2 - m // 2:m * 2 + m // 2 + 2] = (255, 255, 255)
