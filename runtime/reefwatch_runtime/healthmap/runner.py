"""The live health map: one `engine.HealthMap` fed one segmentation frame at a time.

Ported from CoralUnreal `scripts/runtime/healthmap_runtime.py` (logic unchanged). Protocol-
agnostic: the server owns the sockets, this owns the map.

  * `update()` one mask of CLASS IDS (health ids 0..3) with the camera pose, under a lock;
  * `metrics()` - the MAP_METRICS JSON dict (docs/PROTOCOL.md, block 4);
  * `minimap_png()` - the north-up minimap, RGBA with alpha 0 on unsurveyed cells, so one
    image serves as the HUD inset and as the world-overlay texture (AHealthMapOverlayActor);
  * `save_artifacts()` - `healthmap_state.npz` + `class_map.png` + `minimap_final.png` +
    `metrics.json`, all from ONE snapshot so the files describe the same instant.

Thread-safety: `update` / `reset` / `snapshot` take `self.lock`; every read-out works on a
snapshot copy outside the lock. `save_artifacts` also takes `self.save_lock` so the autosave
thread and the disconnect/shutdown saves never write the same files at once.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time

import numpy as np

from .. import config
from . import engine as E
from .png import encode_png_zlib

DEFAULT_DEM = str(config.DEFAULT_DEM)
DEFAULT_GT = str(config.DEFAULT_GT)


# --------------------------------------------------------------------------- image helpers
def _letterbox_indices(nx, ny, size_px):
    """The EXACT placement `engine.HealthMap.render_minimap` uses for its square canvas.

    render_minimap north-ups the (nx, ny) grid, nearest-neighbour scales it by
    `size_px / max(nx, ny)` and centres it in a size_px square.  Reproduced here (and pinned
    by `runtime/tests/test_healthmap.py`) so the alpha channel
    lands on exactly the same pixels as the colours it belongs to.
    """
    H0, W0 = int(nx), int(ny)
    scale = size_px / max(H0, W0)
    h1 = max(int(round(H0 * scale)), 1)
    w1 = max(int(round(W0 * scale)), 1)
    ri = np.clip((np.arange(h1) / scale).astype(int), 0, H0 - 1)
    ci = np.clip((np.arange(w1) / scale).astype(int), 0, W0 - 1)
    r0 = (size_px - h1) // 2
    c0 = (size_px - w1) // 2
    return r0, c0, ri, ci


def letterbox_grid(grid, size_px, fill=0):
    """North-up + nearest-neighbour scale a (nx, ny[, C]) grid into a size_px square."""
    a = np.asarray(grid)
    up = a[::-1, ...]
    r0, c0, ri, ci = _letterbox_indices(a.shape[0], a.shape[1], size_px)
    shape = (size_px, size_px) + a.shape[2:]
    out = np.full(shape, fill, a.dtype)
    out[r0:r0 + len(ri), c0:c0 + len(ci)] = up[np.ix_(ri, ci)]
    return out


def encode_png(rgb_or_rgba: np.ndarray) -> bytes:
    """Pillow when available; otherwise the zlib encoder (the board may only have numpy)."""
    try:
        from PIL import Image
    except ImportError:
        return encode_png_zlib(rgb_or_rgba)
    a = np.ascontiguousarray(rgb_or_rgba)
    mode = "RGBA" if a.shape[-1] == 4 else "RGB"
    bio = io.BytesIO()
    Image.fromarray(a, mode).save(bio, format="PNG", compress_level=1)
    return bio.getvalue()


# --------------------------------------------------------------------------- the runner
class MapRunner:
    """One live 2.5D health map, fed one segmentation frame at a time."""

    def __init__(self, dem=DEFAULT_DEM, gt=None, max_range_m=E.DEFAULT_MAX_RANGE_M,
                 subsample=4, min_obs=E.DEFAULT_MIN_OBS, min_votes=E.DEFAULT_MIN_VOTES,
                 minimap_px=512, reliable_only=False, minimap_min_interval_s=0.2):
        self.dem_path = str(dem) if isinstance(dem, (str, os.PathLike)) else "<array>"
        self.hm = E.HealthMap(dem, max_range_m=max_range_m)
        self.subsample = int(subsample)
        # The read-out gate is the engine's, not ours: `min_obs` rays (the primary gate - vote
        # weight falls off with range, so a weight threshold is a range threshold in disguise)
        # and `min_votes` as an optional extra. ALWAYS passed by KEYWORD (`self.gate`): the
        # engine's read-outs take both keyword-only, precisely so a positional float cannot
        # silently become an observation count.
        self.min_obs = int(min_obs)
        self.min_votes = float(min_votes)
        self.minimap_px = int(minimap_px)
        self.reliable_only = bool(reliable_only)
        self.minimap_min_interval_s = float(minimap_min_interval_s)

        self.gt = None
        self.gt_path = None
        if gt is not None:
            if isinstance(gt, (str, os.PathLike)):
                self.gt_path = str(gt)
                g = np.load(gt, allow_pickle=False)
                self.gt = np.asarray(g["bleached"], bool).reshape(self.hm.nx, self.hm.ny)
            else:
                self.gt = np.asarray(gt, bool).reshape(self.hm.nx, self.hm.ny)

        self.lock = threading.Lock()
        self.save_lock = threading.Lock()      # serialises save_artifacts across threads
        self.last_pose = None
        self.last_cells_updated = 0
        self.last_rays_cast = 0
        self.last_rays_hit = 0
        self.last_minimap_time = 0.0
        self.minimaps_sent = 0
        self.resets = 0
        self.path_cm = []           # travelled world (x, y) cm, one entry per accepted pose

        # constant part of MAP_METRICS: everything AHealthMapOverlayActor needs to size itself
        filled = np.isfinite(self.hm.height)
        self.grid_info = dict(
            nx=int(self.hm.nx), ny=int(self.hm.ny), cell_cm=float(self.hm.cell_cm),
            origin_cm=[float(self.hm.origin_cm[0]), float(self.hm.origin_cm[1])],
            mean_height_cm=float(np.nanmean(self.hm.height)) if filled.any() else 0.0,
            reef_cells=int(filled.sum()),
        )

    # ------------------------------------------------------------------ mutation
    def reset(self):
        with self.lock:
            self.hm.votes[:] = 0.0
            self.hm.obs[:] = 0
            self.hm.frames = 0
            self.hm.rays_cast = 0
            self.hm.rays_hit = 0
            self.path_cm = []
            self.last_cells_updated = 0
            self.last_rays_cast = 0
            self.last_rays_hit = 0
            self.resets += 1

    def update(self, mask_ids, pose, fov_h_deg=E.DEFAULT_FOV_H_DEG):
        """Accumulate one frame.  `mask_ids` is (H, W) uint8 class ids, NOT the palette mask."""
        with self.lock:
            pr = self.hm.update(mask_ids, pose, fov_h_deg=fov_h_deg,
                                subsample=self.subsample, reliable_only=self.reliable_only)
            g = pr["hit"]
            self.last_rays_cast = int(len(g))
            self.last_rays_hit = int(g.sum())
            if self.last_rays_hit:
                flat = pr["ix"][g] * self.hm.ny + pr["iy"][g]
                self.last_cells_updated = int(np.unique(flat).size)
            else:
                self.last_cells_updated = 0
            loc, _ = E.parse_pose(pose)
            self.last_pose = pose
            self.path_cm.append((float(loc[0]), float(loc[1])))
            if len(self.path_cm) > 20000:
                self.path_cm = self.path_cm[-20000:]
        return pr

    # ------------------------------------------------------------------ read-out
    @property
    def gate(self) -> dict:
        """The read-out gate, ALWAYS splatted as keywords - the engine's are keyword-only."""
        return dict(min_obs=self.min_obs, min_votes=self.min_votes)

    def metrics(self, frame_id=0, map_ms=0.0, with_gt=True, snap=None) -> dict:
        """The MAP_METRICS dict (PROTOCOL.md v2, block 4), from the ENGINE's own read-outs.

        `surveyed()`, `class_map()` and `winning_prob()` carry the observation gate and the
        BLEACHED-first tie-break, so every number here (and the alpha mask the world overlay
        is textured with) uses exactly the gate `engine.summary()` would.  The per-class areas
        are counted from that single `class_map()` rather than by calling
        `bleached_area_m2()` / `bleached_fraction()` / ... in turn, because each of those
        recomputes the whole class map and this runs inside a 15 Hz loop;
        the two are pinned together by `runtime/tests/test_healthmap.py`.

        The read itself is a `snapshot()` (taken under the lock, ~2 MB) and the arithmetic
        then runs OUTSIDE the lock, so the serve thread is never blocked by it and the numbers
        cannot be torn across a concurrent `update()`.  Pass `snap` to describe the same
        instant as another read-out - `save_artifacts` does this for the whole file set.
        """
        snap = self.snapshot() if snap is None else snap
        hm = self._frozen(snap)
        gate = self.gate
        surveyed = hm.surveyed(**gate)
        n_s = int(surveyed.sum())
        cls = hm.class_map(**gate)
        counts = [int((cls == c).sum()) for c in range(E.NUM_CLASSES)]
        cov = hm.coverage_fraction(**gate)
        mean_obs = float(hm.obs[surveyed].mean()) if n_s else 0.0
        mean_win = float(hm.winning_prob(**gate)[surveyed].mean()) if n_s else 0.0
        frames = int(hm.frames)
        pose = snap["last_pose"]
        cells_updated = snap["last_cells_updated"]
        rays_cast, rays_hit = snap["last_rays_cast"], snap["last_rays_hit"]
        gt_stats = None
        if with_gt and self.gt is not None:
            g = hm.metrics_vs_gt(self.gt, **gate)
            gt_stats = dict(
                iou=round(g["iou"], 5), precision=round(g["precision"], 5),
                recall=round(g["recall"], 5), f1=round(g["f1"], 5),
                tp=g["tp"], fp=g["fp"], fn=g["fn"],
                gt_area_m2_surveyed=g["gt_area_m2_surveyed"],
                gt_area_m2_total=round(g["gt_bleached_cells_total"] * hm.cell_area_m2, 4),
            )

        area = hm.cell_area_m2
        agent = [-1, -1]
        if pose is not None:
            loc, _ = E.parse_pose(pose)
            ix, iy = hm.index(loc[0], loc[1])
            agent = [int(ix), int(iy)]
        out = dict(
            frame_id=int(frame_id), frames=frames,
            cell_cm=float(hm.cell_cm),
            min_obs=int(self.min_obs), min_votes=float(self.min_votes),
            surveyed_cells=n_s, surveyed_area_m2=round(n_s * area, 4),
            coverage_fraction=round(cov, 5),
            bleached_fraction=round(counts[E.BLEACHED] / max(n_s, 1), 5),
            bleached_area_m2=round(counts[E.BLEACHED] * area, 4),
            class_area_m2=[round(c * area, 4) for c in counts],
            mean_obs_surveyed=round(mean_obs, 2),
            mean_winning_prob=round(mean_win, 4),
            cells_updated=int(cells_updated),
            rays_cast=int(rays_cast), rays_hit=int(rays_hit),
            hit_rate=round(rays_hit / max(rays_cast, 1), 4),
            agent_cell=agent, map_ms=round(float(map_ms), 3),
            grid=self.grid_info,
        )
        if gt_stats is not None:
            out["gt"] = gt_stats
        return out

    def minimap_rgba(self, size_px=None, show_path=True, snap=None) -> np.ndarray:
        """North-up minimap, RGBA.  Alpha is 255 on surveyed cells and 0 everywhere else.

        The alpha mask comes from the engine's own `surveyed()` - the same gate that decided
        which cells `render_minimap` coloured - so the world overlay is transparent exactly
        where the minimap is unpainted.  Rendered from a snapshot, off the lock: the render is
        20-40 ms and used to hold the serve thread out of `update()` for all of it.
        """
        size_px = int(size_px or self.minimap_px)
        gate = self.gate
        snap = self.snapshot() if snap is None else snap
        hm = self._frozen(snap)
        pose = snap["last_pose"]
        path = np.asarray(snap["path_cm"], float) if (show_path and snap["path_cm"]) else None
        rgb = hm.render_minimap(size_px, agent_pose=pose, gt=self.gt, show_path=path, **gate)
        surveyed = hm.surveyed(**gate)
        alpha = letterbox_grid(surveyed.astype(np.uint8) * 255, size_px, fill=0)
        return np.dstack([rgb, alpha.astype(np.uint8)])

    def minimap_png(self, size_px=None, show_path=True, snap=None) -> bytes:
        return encode_png(self.minimap_rgba(size_px=size_px, show_path=show_path, snap=snap))

    def minimap_due(self, now=None) -> bool:
        """Server-side 5 Hz cap, independent of how often the client asks."""
        now = time.perf_counter() if now is None else now
        return (now - self.last_minimap_time) >= self.minimap_min_interval_s

    def note_minimap_sent(self, now=None):
        self.last_minimap_time = time.perf_counter() if now is None else now
        self.minimaps_sent += 1

    def class_map_rgb(self, size_px=None, snap=None) -> np.ndarray:
        """Palette class map on a dark ground (unsurveyed cells stay dark), north-up."""
        snap = self.snapshot() if snap is None else snap
        cls = self._frozen(snap).class_map(**self.gate)
        surveyed = cls >= 0
        img = np.full((self.hm.nx, self.hm.ny, 3), 18, np.uint8)
        img[surveyed] = E.PALETTE[cls[surveyed]]
        if size_px:
            return letterbox_grid(img, int(size_px), fill=18)      # letterbox_grid north-ups
        return img[::-1, ...]                                      # north-up at native size

    # ------------------------------------------------------------------ persistence
    def snapshot(self) -> dict:
        """Cheap copy of ALL the mutable state (≈ 2 MB) so a read-out never holds the lock.

        Everything a read-out needs is copied here, in one critical section: the vote grid,
        the observation grid, the frame counters, the last pose and the path.  A read-out that
        pulled any of these from the live object afterwards could see a `votes +=` from the
        serve thread half applied.
        """
        with self.lock:
            return dict(votes=self.hm.votes.copy(), obs=self.hm.obs.copy(),
                        frames=int(self.hm.frames), rays_cast=int(self.hm.rays_cast),
                        rays_hit=int(self.hm.rays_hit),
                        last_pose=self.last_pose,
                        last_cells_updated=int(self.last_cells_updated),
                        last_rays_cast=int(self.last_rays_cast),
                        last_rays_hit=int(self.last_rays_hit),
                        path_cm=list(self.path_cm))

    def _frozen(self, snap: dict) -> "E.HealthMap":
        """A read-only `HealthMap` over the SNAPSHOT's votes/obs.

        Shares the immutable DEM arrays (height / colour / count) with the live map and swaps
        in the snapshot's own copies of everything `update()` mutates, so the engine read-outs
        can be called verbatim without the lock.
        """
        f = E.HealthMap.__new__(E.HealthMap)
        f.__dict__.update(self.hm.__dict__)
        f.votes = snap["votes"]
        f.obs = snap["obs"]
        f.frames = int(snap["frames"])
        f.rays_cast = int(snap["rays_cast"])
        f.rays_hit = int(snap["rays_hit"])
        return f

    def save_state(self, path: str, snap=None) -> str:
        """Write a `HealthMap.load()`-compatible npz (identical keys to `HealthMap.save`)."""
        snap = snap or self.snapshot()
        hm = self.hm
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(
            path, votes=snap["votes"], obs=snap["obs"],
            origin_cm=hm.origin_cm, cell_cm=np.float64(hm.cell_cm),
            nx=np.int32(hm.nx), ny=np.int32(hm.ny),
            height=hm.height, height_filled=hm.height_filled,
            colour=hm.colour, count=hm.count,
            state=json.dumps(dict(frames=snap["frames"], rays_cast=snap["rays_cast"],
                                  rays_hit=snap["rays_hit"], max_range_m=hm.max_range_m,
                                  step_cm=hm.step_cm, min_range_m=hm.min_range_m,
                                  dem_meta=hm.dem_meta)))
        return path

    def save_artifacts(self, out_dir: str, tag: str = "", extra: dict = None) -> dict:
        """state npz + class-map PNG + minimap PNG + metrics JSON.  Returns the paths.

        ONE snapshot feeds all four files, so the npz, the class map and the metrics cannot
        disagree about how many frames had been folded in.  `self.save_lock` keeps the
        autosave thread and the main thread's disconnect/shutdown saves from writing the same
        names at once (a Windows PermissionError).
        """
        with self.save_lock:
            os.makedirs(out_dir, exist_ok=True)
            suffix = f"_{tag}" if tag else ""
            paths = {}
            snap = self.snapshot()
            frozen = self._frozen(snap)
            paths["state"] = self.save_state(
                os.path.join(out_dir, f"healthmap_state{suffix}.npz"), snap)

            cm_path = os.path.join(out_dir, f"class_map{suffix}.png")
            with open(cm_path, "wb") as fh:
                fh.write(encode_png(self.class_map_rgb(snap=snap)))
            paths["class_map"] = cm_path

            mm_path = os.path.join(out_dir, f"minimap_final{suffix}.png")
            with open(mm_path, "wb") as fh:
                fh.write(self.minimap_png(snap=snap))
            paths["minimap"] = mm_path

            m = self.metrics(frame_id=snap["frames"], snap=snap)
            # The engine's own summary alongside ours, so an offline reader can see the gate
            # that produced these numbers without re-deriving it - from the same snapshot.
            m["engine_summary"] = frozen.summary(**self.gate)
            m["dem"] = self.dem_path
            m["gt_path"] = self.gt_path
            m["minimaps_sent"] = self.minimaps_sent
            m["resets"] = self.resets
            m["saved_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if extra:
                m.update(extra)
            mj = os.path.join(out_dir, f"metrics{suffix}.json")
            with open(mj, "w", encoding="utf-8") as fh:
                json.dump(m, fh, indent=2)
            paths["metrics"] = mj
            return paths


# --------------------------------------------------------------------------- synthetic survey
def lawnmower_path(hm, alt_cm=120.0, spacing_cm=150.0, step_cm=25.0, pitch_deg=-60.0,
                   inset_cm=40.0):
    """Terrain-following boustrophedon over the filled part of the DEM (for the fake client
    and the tests; from CoralUnreal `scripts/healthmap/simulate_run.py::lawnmower`).

    Legs run along +/-X, stepped in +Y; yaw is the leg heading (0 = +X). Altitude is held
    `alt_cm` above the DEM under the vehicle. Returns a list of
    `dict(location_cm=(x, y, z), rotation_deg=(pitch, yaw, roll))`.
    """
    filled = np.isfinite(hm.height)
    ix, iy = np.nonzero(filled)
    x0, x1 = hm.cell_centre(ix.min(), 0)[0] + inset_cm, hm.cell_centre(ix.max(), 0)[0] - inset_cm
    y0, y1 = hm.cell_centre(0, iy.min())[1] + inset_cm, hm.cell_centre(0, iy.max())[1] - inset_cm
    poses = []
    for k, y in enumerate(np.arange(y0, y1 + 1e-6, spacing_cm)):
        xs = np.arange(x0, x1 + 1e-6, step_cm)
        yaw = 0.0
        if k % 2 == 1:
            xs, yaw = xs[::-1], 180.0
        for x in xs:
            z = float(hm.height_at(np.array([x]), np.array([y]))[0])
            if not np.isfinite(z):
                z = float(np.nanmedian(hm.height))
            poses.append(dict(location_cm=(float(x), float(y), z + alt_cm),
                              rotation_deg=(pitch_deg, yaw, 0.0)))
    return poses
