"""Board package: newest-slot dropping, int8 pre/post-processing, one-way receive end-to-end on
localhost (backend none), reply mode (`--reply`) end-to-end and its writer in isolation, map
folding from streamed poses, the PIL-free PNG encoder.
numpy only, except where a test says it skips (torch / Pillow / the Vitis dump).

Waits are on conditions with generous deadlines (BOARD_TEST_TIMEOUT, default 60 s), never fixed
sleeps, so the suite passes on the board's Cortex-A53 as well as on a PC."""
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

BOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_ROOT = os.path.dirname(BOARD_DIR)
sys.path.insert(0, BOARD_DIR)
sys.path.insert(0, os.path.join(BOARD_DIR, "tools"))
sys.path.insert(0, os.path.join(SIM_ROOT, "runtime"))

from reefwatch_board import prepost as PP  # noqa: E402
from reefwatch_board.app import App, make_parser  # noqa: E402
from reefwatch_board.pipeline import MapItem, MapWorker  # noqa: E402
from reefwatch_board.reply import NO_MAP_ERROR, Replier, ReplyContext  # noqa: E402
from reefwatch_board.slot import NewestSlot  # noqa: E402
from reefwatch_board.status import read_status  # noqa: E402
from reefwatch_runtime import config, protocol as P  # noqa: E402
from reefwatch_runtime.healthmap import png as PNG  # noqa: E402
import stream_frames  # noqa: E402

DEPLOY = (r"C:\Users\nicho\Projects\CoralUnreal\artifacts_quant\vitis\qr_m7c_e2allcs_s0"
          r"\deploy_check_data_int\CoralSegDPU")


#: deadline for every condition wait; the A53 is several times slower than a PC
TIMEOUT = float(os.environ.get("BOARD_TEST_TIMEOUT", "60"))


def wait_for(cond, timeout=TIMEOUT, step=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


# --------------------------------------------------------------------------- slot
class TestNewestSlot(unittest.TestCase):
    def test_newest_wins_and_counts_drops(self):
        s = NewestSlot()
        for i in range(5):
            s.put(i)
        self.assertEqual(s.get(timeout=0.1), 4)
        self.assertEqual((s.posted, s.taken, s.dropped), (5, 1, 4))
        self.assertIsNone(s.get(timeout=0.05))

    def test_get_blocks_until_put_and_close_unblocks(self):
        s = NewestSlot()
        got = []
        t = threading.Thread(target=lambda: got.append(s.get(timeout=5)))
        t.start()
        time.sleep(0.05)
        s.put("x")
        t.join(2)
        self.assertEqual(got, ["x"])
        t = threading.Thread(target=lambda: got.append(s.get()))
        t.start()
        s.close()
        t.join(2)
        self.assertFalse(t.is_alive())
        self.assertEqual(got[-1], None)


# --------------------------------------------------------------------------- pre/post
class TestPrePost(unittest.TestCase):
    def test_pad_matches_numpy_pad(self):
        rng = np.random.default_rng(0)
        for (h, w, H, W) in [(360, 640, 512, 1024), (360, 640, 512, 768), (5, 7, 16, 9),
                             (5, 7, 5, 16), (100, 40, 100, 40)]:
            img = rng.integers(-128, 127, (h, w, 3)).astype(np.int8)
            dst = np.zeros((H, W, 3), np.int8)
            dst[:h, :w] = img
            PP.pad_bottom_right(dst, h, w)
            mode = "reflect" if (H - h < h and W - w < w) else "edge"
            np.testing.assert_array_equal(dst, np.pad(img, ((0, H - h), (0, W - w), (0, 0)), mode))

    def test_int8_lut_matches_formula_and_buffer_layout(self):
        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 256, (360, 640, 3)).astype(np.uint8)
        inp = PP.Int8Input(512, 768, fix_point=5)
        self.assertEqual(inp.fill(rgb), (360, 640))
        mean = np.array(PP.IMAGENET_MEAN)
        std = np.array(PP.IMAGENET_STD)
        ref = np.clip(np.round((rgb / 255.0 - mean) / std * 32), -128, 127).astype(np.int8)
        self.assertEqual(inp.buf.shape, (1, 512, 768, 3))          # NHWC, batch 1
        np.testing.assert_array_equal(inp.buf[0, :360, :640], ref)
        np.testing.assert_array_equal(inp.buf[0, 360:], inp.buf[0, 358:206:-1])  # reflect, no edge repeat

    def test_logits_to_ids_crop_and_nearest(self):
        lg = np.zeros((128, 192, 7), np.int8)
        lg[..., 6] = 1                                    # OTHER everywhere ...
        lg[:45, :80, 0] = 5                               # ... BLEACHED in the top-left quarter
        ids = PP.logits_to_ids(lg, 360, 640, 4)
        self.assertEqual(ids.shape, (360, 640))
        self.assertEqual(ids.dtype, np.uint8)
        self.assertTrue((ids[:180, :320] == 0).all() and (ids[180:] == 6).all() and (ids[:, 320:] == 6).all())
        # a frame shrunk into the buffer comes back at its own size
        ids2 = PP.logits_to_ids(lg, 360, 640, 4, out_h=720, out_w=1280)
        self.assertEqual(ids2.shape, (720, 1280))
        self.assertTrue((ids2[:360, :640] == 0).all() and (ids2[360:] == 6).all())

    def test_oversized_frame_is_shrunk_to_fit(self):
        inp = PP.Int8Input(512, 768, 5)
        self.assertEqual(inp.fill(np.zeros((720, 1280, 3), np.uint8)), (432, 768))

    def test_stats(self):
        lg = np.zeros((90, 160, 7), np.float32)
        s = PP.logit_stats(lg, 360, 640, 4, 1.0, 0)
        np.testing.assert_allclose(s, [1 / 7, np.log(7), 1 / 7], rtol=1e-5)

    def test_health_table_matches_runtime(self):
        try:
            from reefwatch_runtime import model as M
        except ImportError:
            self.skipTest("torch not installed (runtime.model imports it)")
        for k, v in M.SEVEN_TO_HEALTH.items():
            self.assertEqual(tuple(PP.SEVEN_TO_HEALTH[k]), tuple(v))
        self.assertEqual(PP.SEVEN_NAMES, M.SEVEN_NAMES)
        self.assertEqual([tuple(int(c) for c in row) for row in M.HEALTH_PALETTE_RGBA],
                         [tuple(row) for row in PP.HEALTH_PALETTE_RGBA])
        self.assertEqual(PP.SEVEN_HEX, tuple(M.SEVEN_HEX[n] for n in M.SEVEN_NAMES))
        self.assertEqual([tuple(int(c) for c in row) for row in M.SEVEN_PALETTE_RGBA],
                         list(PP.SEVEN_PALETTE_RGBA))
        self.assertEqual(PP.IMAGENET_MEAN, M.IMAGENET_MEAN)
        self.assertEqual(PP.IMAGENET_STD, M.IMAGENET_STD)

    @unittest.skipUnless(os.path.isdir(DEPLOY), "Vitis deploy-check dump not on this machine")
    def test_against_quantizer_dump(self):
        """The quantizer's own INT8 tensors: NHWC input at fix 5; the DPU output (the 4x upsample
        rewritten onto the DPU) NHWC at full resolution, fix 3; argmax on int8 == on float."""
        x = np.fromfile(os.path.join(DEPLOY, "CoralSegDPU__input_0.bin"), np.float32)
        xq = np.fromfile(os.path.join(DEPLOY, "CoralSegDPU__input_0_fix.bin"), np.int8)
        np.testing.assert_array_equal(np.clip(np.round(x * 32), -128, 127).astype(np.int8), xq)
        yq = np.fromfile(os.path.join(DEPLOY, "CoralSegDPU__CoralSegDPU_Upsample_up_out__ret_fix.bin"),
                         np.int8).reshape(512, 1024, 7)
        y = np.fromfile(os.path.join(DEPLOY, "CoralSegDPU__CoralSegDPU_Upsample_up_out__ret.bin"),
                        np.float32).reshape(512, 1024, 7)
        np.testing.assert_array_equal(yq.astype(np.float32) / 8, y)
        rough = lambda a: float(np.abs(np.diff(a.astype(np.float32), axis=1)).mean())  # noqa: E731
        self.assertLess(rough(yq[..., 0]), rough(yq.reshape(7, 512, 1024)[0]))        # NHWC, not NCHW
        ids = PP.logits_to_ids(yq, 360, 640)
        np.testing.assert_array_equal(ids, y[:360, :640].argmax(-1))


# --------------------------------------------------------------------------- vart (faked)
class _FakeTensor:
    def __init__(self, dims, fix):
        self.dims, self._fix = dims, fix

    def get_attr(self, name):
        assert name == "fix_point"
        return self._fix


class _FakeSubgraph:
    def __init__(self, name, device):
        self.name, self.device = name, device

    def has_attr(self, k):
        return k == "device"

    def get_attr(self, k):
        return self.device

    def get_name(self):
        return self.name


class _FakeRunner:
    """Stands in for vart.Runner: checks the buffers it is handed and writes logits that are
    BLEACHED on the left half of the 640-px image region and OTHER elsewhere."""

    def __init__(self, in_dims, out_dims):
        self.in_dims, self.out_dims = in_dims, out_dims
        self.calls = 0

    def get_input_tensors(self):
        return [_FakeTensor(self.in_dims, 5)]

    def get_output_tensors(self):
        return [_FakeTensor(self.out_dims, 3)]

    def execute_async(self, inputs, outputs):
        (x,), (y,) = inputs, outputs
        assert x.shape == self.in_dims and x.dtype == np.int8 and x.flags.c_contiguous
        assert y.shape == self.out_dims and y.dtype == np.int8
        y[...] = 0
        y[..., 6] = 8
        stride = self.in_dims[2] // self.out_dims[2]
        y[:, :, :320 // stride, 0] = 16             # left 320 px of the 640-px frame
        self.calls += 1
        return self.calls

    def wait(self, job):
        return 0


class TestVartBackendFaked(unittest.TestCase):
    def _install(self, in_dims=(1, 512, 768, 3), out_dims=(1, 512, 768, 7)):
        import types
        runner = _FakeRunner(in_dims, out_dims)
        sgs = [_FakeSubgraph("input", "USER"), _FakeSubgraph("dpu_main", "DPU"),
               _FakeSubgraph("up", "CPU")]
        root = types.SimpleNamespace(toposort_child_subgraph=lambda: sgs)
        graph = types.SimpleNamespace(get_root_subgraph=lambda: root)
        xir = types.ModuleType("xir")
        xir.Graph = types.SimpleNamespace(deserialize=lambda path: graph)
        vart = types.ModuleType("vart")
        vart.Runner = types.SimpleNamespace(
            create_runner=lambda sg, mode: (self.assertEqual((sg.name, mode), ("dpu_main", "run")), runner)[1])
        self._saved = {k: sys.modules.get(k) for k in ("vart", "xir")}
        sys.modules.update(vart=vart, xir=xir)
        return runner

    def tearDown(self):
        for k, v in getattr(self, "_saved", {}).items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_shapes_from_tensors_and_postprocess(self):
        from reefwatch_board.backends import VartBackend
        runner = self._install()
        be = VartBackend("fake.xmodel", log=lambda m: None)
        self.assertEqual((be.stride, be.num_classes, be.in_fix, be.out_fix), (1, 7, 5, 3))
        ids, stats = be.predict_with_stats(np.zeros((360, 640, 3), np.uint8))
        self.assertEqual(runner.calls, 1)
        self.assertEqual(ids.shape, (360, 640))
        self.assertTrue((ids[:, :320] == 0).all() and (ids[:, 320:] == 6).all())
        # dequantised logits: left BLEACHED 2.0 / OTHER 1.0, right OTHER 1.0, the rest 0
        e = np.e
        p_left, p_right = e ** 2 / (e ** 2 + e + 5), 1 / (e + 6)
        self.assertAlmostEqual(float(stats[2]), 0.5 * (p_left + p_right), places=4)
        self.assertEqual(set(be.timings), {"pre", "dpu", "post"})
        d = be.describe()
        self.assertEqual(d["input"], [1, 512, 768, 3])

    def test_strided_output_build(self):
        from reefwatch_board.backends import VartBackend
        self._install(out_dims=(1, 128, 192, 7))
        be = VartBackend("fake.xmodel", log=lambda m: None)
        self.assertEqual(be.stride, 4)
        ids, _ = be.predict_with_stats(np.zeros((360, 640, 3), np.uint8))
        self.assertTrue((ids[:, :320] == 0).all() and (ids[:, 320:] == 6).all())

    def test_rejects_non_nhwc_input(self):
        from reefwatch_board.backends import VartBackend
        self._install(in_dims=(1, 3, 512, 768), out_dims=(1, 7, 128, 192))
        with self.assertRaises(RuntimeError):
            VartBackend("fake.xmodel", log=lambda m: None)

    def test_pynq_dpu_overlay_is_loaded_before_vart(self):
        import types
        from reefwatch_board.backends import VartBackend
        order = []

        class FakeOverlay:                                   # as in DPU-PYNQ 3.5: no method;
            def __init__(self, bitfile):                     # check_vart_config lives in pynq_dpu.dpu
                order.append(("overlay", bitfile))

        self._install()
        mod = types.ModuleType("pynq_dpu")
        mod.DpuOverlay = FakeOverlay
        sub = types.ModuleType("pynq_dpu.dpu")
        sub.check_vart_config = lambda: order.append(("vart.conf",))
        mod.dpu = sub
        for k in ("pynq_dpu", "pynq_dpu.dpu"):
            self._saved[k] = sys.modules.get(k)
        sys.modules["pynq_dpu"] = mod
        sys.modules["pynq_dpu.dpu"] = sub
        be = VartBackend("fake.xmodel", log=lambda m: None, pynq_overlay="dpu.bit")
        self.assertEqual(order, [("overlay", "dpu.bit"), ("vart.conf",)])
        self.assertIsInstance(be.overlay, FakeOverlay)
        sys.modules["pynq_dpu"] = None                       # not installed -> clear message
        sys.modules["pynq_dpu.dpu"] = None
        with self.assertRaises(RuntimeError) as cm:
            VartBackend("fake.xmodel", log=lambda m: None, pynq_overlay="dpu.bit")
        self.assertIn("SETUP_UBUNTU.md", str(cm.exception))

    def test_clear_error_without_bindings(self):
        from reefwatch_board.backends import VartBackend
        self._saved = {k: sys.modules.get(k) for k in ("vart", "xir")}
        sys.modules["vart"] = None
        with self.assertRaises(RuntimeError) as cm:
            VartBackend("fake.xmodel")
        self.assertIn("import vart, xir", str(cm.exception))
        self.assertIn("xdputil query", str(cm.exception))


# --------------------------------------------------------------------------- vart (shim)
FAKE_VITIS = os.path.join(BOARD_DIR, "tools", "fake_vitis")
XMODEL_768 = os.path.join(BOARD_DIR, "models", "coral_milestone_7c_512x768.xmodel")
DPU_REF = os.path.join(BOARD_DIR, "tests", "data", "dpu_ref_auv_frame_512x768.npz")


@unittest.skipUnless(os.path.isfile(XMODEL_768) and os.path.isdir(FAKE_VITIS),
                     "models/coral_milestone_7c_512x768.xmodel or tools/fake_vitis not here (bundle)")
class TestVartBackendThroughShim(unittest.TestCase):
    """The unmodified VartBackend against tools/fake_vitis: a fake xir that returns what the real
    xir returned for this exact xmodel (md5-checked spec), and a fake vart that is stricter than the
    real binding and replays the bit-exact DPU reference output for the bundled Unreal frame."""

    def setUp(self):
        import importlib.util
        self._saved = {k: sys.modules.get(k) for k in ("vart", "xir")}
        self._env = {k: os.environ.get(k) for k in ("FAKE_VART_COMPUTE", "FAKE_VART_REPLAY")}
        for name in ("xir", "vart"):                     # xir first: the fake vart imports it
            spec = importlib.util.spec_from_file_location(name, os.path.join(FAKE_VITIS, name + ".py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        self.vart = sys.modules["vart"]

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _frame(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow needed to decode the filtered test PNG")
        return np.ascontiguousarray(np.asarray(Image.open(str(config.TEST_FRAME)).convert("RGB")))

    def test_tensors_and_subgraph_from_real_xir_spec(self):
        from reefwatch_board.backends import VartBackend
        os.environ["FAKE_VART_COMPUTE"] = "pattern"
        be = VartBackend(XMODEL_768, log=lambda m: None)
        self.assertEqual(self.vart.CREATED[-1],
                         ("subgraph_CoralSegDPU__CoralSegDPU_Conv2d_head__ret_349", "run"))
        d = be.describe()
        self.assertEqual((d["input"], d["input_fix_point"], d["output"], d["output_fix_point"]),
                         ([1, 512, 768, 3], 5, [1, 512, 768, 7], 3))
        self.assertEqual((be.stride, be.num_classes), (1, 7))
        self.assertEqual(be.subgraph.get_attr("dpu_name"), "DPUCZDX8G_ISA1_B4096")
        ids, stats = be.predict_with_stats(np.zeros((360, 640, 3), np.uint8))
        self.assertEqual((ids.shape, ids.dtype, stats.shape), ((360, 640), np.uint8, (3,)))
        self.assertEqual(be.runner.calls, 1)

    def test_runner_rejects_what_the_real_binding_mishandles(self):
        os.environ["FAKE_VART_COMPUTE"] = "pattern"
        xir = sys.modules["xir"]
        from reefwatch_board.backends import dpu_subgraphs
        r = self.vart.Runner.create_runner(dpu_subgraphs(xir.Graph.deserialize(XMODEL_768))[0], "run")
        x, y = np.zeros((1, 512, 768, 3), np.int8), np.zeros((1, 512, 768, 7), np.int8)
        self.assertEqual(r.wait(r.execute_async([x], [y])), 0)
        for bad in (x.view(np.uint8), np.zeros((1, 512, 1024, 3), np.int8),
                    np.zeros((1, 512, 1536, 3), np.int8)[:, :, ::2]):
            with self.assertRaises(RuntimeError):
                r.execute_async([bad], [y])

    @unittest.skipUnless(os.path.isfile(DPU_REF), "DPU reference fixture missing")
    def test_unreal_frame_matches_dpu_reference_exactly(self):
        from reefwatch_board.backends import VartBackend
        os.environ["FAKE_VART_COMPUTE"] = "replay"
        os.environ["FAKE_VART_REPLAY"] = DPU_REF
        ref = np.load(DPU_REF)
        be = VartBackend(XMODEL_768, log=lambda m: None)
        ids, stats = be.predict_with_stats(self._frame())   # misses raise: the int8 input is exact
        self.assertEqual(be.runner.compute_used, {"replay": 1})
        np.testing.assert_array_equal(ids, ref["ids"])
        np.testing.assert_allclose(stats, ref["stats"], rtol=1e-5)
        bleached = PP.to_health_table(7)[ids] == 1
        self.assertGreater(bleached.mean(), 0.05)          # 9.2 % BLEACHED on this frame in INT8


# --------------------------------------------------------------------------- PNG
class TestPngWithoutPil(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(2)
        for c in (3, 4):
            img = rng.integers(0, 256, (37, 53, c)).astype(np.uint8)
            png = PNG.encode_png_zlib(img)
            np.testing.assert_array_equal(PNG.decode_png_zlib(png), img)
            try:
                import io
                from PIL import Image
            except ImportError:
                continue
            np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(png))), img)

    def test_runner_falls_back_without_pil(self):
        from reefwatch_runtime.healthmap import runner as R
        saved = sys.modules.get("PIL", "absent")
        sys.modules["PIL"] = None                      # makes `from PIL import Image` raise
        try:
            img = np.zeros((8, 8, 4), np.uint8)
            img[..., 1] = 200
            png = R.encode_png(img)
        finally:
            if saved == "absent":
                del sys.modules["PIL"]
            else:
                sys.modules["PIL"] = saved
        np.testing.assert_array_equal(PNG.decode_png_zlib(png), img)


# --------------------------------------------------------------------------- map
@unittest.skipUnless(config.DEFAULT_DEM.is_file(), "DEM missing")
class TestMapFromStreamedPoses(unittest.TestCase):
    def test_route_poses_fill_the_map_and_reset_generations(self):
        from reefwatch_runtime.healthmap import MapRunner
        runner = MapRunner(dem=str(config.DEFAULT_DEM))
        route = stream_frames.RoutePoses(stream_frames.DEFAULT_ROUTE)
        mw = MapWorker(runner, max_hz=0, minimap_hz=0, log=lambda m: None).start()
        ids = np.full((360, 640), 3, np.uint8)              # OTHER
        ids[:, 320:] = 1                                     # BLEACHED on the right
        try:
            for k in range(20):
                cam, rot = route.at(k * 0.1)
                mw.post(MapItem(ids, dict(location_cm=cam, rotation_deg=rot), 78.0, k + 1, 0))
                self.assertTrue(wait_for(lambda: mw.folded == k + 1, 10))
            self.assertGreater(runner.last_rays_hit, 0.5 * runner.last_rays_cast)
            m, _, _ = mw.latest()
            self.assertEqual(m["frames"], 20)
            self.assertGreater(m["surveyed_area_m2"], 0.2)
            self.assertGreater(m["class_area_m2"][1], 0)        # BLEACHED cells
            # generation 1 resets before folding; a late generation-0 item is discarded
            cam, rot = route.at(3.0)
            mw.post(MapItem(ids, dict(location_cm=cam, rotation_deg=rot), 78.0, 99, 1))
            self.assertTrue(wait_for(lambda: mw.folded == 21, 10))
            self.assertEqual(runner.hm.frames, 1)
            mw.post(MapItem(ids, dict(location_cm=cam, rotation_deg=rot), 78.0, 50, 0))
            self.assertTrue(wait_for(lambda: mw.stale == 1, 10))
            self.assertEqual(runner.hm.frames, 1)
        finally:
            mw.stop()


# --------------------------------------------------------------------------- end to end
class TestOneWayEndToEnd(unittest.TestCase):
    """The real App on localhost with backend none: frames in, nothing out, map saved."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rw_board_")
        args = make_parser().parse_args(["--backend", "none", "--host", "127.0.0.1", "--port", "0",
                                         "--status-port", "0", "--out", self.tmp, "--quiet",
                                         "--map-hz", "50", "--minimap-hz", "20", "--save-every", "0"])
        self.app = App(args, log=lambda m: None)
        self.app.setup()
        self.t = threading.Thread(target=self.app.run, daemon=True)
        self.t.start()
        self.port = self.app.receiver.port

    def tearDown(self):
        self.app.stop_event.set()
        self.t.join(15)

    def connect(self):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s

    def send_frames(self, s, n, start=1, w=320, h=180, flags=P.FLAG_POSE_VALID, pace_s=0.01):
        """Paced like a camera: a burst would (correctly) be collapsed to its newest frame."""
        route = stream_frames.RoutePoses(stream_frames.DEFAULT_ROUTE)
        payload = bytes(w * h * 3)
        for k in range(n):
            cam, rot = route.at((start + k) * 0.1)
            s.sendall(P.pack_request_v2(start + k, w, h, len(payload), flags=flags, cam=cam,
                                        rot=rot, fov_h_deg=78.0) + payload)
            time.sleep(pace_s)

    def test_receive_map_status_and_save(self):
        app = self.app
        s = self.connect()
        self.send_frames(s, 40, flags=P.FLAG_POSE_VALID | P.FLAG_RESET_MAP)
        self.send_frames(s, 40, start=41)
        self.assertTrue(wait_for(lambda: app.receiver.frames_rx == 80))
        self.assertEqual(app.receiver.reset_gen, 1)                 # rising edge only
        # a fold takes ~0.6 s on the A53 and the map is newest-wins, so 80 frames in ~1 s fold only
        # 2-3 times there: assert that the map folded at all (surveyed_cells > 0 checks content)
        self.assertTrue(wait_for(lambda: app.pipeline.frames_done > 0 and app.map_worker.folded >= 1))
        self.assertTrue(wait_for(lambda: app.map_worker.latest()[1] is not None))   # a minimap exists
        self.assertEqual(app.receiver.slot.posted, 80)
        # one-way: the board never writes on the frame socket
        s.settimeout(0.5)
        with self.assertRaises(socket.timeout):
            s.recv(1)
        # status port: one message with the latest minimap
        v = socket.create_connection(("127.0.0.1", app.status.port), timeout=TIMEOUT)
        v.settimeout(TIMEOUT)
        end = time.monotonic() + TIMEOUT        # a condition, not a message count (slow A53)
        js, png = read_status(v)
        while not png and time.monotonic() < end:
            js, png = read_status(v)
        v.close()
        self.assertEqual(js["frames_rx"], 80)
        self.assertEqual(js["backend"], "none")
        self.assertTrue(png.startswith(PNG.PNG_SIGNATURE))
        self.assertIn("coverage", js["map"])
        s.close()                                                    # disconnect -> save
        self.assertTrue(wait_for(lambda: os.path.isfile(os.path.join(self.tmp, "metrics.json"))))
        app.stop_event.set()
        self.t.join(15)
        for f in ("healthmap_state.npz", "class_map.png", "minimap_final.png", "metrics.json",
                  "status.jsonl"):
            self.assertTrue(os.path.isfile(os.path.join(self.tmp, f)), f)
        with open(os.path.join(self.tmp, "metrics.json"), encoding="utf-8") as fh:
            m = json.load(fh)
        self.assertEqual(m["reason"], "shutdown")
        self.assertEqual(m["board"]["frames_rx"], 80)
        self.assertGreater(m["surveyed_cells"], 0)

    def test_new_connection_replaces_old_and_malformed_is_dropped(self):
        app = self.app
        a = self.connect()
        self.send_frames(a, 3)
        self.assertTrue(wait_for(lambda: app.receiver.frames_rx == 3))
        b = self.connect()
        a.settimeout(5)
        try:
            self.assertEqual(a.recv(1), b"")                       # closed by the board
        except (ConnectionResetError, ConnectionAbortedError):
            pass
        self.send_frames(b, 2, start=4)
        self.assertTrue(wait_for(lambda: app.receiver.frames_rx == 5))
        self.assertTrue(wait_for(lambda: app.receiver.connections == 2))
        b.sendall(b"JUNK" + bytes(60))
        b.settimeout(5)
        try:
            self.assertEqual(b.recv(1), b"")
        except (ConnectionResetError, ConnectionAbortedError):
            pass
        a.close()
        b.close()


# --------------------------------------------------------------------------- reply mode
def read_response(sock):
    """Client side, as SegNetClient: one response -> parsed dict (CSG2 or CSEG v1)."""
    def exact(n):
        buf = bytearray(n)
        mv, got = memoryview(buf), 0
        while got < n:
            k = sock.recv_into(mv[got:], n - got)
            if k == 0:
                raise ConnectionError("closed")
            got += k
        return bytes(buf)
    magic = exact(4)
    if magic == P.MAGIC_V2:
        head = magic + exact(P.RESP_V2_SIZE - 4)
        plen = struct.unpack_from("<I", head, 28)[0]
        r = P.parse_response_v2(head, exact(plen))
        r["version"] = 2
        return r
    if magic == P.MAGIC_V1:
        head = magic + exact(P.RESP_V1_SIZE - 4)
        plen = struct.unpack_from("<I", head, 24)[0]
        r = P.parse_response_v1(head, exact(plen))
        r["version"] = 1
        return r
    raise ValueError(f"bad response magic {magic!r}")


OTHER_RGBA = bytes(PP.HEALTH_PALETTE_RGBA[3])
LIVE_RGBA = bytes(PP.HEALTH_PALETTE_RGBA[0])


class TestReplyEndToEnd(unittest.TestCase):
    """The real App with --reply on localhost (backend none = every pixel OTHER), driven by a
    client that behaves like Unreal's normal mode: one request in flight, read the response."""

    extra_args = []

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rw_board_reply_")
        args = make_parser().parse_args(["--backend", "none", "--host", "127.0.0.1", "--port", "0",
                                         "--status-port", "0", "--out", self.tmp, "--quiet",
                                         "--map-hz", "50", "--minimap-hz", "20", "--save-every", "0",
                                         "--reply"] + self.extra_args)
        self.app = App(args, log=lambda m: None)
        self.app.setup()
        self.t = threading.Thread(target=self.app.run, daemon=True)
        self.t.start()
        self.port = self.app.receiver.port
        self.route = stream_frames.RoutePoses(stream_frames.DEFAULT_ROUTE)

    def tearDown(self):
        self.app.stop_event.set()
        self.t.join(TIMEOUT)

    def connect(self):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=TIMEOUT)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s

    def exchange(self, s, fid, w=320, h=180, flags=P.FLAG_POSE_VALID, version=2):
        payload = bytes(w * h * 3)
        if version == 2:
            cam, rot = self.route.at(fid * 0.1)
            s.sendall(P.pack_request_v2(fid, w, h, len(payload), flags=flags, cam=cam, rot=rot,
                                        fov_h_deg=78.0, world_time_s=fid * 0.1) + payload)
        else:
            s.sendall(P.pack_request_v1(fid, w, h, len(payload)) + payload)
        return read_response(s)

    def test_one_response_per_request_with_mask_map_and_minimap(self):
        app = self.app
        s = self.connect()
        r = self.exchange(s, 1, flags=P.FLAG_POSE_VALID | P.FLAG_WANT_MINIMAP | P.FLAG_RESET_MAP)
        self.assertEqual((r["version"], r["frame_id"], r["width"], r["height"]), (2, 1, 320, 180))
        self.assertEqual(len(r["mask_rgba"]), 320 * 180 * 4)
        self.assertEqual(r["mask_rgba"], OTHER_RGBA * (320 * 180))      # none backend: all OTHER
        np.testing.assert_allclose(r["class_frac"], [0, 0, 0, 1])
        self.assertEqual(len(r["stats"]), 4)
        self.assertGreaterEqual(r["total_ms"], r["infer_ms"])
        # keep flying (one request in flight) until the map and a minimap come back
        metrics = minimap = None
        fid = 1
        end = time.monotonic() + TIMEOUT
        while (metrics is None or minimap is None) and time.monotonic() < end:
            fid += 1
            r = self.exchange(s, fid, flags=P.FLAG_POSE_VALID | P.FLAG_WANT_MINIMAP)
            self.assertEqual(r["frame_id"], fid)                         # echo, in order
            metrics = r.get("map_metrics") or metrics
            minimap = r.get("minimap_png") or minimap
        self.assertIsNotNone(metrics, "no MAP_METRICS")
        self.assertIsNotNone(minimap, "no MINIMAP_PNG")
        self.assertTrue(minimap.startswith(PNG.PNG_SIGNATURE))
        for k in ("frame_id", "frames", "coverage_fraction", "class_area_m2", "grid"):
            self.assertIn(k, metrics)
        self.assertEqual(app.receiver.reset_gen, 1)
        self.assertTrue(wait_for(lambda: app.reply_ctx.counters.sent == fid))
        self.assertLessEqual(app.reply_ctx.counters.minimaps, fid)        # each render sent once
        self.assertEqual(app.receiver.frames_rx, fid)
        self.assertEqual(app.receiver.slot.dropped, 0)                   # lockstep never drops
        # CSEG v1 request on the same connection -> v1 response
        r = self.exchange(s, fid + 1, version=1)
        self.assertEqual((r["version"], r["frame_id"]), (1, fid + 1))
        self.assertEqual(r["mask_rgba"], OTHER_RGBA * (320 * 180))
        s.close()
        self.assertTrue(wait_for(lambda: os.path.isfile(os.path.join(self.tmp, "metrics.json"))))
        app.stop_event.set()
        self.t.join(TIMEOUT)
        with open(os.path.join(self.tmp, "metrics.json"), encoding="utf-8") as fh:
            m = json.load(fh)
        self.assertEqual(m["board"]["replies"]["sent"], fid + 1)        # health map + saving on

    def test_reset_map_hides_the_old_map_until_applied(self):
        app = self.app
        s = self.connect()
        fid = 0
        end = time.monotonic() + TIMEOUT
        while app.map_worker.folded < 3 and time.monotonic() < end:
            fid += 1
            self.exchange(s, fid)
        gen0 = app.map_worker.applied_gen
        fid += 1
        r = self.exchange(s, fid, flags=P.FLAG_POSE_VALID | P.FLAG_RESET_MAP)
        self.assertEqual(app.receiver.reset_gen, gen0 + 1)
        if "map_metrics" in r:      # only once the map thread has applied the reset
            self.assertGreaterEqual(app.map_worker.applied_gen, gen0 + 1)
            self.assertLessEqual(r["map_metrics"]["frames"], 1)
        s.close()


    def test_stream_replies_answer_only_processed_frames(self):
        """STREAM_REPLIES (Unreal's real-time display): the client sends without waiting; the
        board answers only the frames its pipeline ran, in order, each with FRAME_INFO."""
        app = self.app
        s = self.connect()
        plain = self.exchange(s, 1)                                   # no flag: today's bytes
        self.assertNotIn(P.BLOCK_FRAME_INFO, plain["blocks"])
        self.assertNotIn(P.BLOCK_SERVER_INFO, plain["blocks"])
        replies, err = [], []
        n0, n = 2, 41

        def reader():
            try:
                while not replies or replies[-1]["frame_id"] != n:
                    replies.append(read_response(s))
            except Exception as exc:                                  # noqa: BLE001
                err.append(exc)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        payload = bytes(320 * 180 * 3)
        for fid in range(n0, n + 1):                                  # a burst, never waiting
            cam, rot = self.route.at(fid * 0.1)
            s.sendall(P.pack_request_v2(fid, 320, 180, len(payload),
                                        flags=P.FLAG_POSE_VALID | P.FLAG_STREAM_REPLIES,
                                        cam=cam, rot=rot, fov_h_deg=78.0,
                                        world_time_s=fid * 0.1) + payload)
        t.join(TIMEOUT)
        self.assertEqual(err, [])
        ids = [r["frame_id"] for r in replies]
        self.assertEqual(ids, sorted(set(ids)))                       # once each, in order
        self.assertEqual(ids[-1], n)                                  # the newest always runs
        for r in replies:
            fi = r["frame_info"]
            self.assertTrue(fi["streamed"])
            self.assertEqual(fi["frame_id"], r["frame_id"])
            self.assertAlmostEqual(fi["world_time_s"], r["frame_id"] * 0.1, places=5)
            self.assertEqual(r["mask_rgba"], OTHER_RGBA * (320 * 180))
        self.assertEqual(replies[0]["server_info"]["kind"], "board")  # once per connection
        self.assertEqual(replies[0]["server_info"]["name"], "Board program (no model)")
        si = replies[0]["server_info"]                                # the HUD legend tables
        self.assertEqual(si["health_classes"]["names"], list(PP.HEALTH_NAMES))
        self.assertEqual(si["native_classes"]["names"], list(PP.SEVEN_NAMES))
        self.assertEqual(si["native_classes"]["colors"], list(PP.SEVEN_HEX))
        self.assertTrue(all("server_info" not in r for r in replies[1:]))
        c = app.reply_ctx.counters
        self.assertTrue(wait_for(lambda: c.streamed + c.skipped == n - n0 + 1))
        self.assertEqual(c.streamed, len(replies))
        last = replies[-1]["frame_info"]
        self.assertEqual(last["frames_rx"], n)                         # incl. the plain request
        self.assertEqual(last["frames_done"] + last["frames_skipped"], n)
        s.close()


class TestReplyNoMap(TestReplyEndToEnd):
    """--no-map: a posed request gets one ERROR block, segmentation continues."""

    extra_args = ["--no-map"]

    def test_one_response_per_request_with_mask_map_and_minimap(self):
        s = self.connect()
        r1 = self.exchange(s, 1)
        r2 = self.exchange(s, 2)
        self.assertEqual(r1["error"], NO_MAP_ERROR.decode())
        self.assertNotIn("error", r2)
        self.assertNotIn("map_metrics", r2)
        self.assertEqual(r2["frame_id"], 2)
        s.close()

    def test_reset_map_hides_the_old_map_until_applied(self):
        pass


class TestOneWayDefault(unittest.TestCase):
    def test_reply_is_off_by_default(self):
        a = make_parser().parse_args([])
        self.assertFalse(a.reply)
        self.assertEqual(a.reply_timeout, 2.0)


class TestReplierUnit(unittest.TestCase):
    """The writer in isolation, on a socket pair (no App)."""

    def setUp(self):
        self.board, self.client = socket.socketpair()
        self.board.settimeout(2.0)
        self.client.settimeout(TIMEOUT)
        self.logs = []

    def tearDown(self):
        for x in (self.board, self.client):
            try:
                x.close()
            except OSError:
                pass

    def ctx(self, **kw):
        return ReplyContext(PP.to_health_table(7), None, log=self.logs.append, **kw)

    def ids(self, h=36, w=64, value=2):                     # 7-class ALIVE -> health LIVE
        return np.full((h, w), value, np.uint8)

    def test_superseded_requests_reuse_the_next_mask_in_order(self):
        ctx = self.ctx()
        r = Replier(ctx, self.board, ("pair", 1))
        e1 = r.expect(1, 64, 36, 2, 0, 0, time.perf_counter())
        e2 = r.expect(2, 64, 36, 2, 0, 0, time.perf_counter())
        r.superseded(e1)
        r.done(e2, self.ids(), np.array([1, 0, 0, 0], np.float32), None, 5.0)
        a, b = read_response(self.client), read_response(self.client)
        self.assertEqual((a["frame_id"], b["frame_id"]), (2, 2))         # the mask's own frame
        self.assertEqual(a["mask_rgba"], LIVE_RGBA * (64 * 36))
        self.assertTrue(wait_for(lambda: ctx.counters.sent == 2))
        self.assertEqual(ctx.counters.reused, 1)
        r.close()

    def test_processed_policy_answers_only_frames_that_ran(self):
        ctx = self.ctx(policy="processed")
        r = Replier(ctx, self.board, ("pair", 2))
        es = [r.expect(i, 64, 36, 2, 0, 0, time.perf_counter(), i * 0.1) for i in (1, 2, 3)]
        r.superseded(es[0])
        r.superseded(es[1])
        r.done(es[2], self.ids(), np.array([1, 0, 0, 0], np.float32), None, 5.0)
        a = read_response(self.client)
        self.assertEqual(a["frame_id"], 3)
        self.assertTrue(wait_for(lambda: ctx.counters.sent == 1))
        self.client.settimeout(0.3)
        with self.assertRaises(socket.timeout):
            self.client.recv(1)                                           # nothing for 1 and 2
        r.close()

    def test_stream_flag_selects_processed_per_request(self):
        """lockstep context: flagged requests are answered only if they ran, unflagged ones
        exactly once each; FRAME_INFO / SERVER_INFO only for requests that asked."""
        ctx = self.ctx()
        r = Replier(ctx, self.board, ("pair", 6))
        st = P.FLAG_STREAM_REPLIES
        e1 = r.expect(1, 64, 36, 2, st, 0, time.perf_counter(), 0.1)
        e2 = r.expect(2, 64, 36, 2, st, 0, time.perf_counter(), 0.2)
        r.superseded(e1)
        r.done(e2, self.ids(), np.array([1, 0, 0, 0], np.float32), None, 5.0)
        a = read_response(self.client)
        self.assertEqual(a["frame_id"], 2)
        self.assertTrue(a["frame_info"]["streamed"])
        self.assertEqual((a["frame_info"]["frames_rx"], a["frame_info"]["frames_done"],
                          a["frame_info"]["frames_skipped"]), (2, 1, 1))
        self.assertIn("server_info", a)
        e3 = r.expect(3, 64, 36, 2, 0, 0, time.perf_counter())       # no flag: lockstep
        e4 = r.expect(4, 64, 36, 2, P.FLAG_WANT_INFO, 0, time.perf_counter(), 0.4)
        r.superseded(e3)
        r.done(e4, self.ids(), np.array([1, 0, 0, 0], np.float32), None, 5.0)
        b, c = read_response(self.client), read_response(self.client)
        self.assertEqual((b["frame_id"], c["frame_id"]), (4, 4))       # 3 reuses 4's mask
        self.assertNotIn(P.BLOCK_FRAME_INFO, b["blocks"])            # 3 did not ask
        self.assertFalse(c["frame_info"]["streamed"])
        self.assertNotIn("server_info", c)                           # already sent once
        self.assertTrue(wait_for(lambda: ctx.counters.sent == 3))
        self.assertEqual((ctx.counters.streamed, ctx.counters.skipped, ctx.counters.reused),
                         (1, 1, 1))
        r.close()

    def test_native_classes_flag_per_request(self):
        """WANT_NATIVE_CLASSES: 7 native colours + CLASS_FRAC_NATIVE on that response only;
        without the bit the bytes are the health ones; streaming requests honour it too."""
        ctx = self.ctx()
        self.assertTrue(ctx.native_supported)
        r = Replier(ctx, self.board, ("pair", 7))
        ids = self.ids()
        ids[:, :16] = 0                                               # 1/4 BLEACHED, 3/4 ALIVE
        nat = np.bincount(ids.reshape(-1), minlength=7).astype(np.float32) / ids.size
        health = np.array([0.75, 0.25, 0, 0], np.float32)
        nf = P.FLAG_WANT_NATIVE_CLASSES
        for fid, flags in ((1, 0), (2, nf), (3, nf | P.FLAG_STREAM_REPLIES)):
            e = r.expect(fid, 64, 36, 2, flags, 0, time.perf_counter())
            r.done(e, ids, health, None, 5.0, native_frac=nat)
        plain, a, b = (read_response(self.client) for _ in range(3))
        self.assertNotIn(P.BLOCK_CLASS_FRAC_NATIVE, plain["blocks"])
        self.assertEqual(plain["mask_rgba"][:4], bytes(PP.HEALTH_PALETTE_RGBA[1]))  # BLEACHED white
        self.assertEqual(plain["mask_rgba"][-4:], LIVE_RGBA)
        for x in (a, b):
            px = np.frombuffer(x["mask_rgba"], np.uint8).reshape(36, 64, 4)
            self.assertEqual(tuple(px[0, 0]), PP.SEVEN_PALETTE_RGBA[0])        # BLEACHED yellow
            self.assertEqual(tuple(px[0, -1]), PP.SEVEN_PALETTE_RGBA[2])       # ALIVE green
            np.testing.assert_allclose(x["class_frac_native"], nat)
            self.assertEqual(len(x["class_frac_native"]), 7)
            np.testing.assert_allclose(x["class_frac"], health)                # still 4 health
        r.close()

    def test_native_classes_ignored_by_a_four_class_model(self):
        ctx = ReplyContext(PP.to_health_table(4), None, log=self.logs.append)
        self.assertFalse(ctx.native_supported)
        r = Replier(ctx, self.board, ("pair", 8))
        e = r.expect(1, 64, 36, 2, P.FLAG_WANT_NATIVE_CLASSES, 0, time.perf_counter())
        r.done(e, np.zeros((36, 64), np.uint8), np.array([1, 0, 0, 0], np.float32), None, 5.0,
               native_frac=np.array([1, 0, 0, 0], np.float32))
        a = read_response(self.client)
        self.assertNotIn(P.BLOCK_CLASS_FRAC_NATIVE, a["blocks"])
        self.assertEqual(a["mask_rgba"], LIVE_RGBA * (64 * 36))
        r.close()

    def test_inference_failure_is_answered_with_error(self):
        ctx = self.ctx()
        r = Replier(ctx, self.board, ("pair", 3))
        e = r.expect(7, 64, 36, 2, 0, 0, time.perf_counter())
        r.failed(e, "inference failed on the board: boom")
        a = read_response(self.client)
        self.assertEqual(a["frame_id"], 7)
        self.assertEqual(a["mask_rgba"], OTHER_RGBA * (64 * 36))
        self.assertIn("boom", a["error"])
        self.assertTrue(wait_for(lambda: ctx.counters.failed == 1))
        r.close()

    def test_slow_client_never_blocks_the_caller_and_is_dropped(self):
        ctx = self.ctx(send_timeout=0.3)
        self.board.settimeout(0.3)
        r = Replier(ctx, self.board, ("pair", 4))
        big = self.ids(1080, 1920)                                        # 8 MB per response
        worst = 0.0
        for fid in range(1, 9):                                           # the client never reads
            e = r.expect(fid, 1920, 1080, 2, 0, 0, time.perf_counter())
            if e is None:
                break
            t0 = time.perf_counter()
            r.done(e, big, np.array([1, 0, 0, 0], np.float32), None, 1.0)
            worst = max(worst, time.perf_counter() - t0)
        self.assertLess(worst, 0.05)                                      # an append, no socket
        self.assertTrue(wait_for(lambda: ctx.counters.closed_slow >= 1))
        self.assertTrue(wait_for(lambda: r.closed))
        self.assertIsNone(r.expect(99, 64, 36, 2, 0, 0, time.perf_counter()))
        self.assertTrue(any("longer than" in m for m in self.logs), self.logs)

    def test_client_that_never_reads_is_capped(self):
        ctx = self.ctx(max_pending=4)
        r = Replier(ctx, self.board, ("pair", 5))
        got = [r.expect(i, 64, 36, 2, 0, 0, time.perf_counter()) for i in range(1, 7)]
        self.assertEqual(sum(e is not None for e in got), 4)
        self.assertTrue(r.closed)


if __name__ == "__main__":
    unittest.main()
