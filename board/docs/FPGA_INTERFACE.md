# The Kria KV260 interface: one-way, detection and map on the board

Status: the board program (`board/`) and Unreal's send-only mode are built and have run end to
end **on the dev PC**. The board code is ready to copy, but **nothing has run on a KV260 yet**:
board fps, the DPU/VART calls and on-board accuracy are unmeasured. Setup and the first-hour
checklist are in `board/README.md`.

## The design

The link from the simulator to the board is **one-way**. Unreal behaves like the AUV's camera
plus navigation unit: it streams each newest frame together with the capture-camera pose, and it
never waits for, or reads, an answer. The board does everything else and keeps the results.

```
Unreal (PC, -sendonly)                                 KV260 (board/reefwatch_board)
  SegCapture 640x360 RGB8 @ 10 Hz                        reader thread: drain socket -> newest-frame slot
  + pose (SegCapture transform, FOV, time)  --TCP 5556-->  pipeline: int8 LUT + reflect pad -> DPU -> argmax
  CSG2 request bytes, unchanged                                     -> 7->4 health ids, class fractions
  no response is read                                      map thread: newest posed frame, <= 3 folds/s
                                                           saver: out/ every 60 s, on disconnect, on exit
  board/tools/viewer.py  <--TCP 5557 (optional)--          status: JSON + minimap PNG, <= 1/s, write-only
```

* **Frame link (TCP 5556).** The bytes are exactly the CSG2 requests of `docs/PROTOCOL.md`: the
  64-byte header carries the pose, followed by the RGB8 payload. The board never writes on this
  socket. One client at a time; a new connection replaces the old one. The board drains the
  socket continuously into a single newest-frame slot, so a slow board drops frames (counted)
  instead of backing up TCP and stalling Unreal. RESET_MAP is turned into a generation counter on
  its rising edge, so a reset cannot be lost with a dropped frame.
* **Unreal side.** `ULiveSegComponent::bSendOnly`, or `-sendonly` on the command line;
  `-segserver=<ip>` sets `ServerHost` and `-segport=<n>` sets `ServerPort`. The same capture,
  readback and newest-frame pending slot are used. The worker writes each newest frame at
  CaptureHz and goes straight back for the next one. If a send blocks, newer frames replace the
  pending one. A send stuck past the 2 s timeout, or a closed peer, triggers a reconnect with the
  usual 0.5 s doubling to 5 s backoff. RESET_MAP counts as delivered once the request carrying it
  is on the wire. The HUD shows `STREAMING TO <host>:<port> · N fps sent` instead of the mask
  box, and the minimap and class bars are hidden. `bOnline` means "connected", so `-autosurvey`
  starts as soon as the stream is up. An `AUVPERF_SEND` log line every 5 s reports the send rate.
  With the flag off, the client and its bytes are unchanged.
* **Status port (TCP 5557, optional).** It exists only for plotting during development. Any
  number of viewers can connect, and the board only writes. The message is
  `"RWST" | json_len u32 | png_len u32 | JSON | PNG`, at most once a second; the minimap PNG
  rides along only when it changed. Each viewer has its own sender thread and newest-message
  slot, so the pipeline never waits on a viewer. `--no-status` turns it off, and the final
  product leaves it off.
* **Results stay on the board.** `out/board_<stamp>/` holds `healthmap_state.npz`,
  `class_map.png`, `minimap_final.png`, `metrics.json` (MAP_METRICS plus the board counters)
  and `status.jsonl`. They are written every 60 s, when the client disconnects and on exit.

## The model on the DPU (preprocessing contract)

| item | value | source |
|---|---|---|
| network | ResNet-34 + DAPPM on C5 + FPN-lite (`fpn_resnet34_dappm`), 7 classes `BLEACHED, DEAD, ALIVE, ROCK, OCEAN, SAND, OTHER` | `runtime/reefwatch_runtime/model.py` |
| builds | `coral_milestone_7c_512x768.xmodel` (**default**) and `_512x1024`, both from `MILESTONE_R34_DAPPM_7c.pt`; `DPUCZDX8G_ISA1_B4096`, Vitis AI 3.5, po2 INT8, **1 DPU subgraph** | `board/models/TENSORS_*.md` (xir dump) |
| input tensor | `[1,H,W,3]` int8 **NHWC**, RGB, fix_point **5**: `int8 = clamp(round(((rgb/255 - mean)/std) * 2^5), -128, 127)`, ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) | xir dump; the formula matches the quantizer's deploy-check int8 input bit for bit (`board/tests`) |
| padding | 640x360 -> reflect-pad bottom/right to the xmodel's H x W (replicate if a pad would reach the image size), the PC server's rule. The program reads H x W from the input tensor and hard-codes no size | `prepost.Int8Input` |
| output tensor | `[1,H,W,7]` int8 NHWC, fix_point **3**, **full resolution**: vai_q turned the final 4x bilinear resize into a transposed-depthwise conv on the DPU | xir dump; the deploy-check dump has the same layout |
| CPU part | the compiled graph's only CPU op is a fix2float. It is not run: argmax works on the int8 values (the scale is common to all channels), then the top-left 360x640 is cropped | `prepost.logits_to_ids` |
| after the ids | the same 7->4 table as the PC server (`health`: DEAD+ROCK -> DEGRADED), class fractions, the numpy health map (`runtime/reefwatch_runtime/healthmap`, unchanged) | `prepost.SEVEN_TO_HEALTH`, pinned to `model.py` by a test |
| stats | mean max-softmax, entropy and BLEACHED softmax, on one pixel in 4x4 (a full softmax is too costly on the A53) | `prepost.logit_stats` |

The ARM-side work per frame: three uint8->int8 table lookups (640x360), two slice copies for the
padding, a crop + argmax over 360x640x7 int8 values, and a 7-entry table lookup. The program
needs only numpy. `vart`, `xir` and `torch` are imported lazily.

### Why 512x768 for the simulator

On the PC I emulated the board's own pre/post-processing around the float model:
int8 LUT input, padding to the xmodel size, full-resolution output, crop + argmax
(`board/tools/measure_board_path.py`). I compared it with what the PC server sends on the bundled
Unreal frame plus 19 CoralScapes TEST crops at 640x360:

| xmodel input | pixel agreement, mean (20 frames) | worst frame | Unreal frame | Unreal frame, BLEACHED IoU |
|---|---:|---:|---:|---:|
| **512x768** | **99.71 %** | 98.66 % | 98.66 % | 0.90 |
| 512x1024 | 99.14 % | 93.89 % | 93.89 % | 0.61 |

At 512x768 the remaining 0.3-1.3 % comes from the int8 input quantization. At 512x1024 the
reflect padding covers 1.8x the area, and the DAPPM global and large-window pools see that
mirrored content: on the Unreal frame the BLEACHED fraction rises from 4.2 % to 6.8 %. The INT8
arithmetic inside the DPU is not part of this comparison. The quantizer's own check puts the
512x768 build at 0.5572 INT8 vs 0.5579 float mIoU (`board/models/TENSORS_512x768.md`).

## Measured on the dev PC (2026-10-25)

The board program ran on the PC: `tools\run_board_local.bat`, with Unreal on the same machine in
send-only mode, `-autosurvey` along the lawnmower route. Unreal was capped at 30 fps, capture at
10 Hz, localhost.

| run | rx fps | frames dropped | inference | map folds/s (cap 3) | map fold + metrics |
|---|---:|---:|---:|---:|---:|
| backend `none`, 45 s survey | 9.90 | 0 of 525 | 0.1 ms | 2.93 | 33 ms |
| backend `torch --emulate-dpu` (milestone .pt, 512x768, laptop GPU shared with Unreal), 90 s survey | 9.96 | 1 of 980 | 51.6 ms mean, 81 ms p95: pre 7.8 / model 36.1 / post 7.7 | 2.92 | 38 ms |
| `stream_frames.py --hz 0` (no Unreal, backend `none`) | ~4000 | the pipeline takes the newest only | - | 3.0 | 85-150 ms incl. minimap |

* Unreal sent 10.0 frames/s throughout (`AUVPERF_SEND sent_hz=10.00`, 0 client-side drops), at
  6.9 MB/s on the wire (55 Mbit/s, about 6 % of Gigabit Ethernet).
* Torch run: 272 frames folded into the map, 57.6 m2 surveyed, 21.5 % coverage. The map was
  saved on disconnect and on exit. 95 minimaps reached the viewer over the status port.
* Receive-to-ids latency on the board side was 54.6 ms mean in the torch run.
* The normal (request/response) mode re-ran afterwards with the flag off: masks at 10 Hz,
  rtt 26 ms, map on, as before.

**For the board these are not measurements.** The `pre`/`post` and map numbers above are numpy
on the laptop CPU, and the Cortex-A53 cores will be several times slower. That is why the map is
capped at 3 folds/s (`--map-hz`): frames in between are still detected; they only miss the map.
The board's DPU time for the 512x768 graph is unknown. `xdputil benchmark` on the board is the
first number to take.

## Latency budget (10 Hz capture)

| stage | budget | where |
|---|---:|---|
| capture + readback in Unreal | 1-3 game frames | PC |
| request on the wire | ~7 ms GbE (691 KB) | cable |
| receive + newest slot | < 1 ms | board, reader thread |
| int8 LUT + pad | measure (7.8 ms on the laptop) | board ARM |
| DPU | measure (`dpu` in the status line) | board DPU |
| crop + argmax + fractions | measure (7.7 ms on the laptop) | board ARM |
| map fold | off the detection path, <= 3/s | board ARM, own thread |

The inference thread keeps up at 10 Hz as long as pre + DPU + post stays under 100 ms. When it is
slower, the board detects fewer frames (newest wins) and Unreal is unaffected. Nothing on the PC
depends on the board's speed any more.

## What runs only on the board (unverified until it does)

* The Vitis AI Python bindings (`import vart, xir`) on the board's Python, and
  `vart.Runner.create_runner(subgraph, "run")` / `execute_async` / `wait` with int8 numpy
  buffers of the tensors' shapes. This is the standard 3.5 recipe; it has only been tested
  against fake modules.
* A loaded DPU whose fingerprint matches `DPUCZDX8G_ISA1_B4096` (`xdputil query`, `xmutil`).
* DPU time, ARM pre/post time, map fold rate and Ethernet throughput on the KV260.
* On-board INT8 accuracy against the quantizer's numbers.
* Python and numpy on the board image (expected: Kria Ubuntu 22.04, Python 3.10). The code is
  Python 3.8+ syntax, and its tests pass with only numpy on the path.

(Since then the team ran it on the KV260 under PetaLinux 2022.2 / VART 3.0: pre 23 + DPU 60 +
post 37 ms, about 8 fps, a map fold about 0.6 s. The board is now on Kria Ubuntu 22.04:
`board/SETUP_UBUNTU.md`.)

## Reply mode (`--reply`, 2026-11-02)

For demos the board can send its results back over the same connection, so the Unreal HUD shows
the board's masks and map with Unreal unchanged (normal mode, no `-sendonly`). It is opt-in.
Without `--reply` the receiver, the pipeline and the bytes on the wire are as described above.

```
Unreal (normal mode, one request in flight, 2 s deadline)        KV260 (--reply)
  CSG2 request N  ------------------------------------------>  reader: register N, newest-frame slot
                                                               pipeline: pre + DPU + post -> ids(N)
                                                                 -> append to the connection's writer (no socket call)
                                                                 -> map thread (unchanged, <= --map-hz)
  CSG2 response N  <-----------------------------------------  writer thread: colour + pack + send
  (mask, fractions, stats, MAP_METRICS, MINIMAP_PNG)            (socket timeout --reply-timeout = 2 s)
```

* **When.** Response N carries frame N's own mask and is sent as soon as frame N's ids exist. The
  client sends N+1 only after it has decoded response N, so the slot never drops a frame and the
  board runs at the round-trip rate: about 1 / (wire + pre + DPU + post + ~10 ms). By the
  PetaLinux numbers (23 + 60 + 37 ms) that is about 7 fps and ~130 ms per request, far inside
  Unreal's 2 s deadline.
* **Never blocking.** The pipeline only appends to the writer's list. Colouring, packing and the
  send run on the writer's own thread. A send that does not finish within 2 s (the client's own
  deadline), a send error, or 64 unanswered requests closes that connection. Unreal reconnects
  with its usual backoff. The map and the on-board saving keep running throughout.
* **Exactly one response per request** even with a misbehaving client: a request whose frame was
  replaced in the slot is answered with the next finished frame's mask (counted as `reused`).
  An inference error is answered with an all-OTHER mask plus `ERROR`.
* **Ready for a streaming client.** The writer already has a second policy, `processed`: answer
  only frames the pipeline actually ran, each carrying its own `frame_id`. Each result also keeps
  that frame's `world_time_s`. A future "real-time" HUD mode needs a request flag (e.g. a new
  `STREAM_REPLIES` bit in `flags`) that selects it: Unreal sends at camera rate without waiting,
  the board drops what it cannot process (newest-wins, as now), and replies asynchronously. Since
  the response header has no capture-time field, that mode would add an optional TLV block (e.g.
  id 7 `FRAME_INFO`: frame_id, world_time_s, board receive/done times). Unknown blocks are already
  skipped by both parsers, so this needs no version bump.

**Measured on the dev PC (2026-11-02).** The board program ran on the PC with
`--backend torch --emulate-dpu --map-hz 1 --reply` (laptop GPU shared with Unreal). Unreal ran in
normal mode, `-autosurvey -autosurveylimit=90`, localhost, 640x360 at 10 Hz:

| | value |
|---|---|
| requests received / responses sent | 966 / 966 (0 dropped in the slot, 0 reused, 0 errors, 0 slow-client closes) |
| minimaps sent | 98 (one per render at `--minimap-hz 1`) |
| Unreal `AUVPERF` during the survey (18 lines, 5 s each) | seg_fps 10.2 mean (9.2-13.3), mask_hz 9.8, rtt 84 ms mean (41-107), srv_total 82 ms, e2e 268 ms mean (166-509), client-side capture drops 10 |
| board status line | rx 9.9 fps, tx 9.9 fps / 9.5 MB/s, infer ~86 ms (pre 8 / torch 67 / post 11), map 1.0 fold/s |
| map in Unreal | map_frames 89, 52.6 m2 surveyed, 19.7 % coverage at the end of the survey; `segmentation ONLINE`, never OFFLINE |

The torch model stands in for the DPU here, so these are plumbing numbers, not board numbers.
On the KV260, expect ~7 fps of masks with ~130 ms round trips.

## History

The first version of this note (2026-10-19) weighed two request/response designs:
- **A**: the board replaces the PC server and answers every CSG2 request with a mask.
- **B**: the PC server keeps the map and forwards frames to a `KriaBackend` on the board,
  through a proposed `WANT_IDS` flag / `CLASS_IDS` block.

It recommended B first. Both kept a reply path and made the simulator wait on the board. The user
then chose the one-way design above, which matches how the product works (the AUV does not wait
for its detector) and keeps the PC out of the loop entirely. `KriaBackend`, `WANT_IDS` and
`CLASS_IDS` were never implemented. In 2026-11, design A came back as the opt-in `--reply`
(above) for demos, without changing the default.
