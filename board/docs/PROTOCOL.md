# Wire contract: Unreal client <-> detection server (CSEG v1 / CSG2 v2)

The Unreal AUV client (`ULiveSegComponent` + `FSegNetClient`) streams camera frames to a detection
server over TCP and draws the class mask it gets back. Today the server is the PC process in
`runtime/reefwatch_runtime`; later a Kria KV260 can take its place (see `FPGA_INTERFACE.md`).
Anything that speaks this document is a valid server.

Every field below was checked against the C++ client source (`SegNetClient.cpp/.h`,
`LiveSegComponent.cpp/.h`, `AUVHUD.cpp`) and against the Python definition
(`runtime/reefwatch_runtime/protocol.py`); `runtime/tests/test_protocol.py` pins the sizes and
offsets.

## Transport

| item | value |
|---|---|
| socket | TCP, server listens, client connects. Default `127.0.0.1:5556` (`ULiveSegComponent::ServerHost` / `ServerPort`) |
| byte order | little-endian for every integer and float (`uint32`, IEEE `float32`) |
| exchange | strictly request -> response, **one request in flight**. The client never pipelines; frames captured while a request is out are dropped client-side, newest wins |
| clients | one at a time. A new connection replaces the current one at the next frame boundary; stale queued connections are closed |
| socket options | client sets `TCP_NODELAY`, 4 MiB send/receive buffers; server sets `TCP_NODELAY` |

## Version selection

The version is chosen by the **request magic**, never by a version field: v1's `frame_id`
sits at offset 4, exactly where v2's `version` is. The client speaks CSG2 by default
(`bUseProtocolV2 = true`). A server must answer each request in the version it arrived in.

| version | request magic | request header | response header | adds |
|---|---|---|---|---|
| v1 | `CSEG` | 24 B | 28 B + fixed payload | mask, class fractions, stats |
| v2 | `CSG2` | 64 B | 32 B + TLV payload | camera pose in; health-map metrics and minimap out |

## v1 `CSEG`

Request:

| offset | field | type | value |
|---:|---|---|---|
| 0 | magic | char[4] | `CSEG` |
| 4 | frame_id | uint32 | monotonic, from 1 |
| 8 | width | uint32 | 640 by default (`CaptureWidth`) |
| 12 | height | uint32 | 360 by default (`CaptureHeight`) |
| 16 | fmt | uint32 | 0 = RGB8, 1 = JPEG (server option; the Unreal client always sends 0) |
| 20 | payload_len | uint32 | `w*h*3` for fmt 0 |
| 24 | payload | bytes | RGB8, row-major, **top-left origin**, no row padding |

Response (exactly one per request):

| offset | field | type | value |
|---:|---|---|---|
| 0 | magic | char[4] | `CSEG` |
| 4 | frame_id | uint32 | echo of the request |
| 8 | width | uint32 | mask width (the server returns the request resolution) |
| 12 | height | uint32 | mask height |
| 16 | infer_ms | float32 | model time for this frame |
| 20 | total_ms | float32 | server receive -> send for this frame |
| 24 | payload_len | uint32 | **must equal** `w*h*4 + 16 + 16`; the client drops the connection otherwise |
| 28 | mask_rgba | uint8[w*h*4] | palette-coloured mask, R,G,B,A, alpha 255 |
| .. | class_frac | float32[4] | pixel fractions `[LIVE, BLEACHED, DEGRADED, OTHER]`, sum 1 |
| .. | stats | float32[4] | `[mean_max_prob, mean_entropy, mean_p_bleached, seg_fps_server]` |

## v2 `CSG2`

### Request: 64-byte header, then the same payload as v1

| offset | field | type | value |
|---:|---|---|---|
| 0 | magic | char[4] | `CSG2` |
| 4 | version | uint32 | 2 (any other value: the server drops the client) |
| 8 | frame_id | uint32 | monotonic, from 1 |
| 12 | width | uint32 | 640 |
| 16 | height | uint32 | 360 |
| 20 | fmt | uint32 | 0 = RGB8 (always 0 from Unreal), 1 = JPEG |
| 24 | payload_len | uint32 | `w*h*3` for fmt 0 |
| 28 | flags | uint32 | bit 0 `POSE_VALID`, bit 1 `WANT_MINIMAP`, bit 2 `RESET_MAP`, bit 3 `STREAM_REPLIES`, bit 4 `WANT_INFO`, bit 5 `WANT_NATIVE_CLASSES`, bits 8-11 `WATER_LOOK` (informational) |
| 32 | cam_x | float32 | capture camera world location, cm (`SegCapture->GetComponentLocation()`) |
| 36 | cam_y | float32 | |
| 40 | cam_z | float32 | |
| 44 | cam_pitch | float32 | capture camera rotation, degrees (`FRotator` Pitch) |
| 48 | cam_yaw | float32 | Yaw |
| 52 | cam_roll | float32 | Roll |
| 56 | fov_h_deg | float32 | horizontal FOV, degrees (`SegCapture->FOVAngle`, 78 by default) |
| 60 | world_time_s | float32 | `GetWorld()->GetTimeSeconds()` at capture |
| 64 | payload | bytes | as v1 |

Pose semantics (what the health map relies on):

* it is the **SceneCapture component's** world transform, not the pawn's (the camera sits 72 cm
  ahead of the pawn origin);
* it is stamped when `CaptureScene()` is issued and carried with the GPU readback, so it belongs
  to exactly these pixels even though the readback lands 1-3 frames later;
* world frame is Unreal's: left-handed, cm, +X forward, +Y right, +Z up.

| flag | value | meaning |
|---|---|---|
| `POSE_VALID` | `1 << 0` | the `cam_*` fields hold a real transform; the server may fold this frame into the map |
| `WANT_MINIMAP` | `1 << 1` | render a minimap if the server's rate cap allows. The client sets it every `0.95 / MinimapHz` s (5 Hz) |
| `RESET_MAP` | `1 << 2` | clear the map before folding this frame in. **Sticky** on the client: re-sent on every request until a response to a request that carried it is decoded |
| `STREAM_REPLIES` | `1 << 3` | streaming: the client sends at camera rate and does not wait; the server answers only the frames it processed, asynchronously, each with `FRAME_INFO`. See "Streaming replies" below |
| `WANT_INFO` | `1 << 4` | add `FRAME_INFO` (block 7) to every response and `SERVER_INFO` (block 8) to the first response of the connection. Implied by `STREAM_REPLIES`. The Unreal client sets it in every mode except send-only |
| `WANT_NATIVE_CLASSES` | `1 << 5` | per request: colour this response's `MASK_RGBA` in the model's **native** classes and add `CLASS_FRAC_NATIVE` (block 9). Honoured only by a server whose `SERVER_INFO` lists `native_classes` (a 7-class model); others ignore it and answer in the 4 health colours with no block 9. `CLASS_FRAC`, `STATS` and the map are unchanged (always health). The Unreal client sets it while "Mask classes: 7" is selected, never in send-only. See "Classes and palette" |
| `WATER_LOOK` | bits 8-11, `(flags >> 8) & 0xF` | the water-look preset active when the frame was captured: 0 not reported (older clients), 1 `training`, 2 `tabuhan`, 3 `clear`, 4 `off`, 15 another preset (`docs/WATER_LOOK.md`). Informational only: servers and the board ignore it, but `--record` keeps the raw header, so recordings say which look the model saw (`tools/water/measure_recording.py`). Other bits stay 0 |

With `POSE_VALID` clear the server answers blocks 1-3 only.

### Response: 32-byte header + TLV blocks

| offset | field | type | value |
|---:|---|---|---|
| 0 | magic | char[4] | `CSG2` |
| 4 | frame_id | uint32 | echo of the request |
| 8 | width | uint32 | mask width |
| 12 | height | uint32 | mask height |
| 16 | infer_ms | float32 | model time for this frame |
| 20 | map_ms | float32 | cost of the last map fold (+ minimap render), not of this frame |
| 24 | total_ms | float32 | server receive -> send |
| 28 | payload_len | uint32 | sum of all blocks incl. their 8-byte headers. Client limits: `>= w*h*4`, `<= 128 MiB` |
| 32 | blocks | | repeated `block_id uint32, block_len uint32, bytes[block_len]` |

| id | block | content | client rule |
|---:|---|---|---|
| 1 | `MASK_RGBA` | uint8[w*h*4], palette mask, alpha 255 | required, length must be `w*h*4` |
| 2 | `CLASS_FRAC` | float32[4] `[LIVE, BLEACHED, DEGRADED, OTHER]` | required, length 16 |
| 3 | `STATS` | float32[4] `[mean_max_prob, mean_entropy, mean_p_bleached, seg_fps_server]` | required, length 16 |
| 4 | `MAP_METRICS` | UTF-8 JSON, one line | optional |
| 5 | `MINIMAP_PNG` | PNG bytes, RGBA | optional |
| 6 | `ERROR` | UTF-8 text | optional, logged once per distinct text; never fatal |
| 7 | `FRAME_INFO` | 32 B, below | only on request (`WANT_INFO` / `STREAM_REPLIES`); a reader accepts a longer block and ignores the tail |
| 8 | `SERVER_INFO` | UTF-8 JSON, one line: `{"name": "PC GPU", "kind": "pc", ...}` | only on request, once per connection; `name` is the HUD's mask-source label; `health_classes` / `native_classes` drive the HUD legend (below) |
| 9 | `CLASS_FRAC_NATIVE` | uint32 `count`, then float32[`count`]: pixel share of each native class, in `native_classes` order (7 for the milestone model) | only when the request had `WANT_NATIVE_CLASSES` **and** the server honoured it; its presence means `MASK_RGBA` is in the native colours. Length must be `4 + 4*count`, `1 <= count <= 32`; a malformed block is ignored (the mask is then read as health) |

`FRAME_INFO` (little-endian):

| offset | field | type | value |
|---:|---|---|---|
| 0 | frame_id | uint32 | the frame the mask came from (= the header's frame_id) |
| 4 | info_flags | uint32 | bit 0 `STREAMED` (answered under `STREAM_REPLIES`), bit 1 `REUSED` (lockstep board: the mask is a later frame's) |
| 8 | world_time_s | float32 | that frame's capture world time, echoed from its request |
| 12 | queue_ms | float32 | request fully received -> inference started (server clock) |
| 16 | done_ms | float32 | request fully received -> class ids ready |
| 20 | frames_rx | uint32 | requests received on this connection |
| 24 | frames_done | uint32 | frames inferred on this connection |
| 28 | frames_skipped | uint32 | requests never inferred (replaced by a newer frame) on this connection |

`SERVER_INFO` names: the PC server sends `PC GPU` / `PC CPU` (`--server-name` overrides); the board
program `KV260 DPU` with `--backend vart`, otherwise `Board program (torch[, emulated DPU])` or
`Board program (no model)` (`--server-name` overrides).

`SERVER_INFO` class tables (2026-11; both servers), each `{"names": [...], "colors": ["#rrggbb", ...]}`
in class-id order:

```json
{"name":"KV260 DPU","kind":"board", ...,"classes":7,"mask_default":4,
 "health_classes":{"names":["LIVE","BLEACHED","DEGRADED","OTHER"],
                   "colors":["#00c850","#ffffff","#c83c28","#3c3cc8"]},
 "native_classes":{"names":["BLEACHED","DEAD","ALIVE","ROCK","OCEAN","SAND","OTHER"],
                   "colors":["#f4ec75","#ea7979","#67d88d","#6b4f3a","#68b9ef","#f38400","#e591d6"]}}
```

`native_classes` is present only when the server honours `WANT_NATIVE_CLASSES`. `mask_default` is
the class set of a response to a request **without** the flag (4; 7 for a PC server started with
`--mask-classes native`). The HUD builds its mask legend from these tables and falls back to the
built-in tables above / in "Classes and palette" when a server sends none.

A block id the client does not know is **skipped by its length** (both parsers), so new blocks
can be added without a version bump. A block whose length overruns the payload is a protocol
error (connection dropped). If blocks 1, 2 or 3 are missing or mis-sized the client drops the
connection.

### Map timing

The server folds a frame into the map **after** sending its response, so:

* `MAP_METRICS` on response N describes the map after frame N-1 (the first response of a
  connection carries none);
* a minimap requested on frame N arrives on response N+1 at the earliest; once rendered it
  rides on the next response whether or not that request asked again, and is sent once;
* the server caps minimap renders at `--minimap-hz` (5 Hz, with 5 % slack).

### `MAP_METRICS` JSON

```json
{"frame_id":1234,"frames":1233,"cell_cm":5.0,"min_obs":3,"min_votes":0.0,
 "surveyed_cells":41233,"surveyed_area_m2":103.08,"coverage_fraction":0.44,
 "bleached_fraction":0.021,"bleached_area_m2":2.16,"class_area_m2":[95.3,2.16,3.1,2.5],
 "mean_obs_surveyed":9.4,"mean_winning_prob":0.87,"cells_updated":1180,
 "rays_cast":14400,"rays_hit":14118,"hit_rate":0.98,"agent_cell":[151,127],"map_ms":15.2,
 "grid":{"nx":320,"ny":320,"cell_cm":5.0,"origin_cm":[-1050.0,-550.0],
         "mean_height_cm":-513.46,"reef_cells":90000},
 "gt":{"iou":0.83,"precision":0.90,"recall":0.91,"f1":0.90,"tp":1,"fp":2,"fn":3,
       "gt_area_m2_surveyed":4.1,"gt_area_m2_total":4.41}}
```

Keys the client reads (`ULiveSegComponent::ParseMapMetrics`, missing numbers default to 0):
`frame_id, frames, surveyed_cells, surveyed_area_m2, coverage_fraction, bleached_fraction,
bleached_area_m2, cells_updated, rays_cast, rays_hit, hit_rate, map_ms, class_area_m2[4],
agent_cell[2]`, `grid.{nx, ny, cell_cm, mean_height_cm, reef_cells, origin_cm[2]}` (sizes the
world overlay plane in `AHealthMapOverlayActor`) and, when present, `gt.{iou, precision, recall,
gt_area_m2_total}`. Areas are reef-surface m2 (one 5 cm cell = 0.0025 m2). A cell counts as
surveyed after `min_obs` rays. Grid convention: `world_x = origin_cm[0] + (ix + 0.5) * cell_cm`
(same for y); images are north-up (row 0 = largest X, column 0 = smallest Y). `gt` is present
only when the server has a ground-truth raster (the synthetic bleached colonies of the demo map).

### `MINIMAP_PNG`

North-up RGBA PNG, `--minimap-px` square (512), ~250-380 kB. Alpha is 0 on unsurveyed cells, so
the same image is the HUD inset (over a dark plate) and the world-overlay texture.

## Classes and palette

`CLASS_FRAC` and the map always use the 4 health classes (and so does the mask, unless the
request asks for native classes), in this order:

| id | class | mask colour (R,G,B) |
|---:|---|---|
| 0 | LIVE | (0, 200, 80) |
| 1 | BLEACHED | (255, 255, 255) |
| 2 | DEGRADED | (200, 60, 40) |
| 3 | OTHER | (60, 60, 200) |

The milestone model predicts 7 classes (`BLEACHED, DEAD, ALIVE, ROCK, OCEAN, SAND, OTHER`). The
server collapses them to health ids (`--seven-to-health health`, default: ALIVE -> LIVE,
BLEACHED -> BLEACHED, DEAD + ROCK -> DEGRADED, OCEAN + SAND + OTHER -> OTHER) for `CLASS_FRAC`,
`mean_p_bleached`'s class and the map. Native classes and their label-editor colours:

| id | class | mask colour |
|---:|---|---|
| 0 | BLEACHED | `#f4ec75` |
| 1 | DEAD | `#ea7979` |
| 2 | ALIVE | `#67d88d` |
| 3 | ROCK | `#6b4f3a` |
| 4 | OCEAN | `#68b9ef` |
| 5 | SAND | `#f38400` |
| 6 | OTHER | `#e591d6` |

Which colours a response's mask uses:

| request | server | mask | blocks |
|---|---|---|---|
| no `WANT_NATIVE_CLASSES` | any | 4 health colours (PC server: 7 native colours if started with `--mask-classes native`) | as before, byte for byte |
| `WANT_NATIVE_CLASSES` | PC server or board `--reply`, 7-class model | 7 native colours | + `CLASS_FRAC_NATIVE` (7 floats) |
| `WANT_NATIVE_CLASSES` | 4-class model, or an older server | 4 health colours | no block 9 |

The flag is read per request in every mode (synced, `STREAM_REPLIES`, lockstep board), so a
client can switch between requests on one connection. Cost on the board: a second precomputed
7-entry uint32 palette (the same one-gather colouring) and 7 floats from the histogram the
pipeline already computes. The mask is display-only: the client swizzles it to BGRA for a
`PF_B8G8R8A8` sRGB texture, and the map is fed class ids on the server side, never decoded colours.

**Unreal ("Mask classes" setting).** Settings page row "Mask classes: 4 health | 7 classes"
(saved as `mask_classes` in `Saved/ReefWatchSettings.json`), `-maskclasses=4|7` for one run,
console `RWMaskClasses 4|7` (or `RWMenu maskclasses=7`, saved). The choice is stamped on each
capture, so it takes effect on the next request without reconnecting. The legend, class bar and
percentages under screen B / the PIP follow the response: native names / colours from
`SERVER_INFO` and the fractions from `CLASS_FRAC_NATIVE` when block 9 is present, else the 4 health
classes (a server that ignores the bit stays at 4; the client logs it after 10 health replies).
The health-map legend and numbers are always the 4 health classes. `AUVPERF_DETECT` reports
`mask_classes=<shown>/<asked> mask_colours=<distinct colours in the newest mask>`.

`stats`: `mean_max_prob` = mean over pixels of the max softmax; `mean_entropy` = mean per-pixel
softmax entropy in nats (0..ln C); `mean_p_bleached` = mean softmax of the BLEACHED class;
`seg_fps_server` = EWMA of `1000 / total_ms`. The HUD's SEG FPS is measured client-side from
result arrivals; stats[3] is only logged.

## One-way use (send-only, the KV260)

With `ULiveSegComponent::bSendOnly` (`-sendonly`) the client writes the same CSG2 requests
(byte for byte, header as above) for each newest frame at `CaptureHz`, and never reads a
response. The receiver (`board/reefwatch_board`) never writes on the socket. Everything else in
this document that concerns requests still holds: framing, the flags, pose semantics, and the
malformed-request rule (the receiver closes the connection). Differences:

* there is no deadline, no response and no v1 fall-back;
* `RESET_MAP` is sticky only until the request carrying it has been written to the socket (TCP
  delivers it, and the receiver reads every header even when it drops the frame);
* `WANT_MINIMAP` is still set but ignored: the board renders minimaps for its own status port
  (TCP 5557, write-only, `board/reefwatch_board/status.py`).

See `docs/FPGA_INTERFACE.md`.

## Streaming replies (`STREAM_REPLIES`, 2026-11)

Unreal's "Real time" detection displays set `STREAM_REPLIES` on every request. Without the flag
(and without `WANT_INFO`) every server sends exactly the bytes it sent before; old servers ignore
both bits.

* **Client.** Sends each newest capture at camera rate (10 Hz) without waiting. A second thread
  reads responses as they arrive. The first request of a connection is a probe: if its response
  carries `FRAME_INFO` with `STREAMED`, the link free-runs; otherwise the server is an old one
  that answers every request, and the client paces one request in flight. The 2 s per-request
  deadline does not apply; a connection that answers nothing for 5 s
  (`StreamLivenessSeconds`) while requests are outstanding is closed and reconnected with the
  usual backoff. A peer that closes twice before any response makes the client fall back to the
  synced client (with its v1 fall-back) for the session.
* **Server.** From the first flagged request on, the connection is streaming (a client does not
  mix modes on one connection; switching the display opens a new connection). A reader drains
  the socket into a single newest-frame slot; replaced frames are skipped and counted. The newest
  frame is inferred and answered with its own `frame_id` and `FRAME_INFO`. Responses go out in
  increasing `frame_id`; there is at most one response per request and none for skipped frames.
  The PC server bounds each reply send by `--stream-send-timeout` (2 s), the board by
  `--reply-timeout`. CSEG v1 requests on a streaming connection are malformed.
* **RESET_MAP** is counted on its rising edge by streaming servers (a frame carrying it may be
  skipped, so the sticky flag usually rides on several requests). The client treats the reset as
  acknowledged by any response to a frame sent with or after it, and leaves one request without
  the bit if a newer reset is pending, so the next press is a fresh edge.
* **WANT_MINIMAP** of a skipped frame is inherited by the frame that replaces it.
* **Map timing**: as above; the map folds after the reply, newest-wins.

## Board reply mode (`--reply`)

With `--reply`, the board answers on the frame socket like the PC server, and Unreal runs in its
normal mode (no `-sendonly`). Everything in this document about responses then applies to the
board, with these board-specific points (`board/reefwatch_board/reply.py`):

* one response per request, in request order. `frame_id` is the id of the frame whose mask the
  response carries, which with the one-in-flight client is the request's own id;
* the mask is the 4 health colours (as `--mask-classes 4`), at the request resolution, or the 7
  native colours plus `CLASS_FRAC_NATIVE` for a request with `WANT_NATIVE_CLASSES` (7-class model);
* `infer_ms` = the board's pre + DPU + post for that frame; `map_ms` = the last map fold;
  `total_ms` = request fully received -> response packed. `stats[3]` is the EWMA of
  `1000 / total_ms`;
* `MAP_METRICS` is the latest map the board's map thread folded (at `--map-hz`, so it can lag
  the mask by more than one frame). It is omitted until the map has applied a pending `RESET_MAP`;
* `MINIMAP_PNG`: a minimap the board rendered (at `--minimap-hz`) that this client has not
  received yet, sent once, on the first response after a request with `WANT_MINIMAP`;
* `ERROR` once for a posed request when the board has no map, and with an all-OTHER mask if
  inference fails on a frame;
* a response the board cannot send within `--reply-timeout` (2 s) closes the connection;
* requests with `STREAM_REPLIES` are answered under the `processed` policy (only frames that
  ran, see "Streaming replies"); requests without it keep the one-response-per-request rule.

## Client behaviour a server must tolerate

* **Timeout** (synced mode; streaming uses the 5 s liveness rule above): no complete response within `OfflineTimeoutSeconds` (2 s) of sending -> the client
  closes the socket, shows OFFLINE and reconnects with backoff 0.5 s doubling to 5 s. The first
  forward at an unseen resolution must therefore be paid before listening (server
  `--warmup-res`, which must include the capture size).
* **v1 fall-back**: if the server closes the connection on two CSG2 attempts before any CSG2
  response was ever decoded, the client switches to CSEG (no map) and re-probes CSG2 every 10 s.
  A timeout does not count towards this, and an `ERROR` block never triggers it.
* **Resolution**: the client rebuilds its mask texture if the response width/height differ from
  the capture size, so a server may return a different resolution; this server always echoes
  the request size.
* `frame_id` is not checked against the request; it is logged and shown.

## Malformed requests

The server drops the connection (no response) on an unknown magic, a CSG2 version other than 2,
width/height outside 1..8192, `fmt` not 0/1, a fmt 0 `payload_len != w*h*3`, a payload over
256 MiB, or an undecodable JPEG. Map failures are reported as one `ERROR` block and segmentation
continues; with the map disabled, the first posed request gets
`ERROR "health map disabled on this server (--no-map or DEM missing)"`.
