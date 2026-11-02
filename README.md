# ReefWatch

Coral reef health monitoring: a segmentation model that finds coral bleaching in underwater photos,
designed to run on a Kria KV260 FPGA aboard an underwater vehicle.

| folder | what |
|---|---|
| [`segmentation/`](segmentation/) | the milestone 1 segmentation model: weights, training code, the 100-photo bleaching benchmark, results and logs |
| [`board/`](board/) | the Kria KV260 program: receives camera frames + pose from the Unreal simulator over Ethernet, runs the model on the DPU and the reef health map on the ARM cores (`fpga-board` branch) |
| `runtime/`, `data/` | what the board program needs next to it: the torch-free wire protocol + health map code, and the map inputs |

Milestone 1 result: **bleached-vs-not-bleached mIoU 0.708** on 100 held-out photos from five reef regions
(Kenya, Red Sea, Chagos, Norfolk Island, La Paz), up from 0.497 for the same network trained on CoralScapes alone.
See [`segmentation/README.md`](segmentation/README.md).

## Running on the KV260 (`fpga-board` branch)

[`board/`](board/) holds the board program with the milestone model compiled for the KV260's
B4096 DPU (`board/models/coral_milestone_7c_512x768.xmodel`, plus a 512x1024 build). Unreal
streams frames and camera poses to it. The board segments them on the DPU, builds the 2.5D reef
health map and saves the results. With `--reply`, it also sends the masks and the map back, so
the simulator's HUD shows the board's output.

```sh
git clone --branch fpga-board https://github.com/chgaobt/ReefWatch.git
cd ReefWatch/board
python3 -m unittest discover -s tests -v
./run_board.sh --backend vart --map-hz 1 --reply
```

Start with [`board/SETUP_UBUNTU.md`](board/SETUP_UBUNTU.md) on a freshly flashed board (Vitis AI
runtime, the B4096 DPU, network). [`board/README.md`](board/README.md) has the rest.
