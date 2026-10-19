# ReefWatch

Coral reef health monitoring: a segmentation model that finds coral bleaching in underwater photos,
designed to run on a Kria KV260 FPGA aboard an underwater vehicle.

| folder | what |
|---|---|
| [`segmentation/`](segmentation/) | the milestone 1 segmentation model: weights, training code, the 100-photo bleaching benchmark, results and logs |

Milestone 1 result: **bleached-vs-not-bleached mIoU 0.708** on 100 held-out photos from five reef regions
(Kenya, Red Sea, Chagos, Norfolk Island, La Paz), up from 0.497 for the same network trained on CoralScapes alone.
See [`segmentation/README.md`](segmentation/README.md).
