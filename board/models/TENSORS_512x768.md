# coral_milestone_7c_512x768.xmodel -- tensors for the board program

Generated 2026-10-25 by `artifacts_quant/vitis/gen_tensors_md_hw.py` from the compiled xmodel (xir dump in the Vitis AI 3.5 docker, `dump_tensors_hw.py`; raw dump in TENSORS.json beside the xmodel in CoralUnreal/artifacts_quant/vitis/compiled_coral_milestone_7c_512x768/).

| item | value |
|---|---|
| xmodel | `coral_milestone_7c_512x768.xmodel`, md5 `1cf85891f70f6e3c0dd3e40dac67fdad` |
| target | `DPUCZDX8G_ISA1_B4096` (KV260), lib `libvart-dpu-runner.so` |
| DPU kernel (subgraph) | `subgraph_CoralSegDPU__CoralSegDPU_Conv2d_head__ret_349` |
| source checkpoint | `MILESTONE_R34_DAPPM_7c.pt` (fpn_resnet34_dappm, 7 classes), sha256 a3df84f2e3f57a19439fb62b56958498909a8dad21e7d06c7ca947c4bb5021c6 |
| quantization | vai_q_pytorch 3.5, per-tensor power-of-2 INT8, CLE + bias correction on (qcfg/r34.json), 32 CoralScapes-V2 TRAIN calibration frames |
| graph input | 512x768 (HxW), NHWC, RGB |
| subgraphs | 3 total: USER (1 ops), DPU (222 ops), CPU (1 ops); **DPU subgraphs: 1** |
| INT8 vs float sanity | INT8 0.5572 vs float 0.5579 mIoU, first 100 CoralScapes-V2 TEST frames, 7->4 collapse (ROCK->OTHER); each 512x1024 frame scored as two 512x768 windows at x=0 and x=256 (left half from the first, right half from the second); not comparable with the 512x1024 number |

## Subgraphs

| device | name | ops | inputs | outputs |
|---|---|---|---|---|
| USER | `subgraph_CoralSegDPU__input_0` | 1 | - | `CoralSegDPU__input_0_fix` [1, 512, 768, 3] xint8 fp=5 |
| DPU | `subgraph_CoralSegDPU__CoralSegDPU_Conv2d_head__ret_349` | 222 | `CoralSegDPU__input_0_fix` [1, 512, 768, 3] xint8 fp=5 | `CoralSegDPU__CoralSegDPU_Upsample_up_out__ret_fix` [1, 512, 768, 7] xint8 fp=3 |
| CPU | `subgraph_CoralSegDPU__CoralSegDPU_Upsample_up_out__ret_fix_` | 1 | `CoralSegDPU__CoralSegDPU_Upsample_up_out__ret_fix` [1, 512, 768, 7] xint8 fp=3 | `CoralSegDPU__CoralSegDPU_Upsample_up_out__ret_fix_` [1, 512, 768, 7] float32 |

## DPU input tensor

- name `CoralSegDPU__input_0_fix`, shape [1, 512, 768, 3] (N,H,W,C), dtype xint8, **fix_point 5**
- preprocessing (same as training): RGB uint8 -> x = (rgb/255 - mean)/std, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225); then int8 = clamp(round(x * 2^5), -128, 127).
  Equivalent per channel: int8 = clamp(round((rgb - 255*mean_c) * 2^5 / (255*std_c)), -128, 127).
- simulator frames: 640x360 (WxH) RGB -> reflect-pad bottom by 152 and right by 128 to 768x512, run, crop the output back to the top-left 360x640.

## DPU output tensor

- name `CoralSegDPU__CoralSegDPU_Upsample_up_out__ret_fix`, shape [1, 512, 768, 7] (N,H,W,C = 7 class logits), dtype xint8, **fix_point 3** (logit = int8 * 2^-3).
- The output is already at full input resolution: vai_q rewrote the final 4x bilinear `Upsample` into a `transposed-depthwise-conv2d-fix` (kernel 8, stride 4) that runs **on the DPU** (see TENSORS.json `output_tail_ops`).
- Class map = argmax over C directly on the int8 values (the fix_point scale is common to all channels, so no dequantisation is needed for argmax). Channel order: 0 BLEACHED, 1 DEAD, 2 ALIVE, 3 ROCK, 4 OCEAN, 5 SAND, 6 OTHER.
- 4-class health collapse used for scoring (LIVE 0, BLEACHED 1, DEGRADED 2, OTHER 3): LUT `1,2,0,3,3,3,3` (the live seg server's `--seven-to-health health` collapse instead maps DEAD+ROCK -> DEGRADED).

## CPU subgraph

- `CoralSegDPU__CoralSegDPU_Upsample_up_out__ret_fix_`: type **fix2float** {'fix_point': 3, 'bit_width': 8, 'if_signed': True, 'round_mode': 'DPU_ROUND'} -- input [1, 512, 768, 7] xint8, output [1, 512, 768, 7] float32.
- It is only the int8 -> float32 conversion of the DPU output (the vai_c 'CPU op' named after `Upsample_up_out` is a fix2float, not the resize). A VART `dpu_runner` on the DPU subgraph alone gives the int8 logits; `vitis::ai::GraphRunner` would run this conversion on the ARM.
