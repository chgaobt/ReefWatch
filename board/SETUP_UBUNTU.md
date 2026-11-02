# Setting up a freshly flashed KV260 (Kria Ubuntu 22.04)

This guide sets up the board program (`board/`) on a KV260 that runs the Kria Ubuntu 22.04 image
instead of PetaLinux. Every vendor step below cites its source. **None of these steps has been
tried on our board yet.** Where the sources leave a question open, the guide says so and gives
the command that settles it on the board.

What the program needs:

| need | why | check |
|---|---|---|
| Python 3.8+ with numpy | the whole program | `python3 -c "import numpy, sys; print(sys.version, numpy.__version__)"` |
| `import vart, xir` in that Python | `--backend vart` only | `python3 -c "import vart, xir; print('VART OK')"` |
| a loaded **DPUCZDX8G B4096** DPU, fingerprint **0x101000056010407** | the xmodels were compiled for exactly this DPU | `xdputil query` |
| TCP 5556 (frames, and replies with `--reply`) and 5557 (status) reachable | Unreal and the viewer connect to the board | `sudo ufw status` |
| a static **IPv4** address on the direct cable | Unreal's address parser and the board socket are IPv4-only | `ip -br addr` |

`--backend none` needs only numpy, so steps 1-4 already give a working pipe test (the network,
the map and reply mode) before the DPU is set up in step 5.

## 1. First boot

- Log in on the serial console (micro-USB, the second-numbered COM port, 115200 8N1) or on HDMI.
  The default login is `ubuntu` / `ubuntu`, and you must change the password at first login
  ([Kria Ubuntu 22.04 boot guide][kr260-boot]; the guide is written for the KR260, and the KV260
  uses the same image family).
- Kria-PYNQ (step 5, route A) states: "2022.1 boot firmware is recommended for Ubuntu 22.04 LTS
  user" and "Ubuntu may not boot with mismatched firmware" ([Kria-PYNQ][kria-pynq]).
- Check what you have:
  ```sh
  cat /etc/os-release        # Ubuntu 22.04.x LTS
  python3 --version          # 3.10.x on 22.04
  uname -r
  ```
- The AMD guide then adds the Xilinx PPAs and upgrades ([boot guide][kr260-boot]):
  ```sh
  sudo add-apt-repository ppa:xilinx-apps --yes
  sudo add-apt-repository ppa:ubuntu-xilinx/default --yes
  sudo add-apt-repository ppa:xilinx-apps/xilinx-drivers --yes
  sudo apt update && sudo apt upgrade
  ```
  (That guide also adds `ppa:lely/ppa`, a CANopen stack for the KR260 robotics kit. We do not need it.)
  These steps need internet on the board: use a router cable or Wi-Fi dongle for setup, then
  switch to the direct cable (step 3).
- `xmutil` (listapps / unloadapp / loadapp) is packaged for jammy in `ppa:ubuntu-xilinx/default`
  ([Launchpad][ppa-default]). Older guides install it through the `xlnx-config` snap
  (`sudo snap install xlnx-config --classic --channel=2.x`, then `sudo xlnx-config.sysinit`)
  ([daiphys KV260 notes][daiphys]). **Check first** with `which xmutil && sudo xmutil listapps`.
  Use the snap only if `xmutil` is missing.

## 2. numpy

```sh
python3 -c "import numpy; print(numpy.__version__)" || sudo apt install python3-numpy
```

`python3-numpy` 1.21.5 is in Ubuntu 22.04 (jammy) for arm64 ([packages.ubuntu.com][numpy-deb]).
The sources do not say whether the Kria image preinstalls it. For an offline board, see README
section 4.2 (the aarch64 wheel). Route A in step 5 runs inside the PYNQ virtual environment,
which has its own numpy: check it there with the same one-liner after `source /etc/profile.d/pynq_venv.sh`.

## 3. Static IPv4 on the direct cable

**Symptom.** On a direct PC-to-board cable there is no DHCP server, so the board ends up with only
an IPv6 link-local address (`fe80::...`). Unreal's `-segserver=` parser and the board's listening
socket are IPv4-only. Give both ends a fixed IPv4 address: **PC 192.168.1.1/24, board
192.168.1.2/24**, no gateway. If the PC already uses 192.168.1.x on another adapter, pick another
subnet for both ends, e.g. 192.168.50.x.

Find the wired interface and what manages it:

```sh
ip -br link                       # eth0 on the KV260 images seen so far; newer kernels may say end0
ls /etc/netplan/                  # netplan files
cat /etc/netplan/*.yaml           # look for "renderer: NetworkManager" or networkd
nmcli general 2>/dev/null         # answers only if NetworkManager runs
```

On the KV260 22.04 desktop image, `/etc/netplan/50-cloud-init.yaml` uses
`renderer: NetworkManager` with interface `eth0` ([weenslab KV260 guide][weenslab]). The sources
do not cover the server image. Use the variant that matches what you find:

- **NetworkManager** (keeps any other connections; `eth0` = your interface):
  ```sh
  sudo nmcli con add type ethernet ifname eth0 con-name reefwatch \
       ipv4.method manual ipv4.addresses 192.168.1.2/24 ipv6.method link-local
  sudo nmcli con up reefwatch
  ```
  To use a router again: `sudo nmcli con down reefwatch` (or `sudo nmcli con delete reefwatch`).
- **netplan** (`/etc/netplan/60-reefwatch.yaml`, then `sudo netplan apply`). This turns off DHCP
  on eth0; delete the file and apply again to go back to a router:
  ```yaml
  network:
    version: 2
    ethernets:
      eth0:
        dhcp4: false
        addresses: [192.168.1.2/24]
  ```
  If the existing file sets `renderer: NetworkManager`, keep that line in your file as well.
- **For this boot only:** `sudo ip addr add 192.168.1.2/24 dev eth0 && sudo ip link set eth0 up`.

Check with `ip -br addr show eth0`, which should list `192.168.1.2/24`. On the PC (administrator
console):

```bat
netsh interface ipv4 set address name="Ethernet" static 192.168.1.1 255.255.255.0
ping 192.168.1.2
```

(The adapter name comes from `netsh interface ipv4 show interfaces`. Windows blocks incoming
ping by default, so ping from the PC to the board, not the other way.)

## 4. Firewall

Ubuntu ships `ufw` disabled ([Ubuntu UFW wiki][ufw]). Check, and open the two ports if it is active:

```sh
sudo ufw status verbose
sudo ufw allow 5556/tcp           # frames in, replies out (--reply)
sudo ufw allow 5557/tcp           # status port for board/tools/viewer.py
sudo ufw allow 22/tcp             # ssh, BEFORE you ever run "sudo ufw enable"
```

The Windows firewall needs no change, because all connections are opened from the PC.

Pipe test (no DPU yet): on the board, `cd ~/ReefWatch/board && ./run_board.sh --map-hz 1 --reply`.
On the PC, run `tools\run_sim.bat -segserver=192.168.1.2` (ReefWatchSim repo). Screen B shows an
all-blue (OTHER) mask from the `none` backend, and the minimap fills.

## 5. The Vitis AI runtime and the B4096 DPU

**What is known.**

- Our xmodels were compiled with Vitis AI 3.5 for `DPUCZDX8G_ISA1_B4096`. On this board under
  PetaLinux 2022.2 with VART 3.0 they ran: pre 23 + DPU 60 + post 37 ms.
- For Ubuntu 22.04, the only Vitis AI packages in the Xilinx PPAs are **Vitis AI 2.5**: source
  package `vitis-ai 2.5.0.20220720.f8f2805-0ubuntu1` in `ppa:ubuntu-xilinx/sdk`, plus
  `xilinx-runtime` (XRT) 2.13 ([Launchpad ubuntu-xilinx/sdk][ppa-sdk]). AMD's
  `vitis-ai-runtime-3.5.0` download is a set of RPMs for PetaLinux images
  ([UG1354, installing the AI library][ug1354-install]).
- The Kria example apps (smartcam, aibox-reid) use a **B3136** DPU and "only support Vitis AI
  2.5.0". The B3136 fingerprint is `0x101000016010406`
  ([aibox-reid model guide][aibox-models]). Our xmodels will refuse that DPU.
- The only public trace of our fingerprint `0x101000056010407` we found is in AMD's
  Kria-RoboticsAI tutorial (Ubuntu 22.04 on a K26 SOM). There it shows up in a log the tutorial
  says can be ignored, printed while running the **DPU-PYNQ Vitis AI 3.5** overlay
  ([Kria-RoboticsAI][roboticsai]). That DPU-PYNQ branch "supports PYNQ 3.0 and Vitis AI 3.5.0"
  and lists the **KV260 overlay as B4096**, 1 core ([DPU-PYNQ design_contest_3.5][dpu-pynq]).
- A B4096 firmware app for the KV260 also exists for `xmutil`: `benchmark-b4096` in
  [kria-apps-firmware][apps-fw], installable as `xlnx-firmware-kv260-benchmark-b4096`
  ([daiphys][daiphys]). It dates from the 2022.1 / Vitis AI 2.5 era. **Its fingerprint is not
  documented anywhere we found, so it may or may not be 0x101000056010407.**
- **Not established:** whether a Vitis AI 3.5 xmodel runs on the Vitis AI 2.5 runtime. A PYNQ
  maintainer told a KV260 Ubuntu 22.04 user with a 3.5 model to update the runtime manually or use
  Vitis AI 2.5 ([PYNQ forum][pynq-forum]). Our models did run on VART 3.0 under PetaLinux.

That leaves two routes. **Route B is quicker to try. Route A is the documented Vitis AI 3.5 path
to our fingerprint.** Try B first. If `xdputil query` does not show 0x101000056010407, or the
model refuses to load, `sudo xmutil unloadapp` and use A.

### Route B: apt firmware + apt Vitis AI 2.5 (about 10 minutes, fingerprint unverified)

From [daiphys][daiphys] (package names as given there; `vitis-ai-library` is a binary of the
`vitis-ai` source package in `ppa:ubuntu-xilinx/sdk` ([Launchpad][ppa-sdk])):

```sh
sudo add-apt-repository ppa:ubuntu-xilinx/sdk --yes && sudo apt update
sudo apt install xlnx-firmware-kv260-benchmark-b4096
sudo apt install vitis-ai-library            # pulls vitis-ai-runtime (VART/XIR) 2.5
sudo xmutil listapps
sudo xmutil unloadapp
sudo xmutil loadapp kv260-benchmark-b4096
```

(`xmutil` usage as in the [smartcam deployment guide][smartcam]. We could not confirm which PPA
carries `xlnx-firmware-kv260-benchmark-b4096`. If apt does not find it, `apt-cache search
xlnx-firmware` lists what the configured PPAs offer.)

Then check, in this order:

```sh
xdputil query                 # want: DPUCZDX8G, B4096, fingerprint 0x101000056010407
cat /etc/vart.conf            # VART reads the xclbin path here; the file it names must exist
python3 -c "import vart, xir; print('VART OK')"
cd ~/ReefWatch/board && python3 tools/check_dpu_reference.py     # want: max |d| 0
```

Unconfirmed by the sources: whether the 2.5 packages ship the Python `vart`/`xir` bindings for the
system Python 3.10, and which package provides `xdputil`. If the bindings are missing:
`dpkg -L vitis-ai-runtime | grep -i python` and
`find / -name "vart*.so" -path "*python*" 2>/dev/null`.

### Route B': the overlay the team used under PetaLinux

If the team still has the PetaLinux 2022.2 image's DPU files (`dpu.xclbin`, the `.bit.bin` /
`.dtbo` and `shell.json`), the same DPU can be loaded with `xmutil` on Ubuntu. `dfx-mgr` looks for
apps under `/lib/firmware/<company>/<app>/` containing those files
([Kria app deployment, target side][kria-target]). Copy them to e.g.
`/lib/firmware/xilinx/reefwatch-b4096/`, then `sudo xmutil loadapp reefwatch-b4096`. This DPU is
known to run our models, but you still need a runtime from route B or A.

### Route A: DPU-PYNQ with Vitis AI 3.5 (about 30 minutes, the documented 3.5 route)

AMD's Kria-RoboticsAI tutorial upgrades Kria Ubuntu 22.04 from Vitis AI 2.5 to 3.5 with one script
([Kria-RoboticsAI][roboticsai], [script][roboticsai-script]). The script:

- downloads `vai3.5_kr260.zip` from xilinx.com and runs its `target/runtime_deb/setup.sh`;
- copies `lack_lib/*` to `/usr/lib`;
- runs Kria-PYNQ's `install.sh -b KV260` (the script's default board, despite the file name);
- pip-installs DPU-PYNQ `design_contest_3.5` into the PYNQ venv;
- patches `xdputil` to run inside that venv.

The tutorial says it takes about 30 minutes. Run it as root, with internet:

```sh
sudo -i
git clone https://github.com/amd/Kria-RoboticsAI.git
cd Kria-RoboticsAI/files/scripts
bash install_update_kr260_to_vitisai35.sh
reboot
```

(Kria-PYNQ on its own, `sudo bash install.sh -b KV260`, installs the **Vitis AI 2.5** DPU-PYNQ
([Kria-PYNQ][kria-pynq]). The 3.5 overlay comes from the script above.)

DPU-PYNQ loads its DPU from Python, not with `xmutil`. `pynq_dpu.DpuOverlay("dpu.bit")` programs
the PL and copies the overlay's xclbin to `/usr/lib/dpu.xclbin`, and the overlay's
`check_vart_config()` writes `firmware: /usr/lib/dpu.xclbin` to `/etc/vart.conf`
([dpu.py][dpu-py]). The board program does the same with `--pynq-dpu-overlay dpu.bit`. Run it as
root inside the PYNQ venv; DPU-PYNQ's instructions for SSH/serial sessions are to log in as root
and source the two profile scripts ([DPU-PYNQ][dpu-pynq]):

```sh
sudo -i
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh
sudo xmutil unloadapp            # precaution: no xmutil app should hold the PL at the same time
cd /home/ubuntu/ReefWatch/board
python3 -c "import vart, xir, numpy, pynq_dpu; print('OK')"
./run_board.sh --backend vart --pynq-dpu-overlay dpu.bit --map-hz 1 [--reply]
```

While it runs, check the DPU from a second root shell (with the venv sourced, since the script
patched `xdputil` to run there): `xdputil query`. The fingerprint must be 0x101000056010407. Then
run `python3 tools/check_dpu_reference.py`, which should print `max |d| 0`. That tool does not load
the overlay itself, so run it while the board program is running, or load the overlay first in the
same venv with `python3 -c "from pynq_dpu import DpuOverlay; DpuOverlay('dpu.bit')"`.

Whether the PL keeps the overlay after the Python process exits is not covered by the sources.
`--pynq-dpu-overlay` loads it on every start, so the board program does not depend on it.

The Kria-RoboticsAI log also shows `cannot set read range! ... fingerprint 0x101000056010407 :
Invalid argument [22]`, which the tutorial says can be ignored.

## 6. Run it

On the board (`[--reply]` = send masks and the map back; route A adds `--pynq-dpu-overlay dpu.bit`
and runs as root in the PYNQ venv):

```sh
cd ~/ReefWatch/board
python3 -m unittest discover -s tests -v                    # numpy-only tests; ~10 s on a PC
./run_board.sh --backend vart --map-hz 1                    # one-way (Unreal -sendonly)
./run_board.sh --backend vart --map-hz 1 --reply            # masks + map back to Unreal
```

On the PC (ReefWatchSim repo):

```bat
rem one-way: the board keeps the results, the HUD shows "STREAMING TO ..."
tools\run_sim.bat -sendonly -segserver=192.168.1.2 -segport=5556
rem reply mode: the HUD shows the board's masks on screen B and the board's map
tools\run_sim.bat -segserver=192.168.1.2 -segport=5556
rem unattended 90 s survey that quits itself, logging AUVPERF every 5 s:
tools\run_sim.bat -segserver=192.168.1.2 -segport=5556 -autosurvey -autosurveylimit=90 -autosurveyquit -auvperf=5
rem optional: watch the board's status port
python board\tools\viewer.py --host 192.168.1.2
```

In reply mode, Unreal's log (`unreal\ReefWatchSim\Saved\Logs\ReefWatchSim.log`) shows
`segmentation ONLINE (frame N, CSG2)`, then `AUVPERF ... seg_fps=... rtt=... e2e=...` lines.
On the board, the status line shows `tx <fps>`.

The program has no PetaLinux-only assumptions. It never reads `/etc/vart.conf` or any fixed
path: models, map data and outputs are found relative to `board/`, and `run_board.sh` is POSIX
`sh`. The code is Python 3.8+ syntax, so it runs on 22.04's Python 3.10. `PYTHON=/path/python3
./run_board.sh` picks another interpreter.

## Sources

[kr260-boot]: https://xilinx.github.io/kria-apps-docs/kr260/linux_boot/ubuntu_22_04/build/html/docs/boot.html
[kria-pynq]: https://github.com/Xilinx/Kria-PYNQ
[dpu-pynq]: https://github.com/Xilinx/DPU-PYNQ/tree/design_contest_3.5
[dpu-py]: https://github.com/Xilinx/DPU-PYNQ/blob/design_contest_3.5/pynq_dpu/dpu.py
[roboticsai]: https://github.com/amd/Kria-RoboticsAI
[roboticsai-script]: https://github.com/amd/Kria-RoboticsAI/blob/main/files/scripts/install_update_kr260_to_vitisai35.sh
[apps-fw]: https://github.com/Xilinx/kria-apps-firmware
[daiphys]: https://www.daiphys.com/portal/fpga/xilinx/board/kv260.html
[ppa-sdk]: https://launchpad.net/~ubuntu-xilinx/+archive/ubuntu/sdk/+packages
[ppa-default]: https://launchpad.net/~ubuntu-xilinx/+archive/ubuntu/default/+packages
[aibox-models]: https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/aibox-reid/docs/customize_ai_models-aib.html
[smartcam]: https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/smartcamera/docs/app_deployment.html
[kria-target]: https://xilinx.github.io/kria-apps-docs/creating_applications/2022.1/build/html/docs/target.html
[ug1354-install]: https://docs.amd.com/r/en-US/ug1354-xilinx-ai-sdk/Step-3-Installing-the-AI-Library-Package
[ug1354-xdputil]: https://docs.amd.com/r/en-US/ug1354-xilinx-ai-sdk/Using-the-xdputil-Tool
[pynq-forum]: https://discuss.pynq.io/t/issues-when-loading-custom-xmodel-on-kria-pynq/8443
[weenslab]: https://weenslab.gitbook.io/pages/fpga-tutorials/fpga-boards-getting-started/getting-started-with-pynq-on-kria-kv260-vision-ai-starter-kit
[ufw]: https://wiki.ubuntu.com/UFW
[numpy-deb]: https://packages.ubuntu.com/jammy/python3-numpy

- Kria Ubuntu 22.04 boot guide (KR260): <https://xilinx.github.io/kria-apps-docs/kr260/linux_boot/ubuntu_22_04/build/html/docs/boot.html>
- Kria-PYNQ: <https://github.com/Xilinx/Kria-PYNQ>
- DPU-PYNQ, Vitis AI 3.5 branch: <https://github.com/Xilinx/DPU-PYNQ/tree/design_contest_3.5> and its `pynq_dpu/dpu.py`
- AMD Kria-RoboticsAI (Vitis AI 3.5 on Kria Ubuntu 22.04) and its install script: <https://github.com/amd/Kria-RoboticsAI>
- kria-apps-firmware: <https://github.com/Xilinx/kria-apps-firmware>
- daiphys KV260 notes (benchmark-b4096 firmware, vitis-ai-library on 22.04): <https://www.daiphys.com/portal/fpga/xilinx/board/kv260.html>
- Launchpad `ppa:ubuntu-xilinx/sdk` and `ppa:ubuntu-xilinx/default` package lists
- Kria aibox-reid model guide (B3136, fingerprint 0x101000016010406, Vitis AI 2.5 only), smartcam deployment (xmutil), Kria target-side app layout
- AMD UG1354 (AI library install; xdputil: `xdputil benchmark <xmodel> -i -1 <threads>`)
- PYNQ forum, Vitis AI 3.5 model on KV260 Ubuntu 22.04: <https://discuss.pynq.io/t/issues-when-loading-custom-xmodel-on-kria-pynq/8443>
- weenslab KV260 Ubuntu 22.04 guide (netplan, NetworkManager renderer, eth0)
- Ubuntu UFW wiki; packages.ubuntu.com `python3-numpy` (jammy)
