# USB webcam detection on Raspberry Pi 5 (Ubuntu 22.04, 64-bit)

The included `models/helmet.pt` is the trained YOLOv8s model. It detects five helmet colors, trained on daytime 3–5 m data at 0°, 45°, and 90°. It does not measure distance or identify people without helmets. Night performance has not been validated.

## Install

Run inside this folder on the Pi. `uname -m` should print `aarch64`.

```bash
git clone https://github.com/srinjoybahuti/helmet-testing.git
cd helmet-testing
uname -m
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0 v4l-utils
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pi.txt
sha256sum -c models/SHA256SUMS
v4l2-ctl --list-devices
```

Ubuntu 22.04 normally supplies Python 3.10. Install in a fresh environment on the Pi; do not copy the desktop `venv`. No CUDA is required on the Pi.

## Test the USB webcam

With a monitor and desktop session:

```bash
python webcam.py --source 0 --show
```

Over SSH without a display:

```bash
python webcam.py --source 0 --save
```

The script prints processing FPS and current helmet counts about once per second. `--save` writes an annotated video into a new `predictions/webcam*` folder. Press Ctrl+C to stop; in the preview window, `q` also quits. Processing FPS includes startup and is not a guaranteed camera frame rate. Saved webcam video uses the camera's nominal FPS, so playback can be faster than wall time if inference drops frames.

If needed, change `--source 0` to `--source 1` to select a different video device. Use `v4l2-ctl --list-devices` to identify the capture device (some webcams expose extra metadata devices). If access is denied, add your user to the video group, then log out and back in:

```bash
sudo usermod -aG video "$USER"
```

Other tests:

```bash
# Stop after 100 processed frames.
python webcam.py --source 0 --max-frames 100
# Test a recording without a webcam.
python webcam.py --source test.mp4 --save
# Higher resolution for small helmets, at a speed cost.
python webcam.py --source 0 --imgsz 960 --conf 0.35 --show
```

The default inference size is 640 for CPU testing. Training/test metrics were measured at 960, so they do not directly describe the 640 deployment. Actual Pi FPS and USB webcam behavior must be measured on the device; local testing used a recorded video on an x86 CPU.

## Optional NCNN acceleration

Ultralytics recommends NCNN for Raspberry Pi CPU inference. Export on the desktop or Pi; the resulting folder can be copied to the Pi. The initial export may download conversion tools and install additional packages.

```bash
python -c "from ultralytics import YOLO; YOLO('models/helmet.pt').export(format='ncnn', imgsz=640, device='cpu')"
python webcam.py --model models/helmet_ncnn_model --source 0 --imgsz 640 --show
```

Use the same `--imgsz` as the export. If the runtime requests NCNN, run `python -m pip install ncnn`. The `.pt` path above remains the baseline if NCNN dependencies are unavailable for your OS/Python combination. YOLOv8s is larger than YOLOv8n; real-time frame rates on a Pi are not promised.

Reference: [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/).
