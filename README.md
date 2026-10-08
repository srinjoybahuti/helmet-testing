# Helmet testing on Raspberry Pi 5

USB webcam helmet-color detection with a trained YOLOv8s model, for Raspberry Pi 5 with 64-bit Ubuntu 22.04.

See [installation and webcam instructions](RASPBERRY_PI.md).

```bash
python webcam.py --source 0 --show
# Over SSH without a display:
python webcam.py --source 0 --save
```

The checkpoint is included in `models/helmet.pt`. It detects blue, green, red, white, and yellow helmets. Training used confirmed daytime 3–5 m images at 0°, 45°, and 90°. Night performance is unverified. The model does not measure distance or detect people without helmets.

Training split: 289 training, 47 validation, 57 test images, with 32 omitted boundary frames. Early stopping at epoch 36 selected epoch 16. Held-out test results at image size 960: precision 94.91%, recall 92.77%, mAP50 94.10%, mAP50–95 41.93%. The default webcam size is 640; its accuracy and Pi frame rate have not been measured. Test frames come from the same recordings as training, so these scores do not establish performance at new locations.

The script was tested on recorded video using an x86 CPU, including bounded execution and annotated video output. Hardware webcam testing on the Pi remains to be done.

Dataset attribution: IITU Safety-Helmet Dataset v1.0, Bektemyssova, Bykov, Keresh, Yergazy, and Shaikemelev (2025), CC BY 4.0. Original title: IITU Safety-Helmet Dataset: Drone-Captured and Ground-Level Images with Color Annotations.
