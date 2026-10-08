"""Test the helmet model with a USB webcam or a recorded video."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/helmet.pt',
                        help='Trained .pt file or exported NCNN directory')
    parser.add_argument('--source', default='0', help='USB webcam index (0, 1, ...) or video path')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference size; try 960 for smaller helmets')
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--device', default='cpu', help='Use cpu on Raspberry Pi; 0 for an NVIDIA GPU')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--show', action='store_true', help='Display window (requires desktop; q quits)')
    parser.add_argument('--save', action='store_true', help='Save annotated video under predictions/')
    parser.add_argument('--max-frames', type=int, default=0, help='Stop after N frames; 0 means unlimited')
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f'Model not found: {args.model}')
    if args.imgsz <= 0 or args.imgsz % 32 or args.threads < 1 or args.max_frames < 0:
        parser.error('imgsz must be a positive multiple of 32, threads >= 1, max-frames >= 0')
    if not 0 < args.conf < 1:
        parser.error('conf must be between 0 and 1')
    if not args.source.isdecimal() and not Path(args.source).is_file():
        parser.error('source must be a webcam index or an existing video file')
    if args.show and os.name != 'nt' and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('--show needs a desktop display; omit it for SSH/headless use')

    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    model = YOLO(str(args.model), task='detect')
    print(f'Model: {args.model}\nSource: {args.source}; device: {args.device}; imgsz: {args.imgsz}', flush=True)
    print('Detects helmet colors only. Press Ctrl+C to stop.', flush=True)
    results = model.predict(source=args.source, stream=True, stream_buffer=False,
                            imgsz=args.imgsz, conf=args.conf, device=args.device,
                            show=args.show, save=args.save, verbose=False,
                            project=str(ROOT / 'predictions'), name='webcam')
    count = 0
    started = last_report = time.monotonic()
    try:
        for result in results:
            count += 1
            now = time.monotonic()
            if count == 1 or now - last_report >= 1:
                counts = Counter(result.names[int(c)] for c in result.boxes.cls.tolist())
                print(f'frame={count} average_fps={count / max(now-started, 1e-6):.2f} '
                      f'detections={dict(counts)}', flush=True)
                last_report = now
            if args.max_frames and count >= args.max_frames:
                break
    except KeyboardInterrupt:
        print('\nStopped by user.', flush=True)
    finally:
        results.close()
        if args.show:
            import cv2
            cv2.destroyAllWindows()
    if not count:
        raise SystemExit('No frames received. Check the webcam index, permissions, or video file.')
    print(f'Processed {count} frames.', flush=True)


if __name__ == '__main__':
    main()
