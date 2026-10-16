"""Extract a reproducibly ordered frame sequence from a user-supplied video."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stride', type=int, default=1,
                   help='Keep every Nth decoded frame; no temporal resampling')
    p.add_argument('--crop', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'))
    p.add_argument('--start', type=int, default=0, help='First decoded frame index, inclusive')
    p.add_argument('--stop', type=int, help='Last decoded frame index, exclusive')
    a = p.parse_args(argv)
    if a.stride < 1 or a.start < 0 or (a.stop is not None and a.stop <= a.start):
        p.error('Use stride >= 1, start >= 0 and stop > start')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Use an empty output directory to avoid mixing frame sequences')
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        p.error('Cannot open the video')
    a.output.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if np.isfinite(fps) and fps > 0 else None
    records = []
    index = 0
    try:
        while a.stop is None or index < a.stop:
            ok, image = cap.read()
            if not ok:
                break
            if index >= a.start and (index - a.start) % a.stride == 0:
                if a.crop:
                    x, y, w, h = a.crop
                    if min(x, y) < 0 or min(w, h) < 1 or x+w > image.shape[1] or y+h > image.shape[0]:
                        raise ValueError('Crop must lie inside every decoded frame')
                    image = image[y:y+h, x:x+w]
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                name = f'frame_{index:06d}.png'
                encoded_ok, encoded = cv2.imencode('.png', gray)
                if not encoded_ok:
                    raise OSError('PNG encoding failed')
                encoded.tofile(a.output / name)
                records.append(dict(file=name, decoded_frame_index=index,
                                    nominal_time_s=index/fps if fps else None))
            index += 1
    finally:
        cap.release()
    if not records:
        raise ValueError('No frames extracted')
    (a.output/'extraction.json').write_text(json.dumps(dict(
        reported_fps=fps, stride=a.stride, crop=a.crop, start=a.start,
        stop_exclusive=a.stop, frames=records), indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'frames': len(records), 'reported_fps': fps}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
