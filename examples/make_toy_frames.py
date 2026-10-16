"""Create artificial line-pattern frames for an installation check, without clinical data."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('data/toy_frames'))
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output directory must be empty')
    a.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    scene = np.zeros((320, 320), np.uint8)
    cv2.circle(scene, (160, 160), 145, 45, -1)
    for _ in range(55):
        x, y = rng.integers(40, 280, size=2)
        dx, dy = rng.integers(-28, 29, size=2)
        cv2.line(scene, (int(x), int(y)), (int(x+dx), int(y+dy)),
                 int(rng.integers(90, 230)), int(rng.integers(1, 4)), cv2.LINE_AA)
    for i in range(8):
        transform = np.array([[1, 0, i*2], [0, 1, i]], dtype=np.float32)
        frame = cv2.warpAffine(scene, transform, (320, 320))
        ok, encoded = cv2.imencode('.png', frame)
        if not ok:
            raise OSError('PNG encoding failed')
        encoded.tofile(a.output/f'frame_{i:04d}.png')
    print('Created 8 artificial frames; these are an installation example, not validation data.')


if __name__ == '__main__':
    main()
