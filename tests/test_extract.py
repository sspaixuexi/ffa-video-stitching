from pathlib import Path
import json
import cv2
import numpy as np
import pytest

from ffa_stitch.extract import main


def test_video_sampling_crop_and_frame_order(tmp_path):
    video = tmp_path/'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64, 48))
    if not writer.isOpened():
        pytest.skip('MJPG writer unavailable')
    for i in range(7):
        image = np.full((48, 64, 3), 20+i*25, np.uint8)
        writer.write(image)
    writer.release()
    out = tmp_path/'frames'
    main(['--video', str(video), '--output', str(out), '--stride', '2',
          '--start', '1', '--stop', '6', '--crop', '8', '4', '32', '24'])
    manifest = json.loads((out/'extraction.json').read_text())
    assert [r['decoded_frame_index'] for r in manifest['frames']] == [1, 3, 5]
    images = [cv2.imread(str(out/r['file']), 0) for r in manifest['frames']]
    assert all(im.shape == (24, 32) for im in images)
    assert images[0].mean() < images[1].mean() < images[2].mean()


def test_preserves_existing_frame_directory(tmp_path):
    existing = tmp_path/'frames'
    existing.mkdir()
    (existing/'keep.txt').write_text('preserve')
    with pytest.raises(SystemExit):
        main(['--video', str(tmp_path/'missing.mp4'), '--output', str(existing)])
    assert (existing/'keep.txt').read_text() == 'preserve'
