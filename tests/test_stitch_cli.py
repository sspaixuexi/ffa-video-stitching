"""Clinical-data-free checks of the public CLI and transform export."""
import json

import cv2
import numpy as np
import pytest

from ffa_stitch.core import Config, load_sequence, prepare_frames, stitch, transform_points
from ffa_stitch.matching import MatchEngine
from ffa_stitch.stitch_cli import main, write_png


def texture(seed=42):
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.integers(40,220,(256,320),dtype=np.uint8),(3,3),0)
    for _ in range(50):
        center = tuple(rng.integers([15,15],[305,241]).tolist())
        cv2.circle(image,center,int(rng.integers(2,8)),int(rng.integers(60,250)),-1)
    return image


def test_natural_input_order_and_resize_transform(tmp_path):
    # File names are sorted numerically; resize provenance retains original pixels.
    for number in [10,2,1]:
        write_png(tmp_path/f'frame_{number}.png',np.full((133,167),number,np.uint8))
    frames,paths = load_sequence(tmp_path)
    assert [p.name for p in paths]==['frame_1.png','frame_2.png','frame_10.png']
    assert frames[0]['image'].shape==(128,160)
    np.testing.assert_allclose(frames[0]['original_to_work'],np.diag([160/167,128/133,1]))


def test_nonempty_output_is_rejected_without_modifying_files(tmp_path,capsys):
    frame_dir = tmp_path/'frames'; frame_dir.mkdir()
    write_png(frame_dir/'frame_0.png',texture())
    output = tmp_path/'result'; output.mkdir()
    existing = output/'piece_000.png'
    existing.write_bytes(b'previous output remains unchanged')
    with pytest.raises(SystemExit) as error:
        main(['--frames',str(frame_dir),'--output',str(output),'--matcher','sift'])
    assert error.value.code==2
    assert 'output must be a new or empty directory' in capsys.readouterr().err
    assert list(output.iterdir())==[existing]
    assert existing.read_bytes()==b'previous output remains unchanged'


def test_cpu_cli_known_translation_and_export(tmp_path):
    frame_dir = tmp_path/'frames'; frame_dir.mkdir()
    output = tmp_path/'result'
    base = texture()
    for index,shift in enumerate([0,6,12]):
        translated = cv2.warpAffine(base,np.float32([[1,0,shift],[0,1,0]]),(320,256))
        write_png(frame_dir/f'frame_{index}.png',translated)
    assert main(['--frames',str(frame_dir),'--output',str(output),'--matcher','sift',
                 '--selection','fixed','--fixed-step','1','--no-partition','--threads','1'])==0
    manifest = json.loads((output/'stitch_manifest.json').read_text())
    assert manifest['keyframes']==[0,1,2]
    assert len(manifest['pieces'])==1
    piece = manifest['pieces'][0]
    assert (output/piece['image']).is_file() and (output/piece['mask']).is_file()
    assert all(not edge['registration_failed'] for edge in manifest['edges'])
    point = np.array([[140.,100.]])
    for index,shift in enumerate([0,6,12]):
        h = np.asarray(piece['transforms'][str(index)]['H_work_to_canvas'])
        np.testing.assert_allclose(transform_points(h,point+[[shift,0]]),point,atol=0.15)
        np.testing.assert_allclose(h,piece['transforms'][str(index)]['H_original_to_canvas'])
    assert all('/' not in f['file'] and '\\' not in f['file'] for f in manifest['frames'])


def test_match_cache_reuses_identical_images(tmp_path):
    a = texture(); b = np.roll(a,4,axis=1)
    first = MatchEngine('sift',cache_dir=tmp_path)
    expected = first.match(a,b)
    assert first.computed==1
    second = MatchEngine('sift',cache_dir=tmp_path)
    actual = second.match(a,b)
    assert second.hits_disk==1 and second.computed==0
    for x,y in zip(expected,actual):
        np.testing.assert_array_equal(x,y)


def test_local_and_global_transform_provenance():
    base = texture()
    images = [cv2.warpAffine(base,np.float32([[1,0,s],[0,1,0]]),(320,256)) for s in [0,6,12]]
    frames = prepare_frames(images)
    # A return to an older ROI starts a second local panorama.
    for frame,roi in zip(frames,[(0,0,100,100),(50,0,150,100),(10,0,110,100)]):
        frame['roi'] = roi
    result = stitch(frames,MatchEngine('sift'),Config(selection='fixed',fixed_step=1,canvas='dynamic'))
    assert result['local_groups']==[[0,1],[2]]
    assert len(result['pieces'])==1 and result['merges'][0]['success']
    piece = result['pieces'][0]
    for k,h in piece['transforms'].items():
        geometry = piece['source_geometry'][k]
        np.testing.assert_allclose(geometry['H_local_to_canvas']@geometry['H_work_to_local'],h,atol=1e-10)
        assert len(geometry['local_canvas_size'])==2
        assert len(geometry['projection_route'])==2
        accumulated = np.eye(3)
        for stage in geometry['projection_route']:
            accumulated = np.asarray(stage['matrix'])@accumulated
            assert len(stage['canvas_size_wh'])==2
        np.testing.assert_allclose(accumulated,h,atol=1e-10)
    assert not np.allclose(piece['source_geometry'][2]['H_local_to_canvas'],np.eye(3))
