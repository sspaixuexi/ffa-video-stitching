"""Array-only correction and projection contracts; no study images required."""
import cv2
import numpy as np
import pytest

from ffa_stitch.ghost import full_coverage, postprocess
from ffa_stitch.ghost.sources import content_fov, project


def texture(size=288):
    y, x = np.indices((size, size))
    g = 95 + 24*np.sin(.24*x+.04*y) + 18*np.sin(.19*y) + 9*np.cos(.11*x-.09*y)
    return np.repeat(np.rint(g).clip(0, 255).astype(np.uint8)[..., None], 3, axis=2)


def fixture():
    source = texture()
    shifted = np.zeros_like(source)
    shifted[:, 6:] = source[:, :-6]
    baseline = source.copy()
    baseline[88:200, 88:200] = np.maximum(source, shifted)[88:200, 88:200]
    donors = np.stack([source, shifted])
    masks = np.ones(donors.shape[:3], bool)
    masks[1, :, :6] = False
    return baseline, donors, masks, masks.copy(), [[88, 88, 200, 200]], ['source_0', 'source_1']


@pytest.mark.parametrize('mode', ['B', 'C'])
def test_accepted_edit_retains_outside_and_single_source_core(mode):
    args = fixture()
    copies = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
    result = postprocess.postprocess(*args, mode=mode)
    assert result['summary']['accepted_regions'] == 1
    baseline, donors = args[:2]
    edit, core = result['edit_mask'], result['core_mask']
    assert np.array_equal(result['image'][~edit], baseline[~edit])
    assert core.any() and not np.any(core & ~edit)
    r = result['log'][0]
    x1,y1,x2,y2 = r['crop_xyxy']
    donor = donors[r['selected_donor_index'], y1:y2, x1:x2]
    matrix = np.array([[1,0,r['selected_dx']],[0,1,r['selected_dy']]], np.float32)
    proposal = cv2.warpAffine(donor, matrix, (x2-x1,y2-y1))
    proposal = np.rint(proposal.astype(np.float32)*r['selected_gain']+r['selected_offset']).clip(0,255).astype(np.uint8)
    assert np.array_equal(result['image'][y1:y2,x1:x2][core[y1:y2,x1:x2]], proposal[core[y1:y2,x1:x2]])
    for before, after in zip(copies[:4], args[:4]):
        assert np.array_equal(before, after)


@pytest.mark.parametrize('mode', ['B', 'C'])
def test_no_boxes_and_unavailable_overlap_preserve_input(mode):
    args = list(fixture())
    args[4] = []
    result = postprocess.postprocess(*args, mode=mode)
    assert np.array_equal(result['image'], args[0])
    args = list(fixture())
    args[2][0, :, 144:] = False
    args[2][1, :, :144] = False
    result = postprocess.postprocess(*args, mode=mode)
    assert result['summary']['accepted_regions'] == 0
    assert result['log'][0]['reason'] == 'insufficient_initial_dual_content'
    assert np.array_equal(result['image'], args[0])


@pytest.mark.parametrize('mode', ['B', 'C'])
def test_partial_support_never_creates_coverage(mode):
    args = list(fixture())
    args[2][:, :, 152:] = False
    result = postprocess.postprocess(*args, mode=mode)
    assert result['summary']['accepted_regions'] == 1
    assert not result['edit_mask'][:, 150:].any()
    assert result['summary']['edited_pixels'] < 112*112


def test_fcc_full_support_rejects_partial_region():
    args = list(fixture())
    masks = args[2].copy()
    masks[1, :, 152:] = False
    result = full_coverage.correct(args[0], args[1], masks, args[4], args[5], policy='R1')
    assert result['summary']['accepted_regions'] == 0
    assert np.array_equal(result['image'], args[0])


def test_content_mask_fills_large_fov_and_excludes_small_debris():
    work = np.zeros((128,160), np.uint8)
    cv2.circle(work, (50,64), 28, 20, -1)
    cv2.circle(work, (50,64), 8, 0, -1)
    work[3:5, 140:142] = 255
    mask = content_fov(work)
    assert mask[64,50] == 255
    assert mask[4,141] == 0
    assert not content_fov(np.zeros_like(work)).any()


def test_projection_preserves_intermediate_clipping_and_interpolation():
    work = texture(64)[...,0]
    coarse = np.full(work.shape, 255, np.uint8)
    h1 = np.array([[1,0,.35],[0,1,.4],[0,0,1]], float)
    h2 = np.array([[1,0,12.25],[0,1,8.45],[0,0,1]], float)
    route = [{'matrix':h1.tolist(),'canvas_size_wh':[48,48]},
             {'matrix':h2.tolist(),'canvas_size_wh':[80,80]}]
    donor, content, fov, support, h = project(work, coarse, route)
    expected = cv2.warpAffine(cv2.warpAffine(work,h1[:2],(48,48)),h2[:2],(80,80))
    single = cv2.warpAffine(work,(h2@h1)[:2],(80,80))
    assert np.array_equal(donor[...,0], expected)
    assert not np.array_equal(donor[...,0], single)
    assert np.allclose(h,h2@h1)
    assert not np.any(content & ~support) and not np.any(fov & ~support)
