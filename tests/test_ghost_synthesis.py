"""Generator checks use an artificial line grid, without patient data."""
import importlib
import json
import cv2
import numpy as np
import pytest

from ffa_stitch.ghost.synthesize import load_sources, run


def grid(side=640):
    image = np.full((side,side,3),24,np.uint8)
    mask = np.zeros((side,side),np.uint8)
    for n in range(36,side-30,35):
        cv2.line(mask,(n,15),(n,side-15),255,3)
        cv2.line(mask,(15,n),(side-15,n),255,3)
    image[mask>0] = 220
    return image,mask


@pytest.mark.parametrize('profile', ['pretrain','adaptation'])
@pytest.mark.parametrize('strength', ['strong','medium','negative'])
def test_max_fusion_reference_and_label_geometry(profile,strength):
    engine = importlib.import_module('ffa_stitch.ghost.synthesis_'+profile)
    image,mask = grid()
    prepared = engine.prepare_image(image,mask)
    kwargs = {'region_kind':'standard'} if profile=='adaptation' else {}
    result = engine.synthesize(prepared,22,strength,max_attempts=64,**kwargs)
    m = result['metadata']; support = result['support_mask']
    assert np.array_equal(result['image'][~support],result['nominal_source'][~support])
    assert np.array_equal(result['effect_mask'],np.any(result['image']!=result['clean_reference'],axis=2))
    if profile=='adaptation':
        x0,y0,x1,y1 = m['warp_patch_bbox']
        local_transform = np.asarray(m['residual_patch_transform_2x3'])
    else:
        cx,cy = m['roi_center_xy']
        radius,margin = m['roi_radius'],m['roi_context_margin']
        x0,y0,x1,y1 = int(cx-radius-margin),int(cy-radius-margin),int(cx+radius+margin+1),int(cy+radius+margin+1)
        shift = np.array([[1,0,-x0],[0,1,-y0],[0,0,1.]])
        local_transform = (shift @ np.vstack([m['residual_transform_2x3'],[0,0,1.]]) @ np.linalg.inv(shift))[:2]
    patch = result['nominal_source'][y0:y1,x0:x1]
    warped = cv2.warpAffine(patch,local_transform,(x1-x0,y1-y0))
    photo = lambda x: np.clip(x.astype(np.float32)*m['source_photo_gain']+m['source_photo_offset'],0,255).astype(np.uint8)
    local = support[y0:y1,x0:x1]
    assert np.array_equal(result['image'][y0:y1,x0:x1][local],np.maximum(patch,photo(warped))[local])
    assert np.array_equal(result['clean_reference'][y0:y1,x0:x1][local],np.maximum(patch,photo(patch))[local])
    if strength=='negative':
        assert not result['effect_mask'].any() and result['boxes_xyxy']==[]
    else:
        assert result['label_mask'].any()
        h,w = image.shape[:2]
        for box,row in zip(result['boxes_xyxy'],engine.yolo_rows(result['boxes_xyxy'],w,h)):
            _,xc,yc,bw,bh = row
            assert np.allclose(box,[(xc-bw/2)*w,(yc-bh/2)*h,(xc+bw/2)*w,(yc+bh/2)*h])


def test_user_manifest_generator_and_group_guard(tmp_path):
    image,_ = grid()
    cv2.imwrite(str(tmp_path/'a.png'),image)
    record = {'id':'a','image':'a.png','group':'artificial_a','split':'train'}
    manifest = tmp_path/'sources.json'
    manifest.write_text(json.dumps({'sources':[record]}),encoding='utf-8')
    summary = run(manifest,tmp_path/'generated',count=5,max_proposals=80,qa_count=1)
    assert summary['completed'] and summary['accepted']==5
    assert summary['accepted_by_strength']=={'strong':3,'medium':1,'negative':1}
    assert not summary['data_yaml_written']
    rows = [json.loads(x) for x in (tmp_path/'generated/metadata.jsonl').read_text().splitlines()]
    assert len(rows)==5
    assert all((tmp_path/'generated'/r['label_path']).is_file() for r in rows)
    # A second split cannot reuse either a source group or byte-identical image.
    manifest.write_text(json.dumps({'sources':[record,dict(record,id='b',group='artificial_b',split='val')]}),encoding='utf-8')
    with pytest.raises(ValueError,match='crosses train/val'):
        load_sources(manifest,'pretrain')


def test_bad_sources_reject_instead_of_becoming_negative():
    engine = importlib.import_module('ffa_stitch.ghost.synthesis_pretrain')
    with pytest.raises(engine.SynthesisRejected):
        engine.prepare_image(np.full((384,384,3),80,np.uint8))
