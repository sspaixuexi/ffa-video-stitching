"""I/O contracts for native-canvas boxes and saved-transform correction."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ffa_stitch.ghost.correct import read_boxes, run as correct
from ffa_stitch.ghost.detect import select_boxes
from ffa_stitch.ghost.io import write_json, write_png


def make_inputs(root):
    size=256
    y,x=np.indices((size,size))
    gray=np.rint(95+24*np.sin(.24*x+.04*y)+18*np.sin(.19*y)+9*np.cos(.11*x-.09*y)).astype(np.uint8)
    shifted=np.zeros_like(gray);shifted[:,6:]=gray[:,:-6]
    baseline=gray.copy();baseline[80:176,80:176]=np.maximum(gray,shifted)[80:176,80:176]
    write_png(root/'piece.png',baseline)
    frames=[];transforms={}
    for k,work in enumerate([gray,shifted]):
        write_png(root/f'work_{k}.png',work)
        write_png(root/f'mask_{k}.png',np.full(work.shape,255,np.uint8))
        frames.append({'index':k,'working_image':f'work_{k}.png','working_mask':f'mask_{k}.png',
                       'working_shape':[size,size],'original_to_work':np.eye(3).tolist()})
        transforms[str(k)]={'H_work_to_canvas':np.eye(3).tolist(),'H_original_to_canvas':np.eye(3).tolist(),
            'projection_route':[{'matrix':np.eye(3).tolist(),'canvas_size_wh':[size,size]}]}
    write_json(root/'stitch_manifest.json',{'frames':frames,'pieces':[{'image':'piece.png','keyframes':[0,1],'transforms':transforms}]})
    write_json(root/'boxes.json',{'coordinate_space':'piece_canvas_pixels','image_size_wh':[size,size],
                                'boxes':[{'xyxy':[80,80,176,176],'score':.9,'class_id':0}]})
    return baseline


@pytest.mark.parametrize('strategy',['SGC','BSC','FCC'])
def test_correction_cli_contract(tmp_path,strategy):
    baseline=make_inputs(tmp_path)
    report=correct(tmp_path/'stitch_manifest.json',tmp_path/'boxes.json',tmp_path/'output',strategy=strategy)
    assert report['strategy']==strategy
    assert report['summary']['accepted_regions']==1
    edit=cv2.imread(str(tmp_path/'output/edit_mask.png'),0)>0
    after=cv2.imread(str(tmp_path/'output/corrected.png'),0)
    assert np.array_equal(after[~edit],baseline[~edit])
    with pytest.raises(ValueError,match='empty or new'):
        correct(tmp_path/'stitch_manifest.json',tmp_path/'boxes.json',tmp_path/'output')


def test_boxes_reject_coordinate_and_canvas_mismatch(tmp_path):
    p=tmp_path/'boxes.json'
    write_json(p,{'coordinate_space':'original_frame','image_size_wh':[100,100],'boxes':[]})
    with pytest.raises(ValueError,match='coordinate_space'):
        read_boxes(p,(100,100,3))
    p.write_text(json.dumps({'coordinate_space':'piece_canvas_pixels','image_size_wh':[99,100],'boxes':[]}))
    with pytest.raises(ValueError,match='image_size_wh'):
        read_boxes(p,(100,100,3))


def test_detector_threshold_has_no_nms_and_keeps_equal_score():
    b=[[1,2,10,20],[1,2,10,20],[5,5,9,9],[1,1,4,4]]
    result=select_boxes(b,[.25,.7,.249,.9],[0,0,0,1],100,100,.25)
    assert len(result)==2
    assert result[0]['score']==.25
    assert result[0]['xyxy']==result[1]['xyxy']


def test_detector_adapter_with_external_api_double(tmp_path,monkeypatch):
    from ffa_stitch.ghost import detect
    class Tensor:
        def __init__(self,value):self.value=np.asarray(value)
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.value
    class Boxes:
        xyxy=Tensor([[1,2,30,40],[1,2,30,40]])
        conf=Tensor([.25,.249])
        cls=Tensor([0,0])
    class Result:
        boxes=Boxes()
    class Model:
        names={0:'ghost'}
        def __init__(self,path):assert Path(path).is_file()
        def predict(self,**kwargs):
            assert kwargs['conf']==0 and kwargs['max_det']==300
            assert kwargs['half'] is False and kwargs['save'] is False
            return [Result()]
    checkpoint=tmp_path/'synthetic_api_double.pt';checkpoint.write_text('NOT A MODEL')
    image=tmp_path/'image.png';write_png(image,np.full((64,96),20,np.uint8))
    monkeypatch.setattr(detect,'load_yolov10_class',lambda repo:Model)
    report=detect.run(image,checkpoint,tmp_path/'result.json')
    assert report['image_size_wh']==[96,64] and len(report['boxes'])==1
    assert report['extra_nms'] is False
    assert read_boxes(tmp_path/'result.json',(64,96,3),report['image_sha256'])==[[1.,2.,30.,40.]]


def test_detection_rejects_unadapted_checkpoint(tmp_path,monkeypatch):
    from ffa_stitch.ghost import detect
    class Model:
        names={0:'person',1:'bicycle'}
        def __init__(self,path):pass
        def predict(self,**kwargs):raise AssertionError('Should reject before inference')
    checkpoint=tmp_path/'unadapted.pt';checkpoint.write_text('API DOUBLE')
    image=tmp_path/'image.png';write_png(image,np.full((64,96),20,np.uint8))
    monkeypatch.setattr(detect,'load_yolov10_class',lambda repo:Model)
    with pytest.raises(ValueError,match='single-class ghost'):
        detect.run(image,checkpoint,tmp_path/'result.json')
    assert not (tmp_path/'result.json').exists()


def test_correction_consumes_stitch_cli_manifest(tmp_path):
    from ffa_stitch.stitch_cli import main as stitch_main
    frames=tmp_path/'frames';frames.mkdir()
    rng=np.random.default_rng(22)
    raw=rng.integers(20,220,(256,256),dtype=np.uint8)
    for index,dx in enumerate([0,3,6]):
        view=cv2.warpAffine(raw,np.array([[1,0,dx],[0,1,0]],float),(256,256))
        write_png(frames/f'frame_{index:03d}.png',view)
    stitched=tmp_path/'stitched'
    assert stitch_main(['--frames',str(frames),'--output',str(stitched),'--matcher','orb',
        '--selection','fixed','--fixed-step','1','--canvas','dynamic','--threads','1'])==0
    manifest=json.loads((stitched/'stitch_manifest.json').read_text())
    assert len(manifest['pieces'][0]['keyframes'])>=2
    baseline=cv2.imread(str(stitched/manifest['pieces'][0]['image']),0)
    write_json(tmp_path/'empty_boxes.json',{'coordinate_space':'piece_canvas_pixels',
        'image_size_wh':[baseline.shape[1],baseline.shape[0]],'boxes':[]})
    report=correct(stitched/'stitch_manifest.json',tmp_path/'empty_boxes.json',tmp_path/'corrected')
    assert report['summary']['accepted_regions']==0
    assert np.array_equal(cv2.imread(str(tmp_path/'corrected/corrected.png'),0),baseline)
