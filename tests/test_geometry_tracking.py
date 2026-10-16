import numpy as np
import cv2
from ffa_stitch.core import (mutual_indices,estimate,transform_points,NotebookPartition,
                             Config,prepare_frames,select_keyframes,stitch,blend)


def literal_mnn(a,b,tol):
    if not len(a) or not len(b): return np.array([],int),np.array([],int)
    d=np.linalg.norm(a[:,None]-b[None],axis=2)
    ab=d.argmin(1);ba=d.argmin(0);idx=np.arange(len(a))
    ok=(ba[ab]==idx)&np.all(np.abs(a-b[ab])<=tol,axis=1)
    return idx[ok],ab[ok]


def test_chunked_mutual_nearest_is_literal_equivalent():
    rng=np.random.default_rng(45)
    for na,nb in [(0,3),(7,0),(21,17),(300,210),(512,555)]:
        a=rng.integers(0,20,(na,2)).astype(np.float32)
        b=rng.integers(0,20,(nb,2)).astype(np.float32)
        expected=literal_mnn(a,b,10)
        got=mutual_indices(a,b,10,block=13)
        for x,y in zip(expected,got): np.testing.assert_array_equal(x,y)
    # L-infinity acceptance is intentionally preserved; Euclidean distance >10.
    assert len(mutual_indices([[0,0]],[[9,9]],10)[0])==1


class FakeEngine:
    def __init__(self,matches): self.matches=matches;self.calls=0
    def match(self,a,b,ka=None,kb=None):
        self.calls+=1
        if ka is not None and (ka,kb) in self.matches: return self.matches[(ka,kb)]
        return np.zeros((0,2),np.float32),np.zeros((0,2),np.float32),np.zeros(0)
    def counters(self): return {'calls':self.calls}


def test_first_three_denominator_and_surviving_target_endpoint():
    # First pair has 100 matches, but only 20 continue into the third frame.
    # A denominator of 100 would trigger at i=1; original uses 20 and does not.
    xy=np.column_stack([np.arange(100)*30,np.zeros(100)]).astype(np.float32)
    frames=[dict(image=np.zeros((32,32),np.uint8),key=str(i)) for i in range(5)]
    counts=[100,20,18,3]
    matches={(str(i),str(i+1)):(xy[:n],xy[:n],np.ones(n)) for i,n in enumerate(counts)}
    kfs,targets,log=select_keyframes(frames,FakeEngine(matches),Config())
    assert kfs==[0,3]
    assert log[0]['initial']==20 and not log[0]['trigger']
    np.testing.assert_array_equal(targets[(0,3)],xy[:18])
    assert 4 not in kfs  # The default does not force the final frame.


def test_zero_matches_is_explicit_and_does_not_divide_by_zero():
    frames=[dict(image=np.zeros((32,32),np.uint8),key=str(i)) for i in range(4)]
    assert select_keyframes(frames,FakeEngine({}),Config())[0]==[0,1,2]


def test_motion_restarts_when_input_dimensions_change():
    rng=np.random.default_rng(3)
    images=[rng.integers(0,255,(128,160),dtype=np.uint8),rng.integers(0,255,(160,160),dtype=np.uint8)]
    frames=prepare_frames(images)
    kfs,_,_=select_keyframes(frames,FakeEngine({}),Config(selection='motion'))
    assert kfs==[0,1]


def test_similarity_direction_and_accumulation():
    rng=np.random.default_rng(5);b=rng.uniform(0,600,(100,2)).astype(np.float32)
    expected=np.array([[.99,-.04,31],[.04,.99,-12],[0,0,1.]])
    a=transform_points(expected,b).astype(np.float32)
    h,n=estimate(a,b)
    assert n==100
    np.testing.assert_allclose(transform_points(h,b),a,atol=1e-4)
    p=np.array([[10.,20.]])
    g=np.array([[1,0,5],[0,1,-3],[0,0,1.]])
    np.testing.assert_allclose(transform_points(h@g,p),transform_points(h,transform_points(g,p)))


def test_partition_reentry_and_reset():
    checker=NotebookPartition()
    assert checker.add((0,0,100,100))
    assert checker.add((50,0,150,100))
    assert not checker.add((10,0,110,100))
    assert checker.last is None
    # Caller repairs the missing reseed, then the next ROI can be evaluated.
    assert checker.add((10,0,110,100))
    assert checker.last is not None


def test_failures_preserve_disconnected_frames():
    frames=prepare_frames([np.full((64,64),50+i*20,np.uint8) for i in range(3)])
    result=stitch(frames,FakeEngine({}),Config(selection='fixed',fixed_step=1))
    assert result['keyframes']==[0,1,2]
    assert len(result['pieces'])==3
    assert all(edge['registration_failed'] for edge in result['edges'])
    assert all(not m['success'] for m in result['merges'])


def test_partition_boundary_does_not_reseed_for_an_unused_failed_transform():
    frames=prepare_frames([np.full((64,64),50+i*20,np.uint8) for i in range(5)])
    rois=[(0,0,100,100),(50,0,150,100),(10,0,110,100),(60,0,160,100),(15,0,115,100)]
    for f,roi in zip(frames,rois):f['roi']=roi
    xy=np.array([[10,10],[30,10],[30,30],[10,30]],np.float32)
    matches={(frames[i]['key'],frames[i+1]['key']):(xy,xy,np.ones(4)) for i in [0,2,3]}
    result=stitch(frames,FakeEngine(matches),Config(selection='fixed',fixed_step=1))
    assert result['edges'][1]['partition_cut'] and result['edges'][1]['registration_failed']
    assert result['local_groups']==[[0,1],[2,3,4]]


def test_all_blenders_preserve_nonoverlap_and_dtype():
    a=np.full((128,160),60,np.uint8);b=np.full_like(a,120)
    ma=np.zeros_like(a);ma[:,:100]=255
    mb=np.zeros_like(a);mb[:,60:]=255
    a[ma==0]=0;b[mb==0]=0
    for method in ['max','masked_max','feather','multiband','graphcut']:
        result,mask=blend(a,ma,b,mb,method)
        assert result.dtype==np.uint8 and result.shape==a.shape
        assert np.all(result[:,:60]==60) and np.all(result[:,100:]==120)
        assert np.all(mask==255)
