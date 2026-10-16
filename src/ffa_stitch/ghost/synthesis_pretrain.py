"""Array-only ghost synthesis; OpenCV BGR uint8 input. Algorithmic boxes require visual review."""
from __future__ import annotations
from collections import Counter
import math
import cv2
import numpy as np
from skimage.morphology import skeletonize

VERSION = 'ghosting-synthesis-v3.0-dense-visible-pilot'


MAX_SIDE = 1024


class SynthesisRejected(ValueError):
    def __init__(self, reason, details=None):
        self.reason, self.details = reason, details or {}
        super().__init__(reason)


def tissue_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > 4).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= .01*mask.size:
            cv2.drawContours(filled, [c], -1, 1, -1)
    return filled.astype(bool)


def line_response(gray):
    """Multi-scale bright-ridge Hessian response; blob eigenvalues are suppressed.

    This is a fast heuristic, not a learned or clinically validated segmenter.
    """
    image = gray.astype(np.float32)
    response = np.zeros(gray.shape, np.float32)
    for sigma in (1.0, 1.8, 2.8):
        smooth = cv2.GaussianBlur(image, (0,0), sigma)
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2,0,ksize=3)*sigma*sigma
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0,2,ksize=3)*sigma*sigma
        dxy = cv2.Sobel(smooth, cv2.CV_32F, 1,1,ksize=3)*sigma*sigma
        root = np.sqrt((dxx-dyy)**2+4*dxy*dxy)
        a,b = (dxx+dyy-root)*.5, (dxx+dyy+root)*.5
        small = np.where(np.abs(a)<=np.abs(b), a,b)
        large = np.where(np.abs(a)<=np.abs(b), b,a)
        ratio = np.abs(small)/(np.abs(large)+1e-6)
        ridges = np.exp(-ratio*ratio/(2*.38**2)) * (np.abs(large)/(np.abs(large)+5))
        ridges[large >= 0] = 0
        np.maximum(response, ridges, out=response)
    return response


def _components(mask, min_area=12):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    if n>1:
        keep[1:] = (stats[1:,cv2.CC_STAT_AREA]>=min_area) & (np.maximum(stats[1:,cv2.CC_STAT_WIDTH],stats[1:,cv2.CC_STAT_HEIGHT])>=8)
    return keep[lab]


def line_metrics(mask, minimum_segment=14, is_skeleton=False):
    skeleton = mask.astype(bool) if is_skeleton else skeletonize(mask)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16S, np.ones((3,3), np.uint8))-skeleton.astype(np.int16)
    junction = (neighbors>=3)&skeleton
    # Remove junction centers to count separate long arms; a single straight
    # line or closed lesion ring cannot pass a >=3-arm anatomical gate alone.
    segments = skeleton & ~cv2.dilate(junction.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    n, _, stats, _ = cv2.connectedComponentsWithStats(segments.astype(np.uint8),8)
    lengths = sorted((int(v) for v in stats[1:,cv2.CC_STAT_AREA] if v>=minimum_segment), reverse=True)
    return dict(skeleton_pixels=int(skeleton.sum()), long_segments=len(lengths),
                long_segment_pixels=int(sum(lengths)), segment_lengths=lengths[:12], junction_pixels=int(junction.sum()))


def prepare_image(image, vessel_mask=None, source=None):
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2]!=3:
        raise SynthesisRejected('expected_uint8_BGR')
    h0,w0 = image.shape[:2]
    if min(h0,w0)<128:
        raise SynthesisRejected('image_too_small')
    source = dict(source or {})
    crop = source.get('safe_crop_bbox') or [0,0,w0,h0]
    if len(crop)!=4 or any(int(v)!=v for v in crop):
        raise SynthesisRejected('invalid_safe_crop')
    x1,y1,x2,y2 = map(int,crop)
    if not(0<=x1<x2<=w0 and 0<=y1<y2<=h0):
        raise SynthesisRejected('safe_crop_out_of_bounds')
    if vessel_mask is not None and vessel_mask.shape != (h0,w0):
        raise SynthesisRejected('mask_shape_mismatch')
    cut = image[y1:y2,x1:x2]
    factor = min(1.,MAX_SIDE/max(cut.shape[:2]))
    h,w = int(round(cut.shape[0]*factor)),int(round(cut.shape[1]*factor))
    img = cut.copy() if (h,w)==cut.shape[:2] else cv2.resize(cut,(w,h),interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    tissue = tissue_mask(img)
    if tissue.mean()<.15:
        raise SynthesisRejected('insufficient_FOV')
    response = line_response(g)
    contrast = g.astype(np.float32)-cv2.GaussianBlur(g.astype(np.float32),(0,0),5)
    candidate = (response>=.16)&(contrast>=3)&tissue
    mask_kind = 'multiscale_Hessian_bright_line_heuristic'
    if vessel_mask is not None:
        supplied = cv2.resize((vessel_mask[y1:y2,x1:x2]>0).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        candidate &= cv2.dilate(supplied.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        mask_kind += '_constrained_by_public_vessel_mask'
    candidate = _components(candidate)
    skeleton = skeletonize(candidate)
    if skeleton.sum()<120:
        raise SynthesisRejected('insufficient_linear_vessels',{'skeleton_pixels':int(skeleton.sum())})
    if candidate.mean()>.30:
        raise SynthesisRejected('overdense_unreliable_candidate')
    interior = cv2.distanceTransform(np.pad(tissue.astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    inside = cv2.distanceTransform(candidate.astype(np.uint8),cv2.DIST_L2,5)
    width = float(np.clip(2*np.percentile(inside[candidate],70),2,10))
    min_radius = max(38,min(66,min(h,w)//8))
    kernel = 2*min_radius+1
    density = cv2.boxFilter(candidate.astype(np.float32),-1,(kernel,kernel),normalize=True)
    skeleton_density = cv2.boxFilter(skeleton.astype(np.float32),-1,(kernel,kernel),normalize=True)
    eligible = skeleton & (interior>=min_radius+32) & (density>=.017)
    points = np.argwhere(eligible)
    if not len(points):
        raise SynthesisRejected('no_dense_interior_sampling_centers')
    values = density[eligible]*np.sqrt(np.maximum(skeleton_density[eligible],1e-6))
    threshold = np.percentile(values,40)
    take = values>=threshold
    points, values = points[take], values[take]
    weights = np.maximum(values,1e-8)**1.5
    weights /= weights.sum()
    sx,sy=w/(x2-x1),h/(y2-y1)
    forward=np.array([[sx,0,-sx*x1],[0,sy,-sy*y1],[0,0,1]],float)
    mapping=dict(original_shape_hw=[h0,w0], safe_crop_bbox=[x1,y1,x2,y2], output_shape_hw=[h,w],
                 input_to_output_matrix=forward.tolist(),output_to_input_matrix=np.linalg.inv(forward).tolist(),
                 coordinate_convention='half_open_pixel_boundaries', resize='AREA_image_NEAREST_mask')
    return dict(image=img,gray=g,tissue=tissue,response=response,candidate=candidate,skeleton=skeleton,
                density=density,interior=interior,points=points,weights=weights,width=width,min_radius=min_radius,
                source=source,mapping=mapping,mask_kind=mask_kind,
                preprocessing=dict(version=VERSION,scales=[1.,1.8,2.8],candidate_density=float(candidate.mean()),
                    skeleton_pixels=int(skeleton.sum()),eligible_centers=len(points),estimated_vessel_width=width))


def _xyxy(mask,pad=3):
    yy,xx=np.where(mask)
    if not len(xx):raise SynthesisRejected('empty_label')
    h,w=mask.shape
    return [max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(w,int(xx.max())+pad+1),min(h,int(yy.max())+pad+1)]


def yolo_rows(boxes,w,h):
    rows=[]
    for a,b,c,d in boxes:
        if not(0<=a<c<=w and 0<=b<d<=h):raise SynthesisRejected('invalid_box')
        rows.append([0,(a+c)/(2*w),(b+d)/(2*h),(c-a)/w,(d-b)/h])
    return rows


def _generate(prepared,rng,strength,mode):
    p=prepared
    image=p['image']; h,w=image.shape[:2]
    negative=strength=='negative'
    cy,cx=p['points'][int(rng.choice(len(p['points']),p=p['weights']))]
    lo=p['min_radius']; hi=max(lo+1,min(132,min(h,w)//4))
    radius=int(rng.integers(lo,hi+1))
    if strength=='strong':
        displacement=float(np.clip(rng.uniform(2.,4.)*p['width'],rng.uniform(7.,10.),24.))
    elif strength=='medium':
        displacement=float(np.clip(rng.uniform(1.2,2.5)*p['width'],4.,18.))
    else: displacement=0.
    # Shared border constraints across positives/negatives; do not encode class
    # merely through which field positions can be selected.
    margin=int(math.ceil(24.+radius*(math.sin(math.radians(3))+.015)))+5
    if p['interior'][cy,cx] < radius+margin:
        raise SynthesisRejected('full_ROI_and_context_not_in_FOV')
    if min(cx-radius-margin,cy-radius-margin,w-1-cx-radius-margin,h-1-cy-radius-margin)<0:
        raise SynthesisRejected('full_ROI_and_context_not_in_canvas')
    direction=float(rng.uniform(-math.pi,math.pi))
    tx,ty=displacement*math.cos(direction),displacement*math.sin(direction)
    angle=0. if negative else float(rng.uniform(-3,3))
    scale=1. if negative else float(rng.uniform(.99,1.01))
    transform=cv2.getRotationMatrix2D((float(cx),float(cy)),angle,scale)
    transform[:,2]+=[tx,ty]
    # The ROI crop contains sufficient context for all interpolation; only this
    # small patch is warped. Matrices in metadata remain full-image transforms.
    qx0,qy0=int(cx-radius-margin),int(cy-radius-margin)
    qx1,qy1=int(cx+radius+margin+1),int(cy+radius+margin+1)
    patch=image[qy0:qy1,qx0:qx1]
    ph,pw=patch.shape[:2]
    t3=np.vstack([transform,[0,0,1.]])
    shift=np.array([[1.,0.,-qx0],[0.,1.,-qy0],[0.,0.,1.]])
    local=(shift@t3@np.linalg.inv(shift))[:2]
    yy,xx=np.indices((ph,pw))
    dx,dy=xx-(cx-qx0),yy-(cy-qy0)
    support=dx*dx+dy*dy<=radius*radius
    theta=float(rng.uniform(-math.pi/2,math.pi/2))
    if mode=='overlapping_views':
        signed=dx*math.cos(theta)+dy*math.sin(theta)
        support &= np.abs(signed)<=.78*radius
    valid_patch=p['tissue'][qy0:qy1,qx0:qx1]
    valid_warp=cv2.warpAffine(valid_patch.astype(np.uint8),local,(pw,ph),flags=cv2.INTER_NEAREST).astype(bool)
    if not np.all(valid_warp[support]):raise SynthesisRejected('warped_FOV_does_not_cover_full_support')
    original_mask=p['candidate'][qy0:qy1,qx0:qx1]
    original_skeleton=p['skeleton'][qy0:qy1,qx0:qx1]
    anatomical=line_metrics(original_skeleton&support,minimum_segment=16,is_skeleton=True)
    density=float((original_mask&support).sum()/support.sum())
    if density<.02 or density>.27 or anatomical['long_segments']<3 or anatomical['skeleton_pixels']<95:
        raise SynthesisRejected('not_dense_multisegment_vasculature',dict(density=density,**anatomical))
    use_photo=bool(rng.random()<.5)
    photo_gain=float(rng.uniform(1.03,1.12)) if use_photo else 1.
    photo_offset=float(rng.uniform(0,3)) if use_photo else 0.
    warped=cv2.warpAffine(patch,local,(pw,ph),flags=cv2.INTER_LINEAR)
    warped_mask=cv2.warpAffine(original_mask.astype(np.uint8),local,(pw,ph),flags=cv2.INTER_NEAREST).astype(bool)
    nominal_photo=np.clip(patch.astype(np.float32)*photo_gain+photo_offset,0,255).astype(np.uint8)
    warped_photo=np.clip(warped.astype(np.float32)*photo_gain+photo_offset,0,255).astype(np.uint8)
    clean=patch.copy(); augmented=patch.copy()
    clean[support]=np.maximum(patch[support],nominal_photo[support])
    augmented[support]=np.maximum(patch[support],warped_photo[support])
    effect=np.any(augmented!=clean,axis=2)
    labels=np.zeros((ph,pw),bool)
    visible=np.zeros((ph,pw),bool)
    boxes=[];visibility={}
    if negative:
        if effect.any():raise SynthesisRejected('identity_negative_changed_residual')
        visibility=dict(visible_pixels=0,long_segments=0,long_segment_pixels=0,geometric_residual_identity=True)
    else:
        gray_aug=cv2.cvtColor(augmented,cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_clean=cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_warp=cv2.cvtColor(warped_photo,cv2.COLOR_BGR2GRAY).astype(np.float32)
        local_contrast=gray_warp-cv2.GaussianBlur(gray_warp,(0,0),4)
        separation=cv2.distanceTransform((~original_mask).astype(np.uint8),cv2.DIST_L2,5)
        min_contrast=12 if strength=='strong' else 8
        inner=cv2.erode(support.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
        visible=warped_mask&inner&(separation>=2)&(gray_aug-gray_clean>=min_contrast)&(local_contrast>=min_contrast)
        visible=_components(visible,min_area=10)
        vmetrics=line_metrics(visible,minimum_segment=14 if strength=='strong' else 12)
        required_length=50 if strength=='strong' else 32
        if vmetrics['long_segments']<2 or vmetrics['long_segment_pixels']<required_length:
            raise SynthesisRejected('insufficient_visible_long_repeated_vessels',dict(strength=strength,**vmetrics))
        distance_back=cv2.distanceTransform((~warped_mask).astype(np.uint8),cv2.DIST_L2,5)
        original_only=original_mask&inner&(distance_back>=2)
        ometrics=line_metrics(original_only,minimum_segment=12)
        if ometrics['long_segment_pixels']<30:
            raise SynthesisRejected('original_contour_not_separately_visible')
        labels=(original_mask|warped_mask)&support
        a,b,c,d=_xyxy(labels)
        boxes=[[a+qx0,b+qy0,c+qx0,d+qy0]]
        if a+qx0<4 or b+qy0<4 or c+qx0>w-4 or d+qy0>h-4:
            raise SynthesisRejected('candidate_box_touches_canvas')
        visibility=dict(visible_pixels=int(visible.sum()),min_contrast=min_contrast,separation_threshold_pixels=2,
                        median_visible_separation=float(np.median(separation[visible])),
                        median_visible_increment=float(np.median((gray_aug-gray_clean)[visible])),
                        original_separate_length=ometrics['long_segment_pixels'],**vmetrics)
    output=image.copy();reference=image.copy()
    output[qy0:qy1,qx0:qx1]=augmented
    reference[qy0:qy1,qx0:qx1]=clean
    def full(mask):
        a=np.zeros((h,w),bool);a[qy0:qy1,qx0:qx1]=mask;return a
    inverse=np.array(p['mapping']['output_to_input_matrix'])
    input_boxes=[]
    for a,b,c,d in boxes:
        q1=inverse@np.array([a,b,1.]);q2=inverse@np.array([c,d,1.])
        input_boxes.append([float(q1[0]),float(q1[1]),float(q2[0]),float(q2[1])])
    metadata=dict(version=VERSION,strength=strength,negative=negative,mode=mode,
        label_origin='algorithmic_synthetic_pending_visual_QA',clinical_gold_standard=False,
        residual_transform_2x3=transform.tolist(),rotation_degrees=angle,scale_factor=scale,
        translation_xy=[tx,ty],translation_magnitude_pixels=displacement,
        residual_relative_to_vessel_width=displacement/p['width'],estimated_vessel_width=p['width'],
        source_photo_gain=photo_gain,source_photo_offset=photo_offset,
        negative_type=('identity_photometric_boundary' if use_photo else 'identity_vascular_context') if negative else None,
        roi_center_xy=[int(cx),int(cy)],roi_radius=radius,roi_context_margin=margin,roi_fov_distance=float(p['interior'][cy,cx]),
        density=dict(vessel_fraction=density,**anatomical),visibility=visibility,
        quality=dict(complete_internal_support=True,changed_pixels=int(effect.sum()),label_pixels=int(labels.sum()),
                     visual_QA_required=True,lesion_texture_false_selection_still_possible=True),
        coordinate_mapping=p['mapping'],boxes_input_xyxy=input_boxes,
        candidate_mask_method=p['mask_kind'],preprocessing=p['preprocessing'],
        clean_reference='same_photo_and_support_with_identity_residual',
        effect_mask='difference_from_matched_identity_reference_not_all_photometric_edits',
        mode_limitation='controlled_local_support; nonoverlap_nominal_pixels_frozen',
        fusion='max_of_valid_nominal_A_and_residual_B_inside_support')
    return dict(image=output,clean_reference=reference,nominal_source=image,boxes_xyxy=boxes,
                label_mask=full(labels),support_mask=full(support),valid_roi=full(support),effect_mask=full(effect),
                visible_change_candidate=full(visible),candidate_vessel_mask=p['candidate'],metadata=metadata)


def synthesize(prepared,seed,strength='strong',mode='local_residual',max_attempts=32):
    if strength not in ['strong','medium','negative']:raise SynthesisRejected('invalid_strength')
    if mode not in ['local_residual','overlapping_views']:raise SynthesisRejected('invalid_mode')
    if isinstance(prepared,np.ndarray):prepared=prepare_image(prepared)
    rng=np.random.default_rng(seed);failures=Counter()
    for attempt in range(max_attempts):
        try:
            result=_generate(prepared,rng,strength,mode)
            result['metadata'].update(seed=int(seed),attempts_used=attempt+1,rejected_attempt_counts=dict(failures))
            return result
        except SynthesisRejected as e:failures[e.reason]+=1
    raise SynthesisRejected('proposal_attempt_budget_exhausted',dict(strength=strength,counts=dict(failures)))


def quotas(count):
    negative=int(round(count*.2));medium=int(round(count*.2));strong=count-negative-medium
    return dict(strong=strong,medium=medium,negative=negative)
