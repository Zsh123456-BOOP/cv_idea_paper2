"""Fixed, label-free pilot: shrink a detector mask and protect unusually dark ink.

No annotation is an input to the heuristic. Human erase masks are used only by
the separate comparison tool. All returned erase masks are subsets of the prior.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

from .tighten import non_white


@dataclass(frozen=True)
class Settings:
    shrink: int = 1
    window: int = 19
    bin_width: int = 8
    minimum_ink: int = 12
    minimum_drop: int = 16
    iqr_factor: float = .5
    hole_margin: int = 1
    anchor_reach: int = 12
    ink_ceiling: int = 200
    minimum_component: int = 3
    global_std_factor: float = .5


def quantiles(grey, pool, settings):
    """Local ink quantiles via sliding histograms; paper is never in the pool."""
    size=settings.window
    total=ndi.uniform_filter(pool.astype(np.float32),size,mode='constant')*size*size
    cumulative=np.zeros_like(total)
    outputs=[np.full(grey.shape,255,np.float32) for _ in range(3)]
    found=[np.zeros(grey.shape,bool) for _ in range(3)]
    for low in range(0,256,settings.bin_width):
        members=pool & (grey>=low) & (grey<low+settings.bin_width)
        cumulative+=ndi.uniform_filter(members.astype(np.float32),size,mode='constant')*size*size
        for q,out,done in zip((.25,.5,.75),outputs,found):
            hit=(~done)&(total>=settings.minimum_ink)&(cumulative>=q*total)
            out[hit]=min(255,low+settings.bin_width/2);done|=hit
    return *outputs,total


def shifted(array, dy, dx):
    """Sample a displaced neighbour; out-of-image pixels are false, never wrapped."""
    out=np.zeros_like(array)
    h,w=array.shape
    y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
    if y1>y0 and x1>x0:out[y0:y1,x0:x1]=array[y0+dy:y1+dy,x0+dx:x1+dx]
    return out


def opposite_anchors(grey, prior, reach):
    """Conservative extra arm: ink paths reach outside the prior on both sides.

    Four orientations, at most one pixel lateral tolerance. This is geometric
    support for a candidate, not proof that the pixels belong to original text.
    """
    ink=grey<200;path=ndi.binary_dilation(ink,np.ones((3,3),bool))
    outside=ink&~prior;result=np.zeros_like(prior)
    for dy,dx in ((1,0),(0,1),(1,1),(1,-1)):
        sides=[]
        for sign in (-1,1):
            alive=path.copy();supported=np.zeros_like(prior)
            for distance in range(1,reach+1):
                y,x=sign*distance*dy,sign*distance*dx
                alive &= shifted(path,y,x)
                if distance>=2:supported |= alive&shifted(outside,y,x)
            sides.append(supported)
        result |= sides[0]&sides[1]
    return result


def crossing_support(ink, prior, domain, reach):
    """Eight straight directions with two-pixel outside anchors on both ends.

    Also require transverse ink inside the prior. This is a crossing proxy,
    not recognition of the original letter; curved or fully enclosed letters
    can lack support. No pixels outside the selected polygon contribute.
    """
    path=ndi.binary_dilation(ink,np.ones((3,3),bool))&domain
    outside=ink&~prior;inside=ink&prior
    result=np.zeros_like(prior)
    for dy,dx in ((1,0),(0,1),(1,1),(1,-1),(2,1),(2,-1),(1,2),(1,-2)):
        norm=np.hypot(dy,dx)
        offsets=list(dict.fromkeys((int(round(d*dy/norm)),int(round(d*dx/norm)))
                                   for d in range(1,reach+1)))
        sides=[]
        for sign in (-1,1):
            alive=path.copy();supported=np.zeros_like(prior);previous=np.zeros_like(prior)
            for y,x in offsets:
                alive &= shifted(path,sign*y,sign*x)
                anchor=shifted(outside,sign*y,sign*x)
                supported |= alive&anchor&previous
                previous=anchor
            sides.append(supported)
        ty,tx=int(round(2*dx/norm)),int(round(-2*dy/norm))
        transverse=shifted(inside,ty,tx)&shifted(inside,-ty,-tx)
        result |= sides[0]&sides[1]&transverse
    return result&ink


def region_holes(grey, prior, domain, baseline, settings):
    """Exclude pale/dust pixels from statistics; keep direct erasure as base.

    Each polygon is one statistical region. No white padding or gold mask
    influences its mean. Joint and structure arms are nested restrictions of
    the global candidate, with the same one-pixel margin for all three arms.
    """
    pool=(grey<settings.ink_ceiling)&domain
    labels,_=ndi.label(pool,np.ones((3,3),bool))
    sizes=np.bincount(labels.ravel());sizes[0]=0
    pool &= sizes[labels]>=settings.minimum_component
    values=grey[pool].astype(float)
    mean=float(values.mean()) if len(values) else None
    std=float(values.std()) if len(values) else None
    threshold=mean-max(settings.minimum_drop,settings.global_std_factor*std) if len(values) else None
    global_candidate=np.zeros_like(prior)
    if len(values)>=settings.minimum_ink:
        global_candidate=baseline&pool&(grey<threshold)
    q25,median,q75,count=quantiles(grey,pool,settings)
    local_threshold=median-np.maximum(settings.minimum_drop,settings.iqr_factor*(q75-q25))
    joint=global_candidate&(count>=settings.minimum_ink)&(grey<local_threshold)
    structure=joint&crossing_support(pool,prior,domain,settings.anchor_reach)
    holes={}
    for name,candidate in (('global',global_candidate),('joint',joint),('structure',structure)):
        expanded=ndi.binary_dilation(candidate,iterations=settings.hole_margin) if settings.hole_margin else candidate
        holes[name]=expanded&baseline&pool
    return holes,{'analysis_ink':int(pool.sum()),'mean_gray':mean,'std_gray':std,
                  'global_gray_threshold':threshold,
                  'protected':{name:int(mask.sum()) for name,mask in holes.items()}}


def process(grey, prior, domain=None, settings=Settings()):
    grey=np.asarray(grey,np.uint8);prior=np.asarray(prior,bool)
    assert grey.shape==prior.shape
    domain=np.ones_like(prior) if domain is None else np.asarray(domain,bool)
    ink=non_white(grey)
    baseline=prior&domain&(grey<245)
    eroded=ndi.binary_erosion(prior,iterations=settings.shrink) if settings.shrink else prior.copy()
    # Do not delete a thin connected detection merely because erosion empties it.
    labels,count=ndi.label(prior,np.ones((3,3),bool))
    surviving=np.zeros(count+1,bool);surviving[np.unique(labels[eroded])]=True
    surviving[0]=True
    shrunk=eroded|(prior&~surviving[labels])
    tight=shrunk&ink&domain
    q25,median,q75,support=quantiles(grey,shrunk&ink,settings)
    threshold=median-np.maximum(settings.minimum_drop,settings.iqr_factor*(q75-q25))
    candidate=tight&(support>=settings.minimum_ink)&(grey<threshold)
    supported=candidate&opposite_anchors(grey,prior,settings.anchor_reach)
    expand=lambda m: (ndi.binary_dilation(m,iterations=settings.hole_margin) if settings.hole_margin else m)&tight
    holes=expand(candidate);linked_holes=expand(supported)
    region,statistics=region_holes(grey,prior,domain,baseline,settings)
    return {**{name:baseline&~mask for name,mask in region.items()},
            **{name+'_holes':mask for name,mask in region.items()},
            'direct':baseline,'tight':tight,'gray':tight&~holes,'linked':tight&~linked_holes,
            'holes':holes,'linked_holes':linked_holes,'candidate':candidate,
            'statistics':{'prior_ink':int(baseline.sum()),'tight_ink':int(tight.sum()),
                          'dark_candidates':int(candidate.sum()),'holes':int(holes.sum()),
                          'linked_holes':int(linked_holes.sum()),
                          'region':statistics,
                          'near_black_fraction':float((grey[baseline]<=10).mean()) if baseline.any() else None},
            'settings':asdict(settings)}
