"""Lossless native-resolution erase masks exchanged by the review editor."""
import numpy as np

from ..util import require

SCHEMA = 'paper2-pixel-review-v1'


def encode(mask):
    flat=np.asarray(mask,bool).reshape(-1)
    edges=np.diff(np.r_[False,flat,False].astype(np.int8))
    return [[int(a),int(b-a)] for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]


def decode(runs, shape):
    require(len(shape)==2 and 0 < int(np.prod(shape)) <= 12000000,'invalid mask dimensions')
    flat=np.zeros(int(np.prod(shape)),bool); end=0
    for item in runs:
        require(len(item)==2 and all(type(v) is int for v in item),'mask run must be two integers')
        start,length=item
        require(start>=end and length>0 and start+length<=len(flat),'invalid or overlapping mask run')
        flat[start:start+length]=True;end=start+length
    return flat.reshape(shape)


def on_page(pixel_mask, size):
    l,t,r,b=pixel_mask['box']; w,h=size
    require(all(type(v) is int for v in [l,t,r,b]) and 0<=l<r<=w and 0<=t<b<=h,'mask box outside page')
    result=np.zeros((h,w),bool)
    result[t:b,l:r]=decode(pixel_mask['runs'],(b-t,r-l))
    return result


def region_pixels(region, box):
    l,t,r,b=box
    if not region:
        return np.ones((b-t,r-l),bool)
    points=np.asarray(region,dtype=float)
    require(points.ndim==2 and points.shape[1]==2 and len(points)>=3 and np.isfinite(points).all(),'invalid review polygon')
    yy,xx=np.mgrid[t:b,l:r];x,y=xx+.5,yy+.5
    inside=np.zeros((b-t,r-l),bool)
    for a,z in zip(points,np.roll(points,1,axis=0)):
        if a[1]!=z[1]:
            inside ^= ((a[1]>y)!=(z[1]>y)) & (x<(z[0]-a[0])*(y-a[1])/(z[1]-a[1])+a[0])
    return inside


def has_uncertainty(revision):
    """Unknown pixels are separate annotations, never implicit 'keep' ground truth.

    The current binary training/evaluation readers cannot represent ignored pixels.
    Refuse such a confirmed crop until its uncertainty is resolved or a masked-loss
    consumer is used. Unknown edits parked outside the active polygon do not count.
    """
    layer=revision.get('uncertainMask')
    if not layer:
        return False
    l,t,r,b=revision['box'];a,c,z,d=layer['box']
    unknown=decode(layer['runs'],(d-c,z-a))
    left,top,right,bottom=max(l,a),max(t,c),min(r,z),min(b,d)
    if left>=right or top>=bottom:
        return False
    domain=region_pixels(revision.get('region'),revision['box'])
    return bool((unknown[top-c:bottom-c,left-a:right-a] & domain[top-t:bottom-t,left-l:right-l]).any())


def reviewed_crop(page, revision):
    """Native review crop, using the editor's pixel-centre polygon convention.

    Archived outsideMask is deliberately ignored: it is recoverable editing state,
    not a label. Old rectangular exports without region remain readable.
    """
    grey=np.asarray(page.convert('L'));h,w=grey.shape
    require(not has_uncertainty(revision),'uncertain pixels require an ignore-mask consumer; cannot use this region as complete binary ground truth')
    l,t,r,b=revision['box']
    require(all(type(v) is int for v in [l,t,r,b]) and 0<=l<r<=w and 0<=t<b<=h,'review crop outside scan')
    image=grey[t:b,l:r].copy();mask=decode(revision['runs'],image.shape)
    region=revision.get('region')
    if region:
        points=np.asarray(region,dtype=float)
        require(points.ndim==2 and points.shape[1]==2 and len(points)>=3 and np.isfinite(points).all(),'invalid review polygon')
        require(((points[:,0]>=0)&(points[:,0]<=w)&(points[:,1]>=0)&(points[:,1]<=h)).all(),'review polygon outside scan')
        inside=region_pixels(region,revision['box'])
        image[~inside]=255;mask &= inside
    mask &= image<245
    if revision.get('not_revision'):
        mask[:]=False
    return image,mask.astype(np.uint8)


def geometry(state):
    """Only user-confirmed revisions become labels; untouched predictions stay out."""
    rows=state['revisions']
    require(not state.get('done') or all(r.get('confirmed') for r in rows),'completed page has unreviewed revisions')
    strokes=[];groups=[]
    for r in rows:
        if not r.get('confirmed') or r.get('not_revision'):
            continue
        require(not has_uncertainty(r),'uncertain pixels cannot be exported as complete geometry ground truth')
        key=r['id']
        strokes.append({'id':key,'kind':'line','path':[], 'pixel_mask':{'box':r['box'],'runs':r['runs']},'human_confirmed':True})
        groups.append({'id':key,'strokes':[key],'region':r['region'],'region_auto':False,
                       'text':r.get('text',''),'unreadable':r.get('unreadable',False)})
    return {'done':bool(state.get('done')), 'strokes':strokes,'groups':groups,'review_seconds':state.get('seconds',0)}
