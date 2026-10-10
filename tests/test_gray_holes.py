import numpy as np

from revrender.gray_holes import Settings, process, shifted
from revrender.tools.gray_holes_view import confusion, metrics


def test_dark_crossing_is_protected_without_using_a_human_mask():
    grey=np.full((45,65),255,np.uint8);prior=np.zeros_like(grey,bool)
    grey[20:25,5:60]=95;prior[18:27,4:61]=True
    grey[5:40,30:33]=25
    result=process(grey,prior)
    assert result['candidate'][22,31] and result['holes'][22,31]
    assert result['linked_holes'][22,31]
    assert not result['gray'][22,31] and result['gray'][22,15]
    assert all(not (result[a]&~result['direct']).any() for a in ('tight','gray','linked'))
    assert (grey[5:40,30:33]==25).all()  # no in-place image edits


def test_saturated_uniform_stroke_does_not_invent_dark_crossings():
    grey=np.full((20,40),255,np.uint8);prior=np.zeros_like(grey,bool)
    grey[8:12,4:36]=0;prior[6:14,3:37]=True
    r=process(grey,prior)
    assert r['direct'].any() and not r['holes'].any()


def test_thin_detection_survives_shrink_and_domain_limits_every_erasure():
    grey=np.full((20,40),255,np.uint8);prior=np.zeros_like(grey,bool)
    grey[10,4:36]=60;prior[10,4:36]=True
    domain=np.ones_like(prior);domain[:,20:]=False
    r=process(grey,prior,domain)
    assert r['tight'][10,4:20].all()
    assert all(not r[a][:,20:].any() for a in ('direct','tight','gray','linked'))
    empty=process(grey,np.zeros_like(prior));assert not empty['gray'].any()


def test_shifts_never_wrap_and_counts_separate_saved_ink_from_missed_strike():
    a=np.zeros((3,4),bool);a[0,0]=True
    assert not shifted(a,1,1).any()
    assert shifted(a,-1,-1)[1,1]
    c=confusion(np.array([[1,1,0]],bool),np.array([[1,0,1]],bool),np.ones((1,3),bool))
    assert c==dict(tp=1,fp=1,fn=1,valid=1)
    assert metrics(c)['f1']==.5 and metrics(c)['valid_damage']==1


def test_viewer_navigation_filters_and_human_reference_are_independent():
    import json
    from pathlib import Path
    import shutil
    import subprocess
    import pytest
    from revrender.tools.gray_holes_view import TEMPLATE, ARMS, HOLE_BASES
    node=shutil.which('node')
    if not node:pytest.skip('needs node')
    score=metrics(dict(tp=1,fp=1,fn=1,valid=2))
    card={'n':1,'id':'reviewed','box':[10,20,70,60],'confirmed':True,'statistics':dict(prior_ink=2,tight_ink=2,holes=1,linked_holes=1,region=dict(analysis_ink=12,mean_gray=90)),
          'protected':{arm:1 for arm in HOLE_BASES},'metrics':{arm:score for arm in ARMS},'images':{key:key for key in [*ARMS,*[a+'_overlay' for a in HOLE_BASES],'original','prior','human']}}
    second={**card,'n':2,'id':'framed','box':[0,0,1200,800],'confirmed':False,'metrics':{},'statistics':{**card['statistics'],'holes':0}}
    data={'cards':[card,second],'arms':ARMS,'hole_bases':HOLE_BASES,'summary':dict(count=2,confirmed=1,framed_only=1,zero_detection=0,arms={a:score for a in ARMS})}
    script=TEMPLATE.read_text().rsplit('<script>',1)[1].split('</script>')[0]
    harness=r'''
    const vm=require('vm'),nodes=new Map(),handlers={};
    function element(id){if(nodes.has(id))return nodes.get(id);const n={value:'1',style:{},options:[],scrollLeft:0,scrollTop:0,clientWidth:600,
      append(){},add(x){this.options.push(x);},replaceChildren(){this.options=[];},removeAttribute(k){delete this[k];},click(){this.onclick?.();}};nodes.set(id,n);return n;}
    element('data').textContent=JSON.stringify(DATA);element('filter').value='all';
    const views=['view0','view1','view2','view3'].map(element);
    const context={window:{innerHeight:900,addEventListener:(k,f)=>handlers['window:'+k]=f},document:{getElementById:element,createElement:()=>element(Symbol()),querySelectorAll:q=>q.endsWith('img')?['original','beforeImage','afterImage','human'].map(element):views,addEventListener:(k,f)=>handlers[k]=f},Option:class{constructor(text,value){this.text=text;this.value=value;}}};
    vm.createContext(context);vm.runInContext(SCRIPT,context);
    const initial=element('humanPanel').hidden===false&&element('afterImage').src==='structure';
    const small=element('original').style.width==='120px'&&element('zoomLabel').textContent==='200%';
    element('zoom').value='.5';element('zoom').oninput();
    const shrink=element('original').style.width==='30px';
    element('showPrior').checked=true;element('showPrior').onchange({type:'change'});
    const preserve=element('original').style.width==='30px';
    element('native').click();const native=element('original').style.width==='60px';
    element('zoom').value='8';element('zoom').oninput();
    views[0].scrollLeft=55;views[0].scrollTop=37;views[0].onscroll();
    const sync=views.every(v=>v.scrollLeft===55&&v.scrollTop===37);
    element('next').click();
    const large=parseFloat(element('original').style.width)<=600&&parseFloat(element('original').style.width)*800/1200<324;
    const reset=views.every(v=>v.scrollLeft===0&&v.scrollTop===0);
    views.forEach(v=>v.clientWidth=180);handlers['window:resize']();
    const narrow=parseFloat(element('original').style.width)<=180;
    const equal=['original','beforeImage','afterImage','human'].every(id=>element(id).style.width===element('original').style.width);
    element('zoom').value='8';element('zoom').oninput();element('fit').click();
    const fit=parseFloat(element('original').style.width)<=180;
    const unconfirmed=element('humanPanel').hidden&&element('rowMetrics').textContent.includes('不报告准确率');
    element('filter').value='confirmed';element('filter').onchange();const reviewed=element('pick').options.length===1&&!element('humanPanel').hidden;
    element('showHoles').checked=true;element('showHoles').onchange();const overlay=element('afterImage').src==='structure_overlay';
    element('after').value='linked';element('after').onchange();const linked=element('afterImage').src==='linked_overlay';
    element('filter').value='framed';element('filter').onchange();const framed=element('pick').options.length===1&&element('humanPanel').hidden;
    console.log(JSON.stringify({initial,unconfirmed,reviewed,overlay,linked,framed,small,shrink,preserve,native,sync,large,reset,narrow,equal,fit}));
    '''.replace('DATA',json.dumps(data)).replace('SCRIPT',json.dumps(script))
    got=json.loads(subprocess.check_output([node,'-e',harness],text=True))
    assert got==dict(initial=True,unconfirmed=True,reviewed=True,overlay=True,linked=True,framed=True,small=True,shrink=True,preserve=True,native=True,sync=True,large=True,reset=True,narrow=True,equal=True,fit=True)


def test_region_statistics_ignore_white_pale_and_isolated_noise():
    grey=np.full((45,65),255,np.uint8);prior=np.zeros_like(grey,bool)
    grey[20:25,5:60]=110;prior[18:27,4:61]=True
    grey[5:40,30:33]=25
    r=process(grey,prior)
    noisy=grey.copy();noisy[:3]=220;noisy[40,60]=30
    other=process(noisy,prior)
    assert r['statistics']['region']==other['statistics']['region']
    assert r['global_holes'][22,31] and r['structure_holes'][22,31]
    for arm in ('global','joint','structure'):
        np.testing.assert_array_equal(r[arm],other[arm])
        assert not (r[arm]&~r['direct']).any()
        assert r[arm][22,15]  # ordinary strike remains erased
    assert not (r['structure_holes']&~r['joint_holes']).any()
    assert not (r['joint_holes']&~r['global_holes']).any()


def test_crossing_requires_two_sided_ink_not_just_a_dark_blob():
    from revrender.gray_holes import crossing_support
    ink=np.zeros((45,65),bool);prior=np.zeros_like(ink);domain=np.ones_like(ink)
    ink[20:25,5:60]=True;ink[5:40,30:33]=True;prior[18:27,4:61]=True
    assert crossing_support(ink,prior,domain,12)[22,31]
    ink[27:,30:33]=False
    assert not crossing_support(ink,prior,domain,12)[22,31]
    domain[:18]=False
    assert not crossing_support(ink&domain,prior,domain,12)[22,31]


def test_uniform_black_and_empty_regions_do_not_create_new_holes():
    for value in (0,255):
        grey=np.full((20,30),value,np.uint8);prior=np.ones_like(grey,bool)
        r=process(grey,prior)
        for arm in ('global','joint','structure'):
            assert not r[arm+'_holes'].any()
            np.testing.assert_array_equal(r[arm],r['direct'])
    grey=np.full((25,35),110,np.uint8);grey[:,20:]=0
    domain=np.zeros_like(grey,bool);domain[:,:20]=True
    r=process(grey,np.ones_like(domain),domain)
    assert r['statistics']['region']['mean_gray']==110
    assert not r['global_holes'].any()
    assert not r['structure'][:,20:].any()
