"""Regression checks for human edits surviving region changes and legacy imports."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

NODE=shutil.which('node')
RULES=Path(__file__).parents[1]/'revrender/tools/annotate/pixel_rules.js'
pytestmark=pytest.mark.skipif(not NODE,reason='needs node')


def js(script):
    return json.loads(subprocess.check_output([NODE,'-e',f'const R=require({json.dumps(str(RULES))});console.log(JSON.stringify((()=>{{{script}}})()));'],text=True))


def test_expansion_does_not_scale_or_shift_a_human_mask_and_shrink_is_reversible():
    result=js('''
      const old={box:[10,10,20,20],region:[[10,10],[20,10],[20,20],[10,20]],runs:[[23,1],[74,2]],confirmed:true,note:'human'};
      const expanded=R.reshape(old,[[6,6],[30,6],[30,30],[6,30]],[100,100]);
      const native=R.decode(expanded.runs,(expanded.box[2]-expanded.box[0])*(expanded.box[3]-expanded.box[1]));
      const positions=[];native.forEach((v,i)=>{if(v)positions.push([i%(expanded.box[2]-expanded.box[0])+expanded.box[0],Math.floor(i/(expanded.box[2]-expanded.box[0]))+expanded.box[1]]);});
      const back=R.remap(expanded.runs,expanded.box,old.box);
      return {positions,back,old,confirmed:expanded.confirmed,note:expanded.note};
    ''')
    assert result['positions']==[[13,12],[14,17],[15,17]]
    assert result['back']==[[23,1],[74,2]] and result['old']['confirmed']
    assert result['confirmed'] is False and result['note']=='human'


def test_cleaning_removes_only_nonink_without_moving_valid_pixels():
    result=js('''
      const rgba=new Uint8Array([10,10,10,255,255,255,255,255,244,244,244,255,245,245,245,255]);
      return R.filterInk([[0,4]],rgba);
    ''')
    assert result=={'runs':[[0,1],[2,1]],'removed':2}


def test_legacy_import_and_edited_polygon_keep_box_masks_notes_and_other_pages():
    result=js('''
      const base={id:'p#r',page:0,box:[10,10,20,20],region:[[10,10],[20,10],[20,20],[10,20]],runs:[]};
      const pack={id:'x',pages:[{id:'p',size:[100,100]},{id:'q',size:[100,100]}],rows:[base]};
      const old={rows:{},extra:[],done:{1:true},pageSeconds:{1:20}};
      const edited={...base,box:[5,5,35,25],region:[[5,5],[35,5],[35,25],[20,23],[5,25]],runs:[[240,3]],confirmed:true,note:'keep me',seconds:12,ops:4};
      const value={schema:'paper2-pixel-review-v1',pack:'x',pages:{p:{done:true,revisions:[edited]}}};
      const got=R.importData(pack,old,value);
      const legacy=R.restoreRow(base,{runs:[[15,2]],confirmed:true,note:'legacy'});
      let rejected=false;try{R.importData(pack,old,{...value,pages:{p:{revisions:[{...edited,box:[-1,5,35,25]}]}}});}catch(e){rejected=true;}
      return {got,old,legacy,rejected};
    ''')
    row=result['got']['rows']['p#r']
    assert row['box']==[5,5,35,25] and len(row['region'])==5 and row['runs']==[[240,3]]
    assert row['note']=='keep me' and row['seconds']==12 and row['confirmed']
    assert result['got']['done']['1'] and result['got']['pageSeconds']['1']==20
    assert result['old']['rows']=={} and result['rejected']
    assert result['legacy']['box']==[10,10,20,20] and result['legacy']['runs']==[[15,2]]


def test_whole_question_pointer_navigation_and_polygon_handles():
    # Exercise the actual controller with a no-op drawing context; no browser or user storage.
    overview=RULES.with_name('pixel_overview.js')
    script='''
      const fs=require('fs'),vm=require('vm');
      const G=require(RULES.replace('pixel_rules.js','page_rules.js'));
      const cx=new Proxy({}, {get:(target,key)=>target[key]||(()=>{}),set:(target,key,value)=>(target[key]=value,true)});
      const canvas={clientWidth:800,clientHeight:500,parentElement:{},addEventListener(){},getContext:()=>cx,focus(){},setPointerCapture(){},getBoundingClientRect:()=>({left:0,top:0})};
      let region={id:'r',region:[[10,10],[50,10],[50,30],[10,30]]},opened=[],created=[],changes=[];
      const context={...G,ResizeObserver:class{observe(){}},Math};vm.createContext(context);
      vm.runInContext(fs.readFileSync(OVERVIEW,'utf8')+';this.Overview=PixelOverview;',context);
      const view=new context.Overview(canvas,{rows:()=>[region],activity(){},select(){},zoom(){},open:id=>opened.push(id),create:p=>created.push(p),reshape:(id,p)=>{region={id,region:p};changes.push(p);}});
      view.setImage({width:100,height:80});
      const event=(x,y,more={})=>({clientX:x*view.scale+view.x,clientY:y*view.scale+view.y,button:0,pointerId:1,preventDefault(){},...more});
      view.down(event(30,20));view.up();
      view.editing=true;view.selected='r';view.down(event(50,30));view.move(event(60,40));view.up();
      view.down(event(30,10));view.move(event(30,5));view.up();
      const countAfterAdd=region.region.length;
      view.down(event(30,5,{altKey:true}));view.up();
      view.editing=false;view.down(event(70,50));view.move(event(90,70));view.up();
      view.down(event(20,60,{button:2}));view.move(event(25,65));view.up();
      return {opened,created,changes,countAfterAdd,region};
    '''.replace('RULES',json.dumps(str(RULES))).replace('OVERVIEW',json.dumps(str(overview)))
    result=js(script)
    assert result['opened']==['r']
    assert result['changes'][0][2]==[60,40] and result['countAfterAdd']==5
    assert len(result['region']['region'])==4
    assert result['created']==[[[70,50],[90,50],[90,70],[70,70]]]


def test_deleted_rows_roundtrip_separately_from_training_labels():
    got=js('''
      const base={id:'p#r',page:0,box:[0,0,10,10],region:[[0,0],[10,0],[10,10]],runs:[[2,3]]};
      const pack={id:'x',pages:[{id:'p',size:[10,10]}],rows:[base]};
      const state={rows:{'p#r':R.restoreRow(base,{confirmed:true,deleted:true,note:'keep',seconds:10})},extra:[],done:{0:true},pageSeconds:{}};
      const pages=R.exportPages(pack,state);
      const imported=R.importData(pack,{rows:{},extra:[],done:{},pageSeconds:{}},{schema:'paper2-pixel-review-v1',pack:'x',pages});
      return {pages,imported};
    ''')
    assert got['pages']['p']['revisions']==[]
    deleted=got['pages']['p']['deleted_revisions'][0]
    assert deleted['runs']==[[2,3]] and deleted['note']=='keep'
    assert got['imported']['rows']['p#r']['deleted']
    assert got['imported']['done']['0']


def test_synchronized_drag_zoom_and_fit():
    path=RULES.with_name('pixel_viewports.js')
    got=js('''
      const Viewports=require(PATH);
      function view(){return {scrollLeft:0,scrollTop:0,clientWidth:800,clientHeight:400,addEventListener(){},setPointerCapture(){}};}
      const a=view(),b=view(),canvases=[{width:200,height:80,style:{}},{width:200,height:80,style:{}}];
      const controller=new Viewports([a,b],canvases,()=>{});controller.fit();
      const fit=controller.scale;controller.zoom(8);const same=canvases[0].style.width===canvases[1].style.width;
      controller.down({clientX:500,clientY:200,pointerId:1,preventDefault(){}},b);
      controller.move({clientX:450,clientY:180});controller.up();
      const synchronized=a.scrollLeft===b.scrollLeft&&a.scrollTop===b.scrollTop;
      const x=(a.scrollLeft+100)/controller.scale;controller.zoom(16,[100,100]);
      return {fit,same,synchronized,anchored:x===(a.scrollLeft+100)/controller.scale};
    '''.replace('PATH',json.dumps(str(path))))
    assert got=={'fit':3.99,'same':True,'synchronized':True,'anchored':True}


def test_offline_editor_load_autosave_delete_undo_and_restore_from_embedded_draft():
    # Run the actual editor with deterministic DOM/storage adapters, without accessing a browser.
    editor=RULES.with_name('pixel_editor.js')
    viewport=RULES.with_name('pixel_viewports.js')
    script=r'''
      const fs=require('fs'),vm=require('vm'),Viewports=require(VIEWPORT);
      const base={id:'p#r',page:0,box:[0,0,10,10],region:[[0,0],[10,0],[10,10]],runs:[[2,3]]};
      const pack={id:'fixture',standalone:true,image_revision:'clean',pages:[{id:'p',question:1,size:[10,10],image:'data:image/png;base64,x'}],rows:[base]};
      pack.initial_export={schema:'paper2-pixel-review-v1',pack:'fixture',pages:{p:{done:false,revisions:[{...base,confirmed:true,note:'my annotation'}]}}};
      const disk=new Map(),nodes=new Map();
      function node(id=''){
        if(nodes.has(id))return nodes.get(id);
        const n={id,style:{},dataset:{},value:2,children:[],clientWidth:800,clientHeight:400,scrollTop:0,scrollLeft:0,
          classList:{toggle(){},add(){},remove(){}},append(x){this.children.push(x);},replaceChildren(){this.children=[];},addEventListener(){},setPointerCapture(){},
          click(){this.onclick?.();},getBoundingClientRect:()=>({left:0,top:0,width:10,height:10}),
          getContext:()=>({drawImage(){},putImageData(){},getImageData:(x,y,w,h)=>({width:w,height:h,data:new Uint8ClampedArray(w*h*4).fill(20)})})};
        if(id)nodes.set(id,n);return n;
      }
      const buttons=['add','keep','paint','erase','pan'].map(mode=>{const n=node(mode);n.dataset.mode=mode;return n;});
      node('review-pack').textContent=JSON.stringify(pack);node('review-draft').textContent='null';
      const doc={visibilityState:'visible',getElementById:node,createElement:()=>node(),addEventListener(){},
        querySelectorAll:q=>q==='[data-mode]'?buttons:buttons.slice(0,2),querySelector:()=>buttons[2]};
      const ctx={document:doc,PixelReview:R,PixelViewports:Viewports,console,Uint8Array,Uint8ClampedArray,Map,Date,Math,JSON,Promise,setTimeout,clearTimeout,
        localStorage:{getItem:k=>disk.get(k)||null,setItem:(k,v)=>disk.set(k,v)},
        Image:class{async decode(){}},ImageData:class{constructor(data,width,height){Object.assign(this,{data,width,height});}},
        PixelOverview:class{constructor(c,handlers){this.handlers=handlers;}setImage(im,id,editing){this.selected=id;this.editing=editing;}draw(){}current(){return null;}fit(){}select(id){this.selected=id;}},
        fetch(){throw Error('Offline editor must not fetch');}};
      ctx.window={addEventListener(){}};vm.createContext(ctx);
      const code=fs.readFileSync(EDITOR,'utf8');vm.runInContext(code.slice(0,code.lastIndexOf('start().catch')),ctx);
      async function run(){
        await vm.runInContext('start()',ctx);
        const loaded=vm.runInContext('({count:rows.length,note:state().note,confirmed:state().confirmed,mode})',ctx);
        await vm.runInContext('show(0)',ctx);
        vm.runInContext("$('note').value='saved human note';$('note').oninput();",ctx);
        const saved=[...disk.values()].map(JSON.parse).find(s=>s.rows?.['p#r']?.note==='saved human note');
        vm.runInContext("deleteRegion('p#r')",ctx);await new Promise(r=>setTimeout(r,0));
        const deleted=vm.runInContext('exported().pages.p',ctx);
        await vm.runInContext('restore(history,future)',ctx);
        const restored=vm.runInContext('({count:rows.length,note:state().note,runs:state().runs})',ctx);
        node('review-draft').textContent=JSON.stringify(saved);disk.clear();await vm.runInContext('start()',ctx);
        const reopened=vm.runInContext('state().note',ctx);
        await vm.runInContext('show(0)',ctx);
        vm.runInContext("grey.fill(255);mask.fill(0);mask[44]=1;grey[44]=20;grey[45]=200;state().runs=R.encode(mask);$('cleanupThreshold').value=180;$('cleanupRadius').value=2;$('cleanupArea').value=3;$('cleanupGray').checked=true;$('cleanupDots').checked=false;",ctx);
        const before=vm.runInContext('JSON.stringify(state().runs)',ctx);
        vm.runInContext('previewCleanup()',ctx);
        const preview=vm.runInContext('({total:cleanupPreview.total,runs:JSON.stringify(state().runs)})',ctx);
        vm.runInContext("$('cleanupCancel').click()",ctx);
        const cancelled=vm.runInContext('JSON.stringify(state().runs)',ctx);
        vm.runInContext("previewCleanup();$('cleanupApply').click()",ctx);
        const applied=vm.runInContext('({runs:state().runs,confirmed:state().confirmed,preview:cleanupPreview})',ctx);
        await vm.runInContext('restore(history,future)',ctx);
        const undone=vm.runInContext('JSON.stringify(state().runs)',ctx);

        console.log(JSON.stringify({loaded,saved:!!saved,deleted,restored,reopened,before,preview,cancelled,applied,undone}));
      }run().catch(e=>{console.error(e);process.exit(1);});
    '''.replace('VIEWPORT',json.dumps(str(viewport))).replace('EDITOR',json.dumps(str(editor)))
    command=f'const R=require({json.dumps(str(RULES))});'+script
    got=json.loads(subprocess.check_output([NODE,'-e',command],text=True))
    assert got['loaded']=={'count':1,'note':'my annotation','confirmed':True,'mode':'paint'}
    assert got['saved'] and got['deleted']['revisions']==[]
    assert got['deleted']['deleted_revisions'][0]['note']=='saved human note'
    assert got['restored']=={'count':1,'note':'saved human note','runs':[[2,3]]}
    assert got['reopened']=='saved human note'

    assert got['preview']=={'total':1,'runs':got['before']}
    assert got['cancelled']==got['before']==got['undone']
    assert got['applied']=={'runs':[[44,2]],'confirmed':False,'preview':None}


def test_residue_proposals_are_local_protect_dark_ink_and_do_not_mutate_masks():
    got=js('''
      const w=40,h=24,g=new Uint8Array(w*h).fill(255),m=new Uint8Array(w*h);
      function ink(x,y,value,erased=false){g[y*w+x]=value;if(erased)m[y*w+x]=1;}
      ink(5,5,30,true);ink(6,5,190); // pale leftover next to erase
      ink(5,15,30,true);ink(6,15,150); // dark fragment: dot mode only
      ink(16,5,30,true);ink(17,5,210);ink(18,5,30);ink(19,5,30);ink(20,5,30); // preserved character
      ink(30,15,30,true);ink(32,15,100); // originally separate punctuation
      ink(25,8,200); // far from any erase
      ink(1,0,30,true);ink(2,0,220); // crop boundary, ambiguous
      const before=Array.from(m),a=R.residueCandidates(g,m,w,{dots:false});
      const b=R.residueCandidates(g,m,w,{gray:false});
      const strict=R.residueCandidates(g,m,w,{threshold:200,dots:false});
      const empty=R.residueCandidates(g,new Uint8Array(w*h),w);
      return {gray:Array.from(a.added).flatMap((x,i)=>x?[i]:[]),dots:Array.from(b.added).flatMap((x,i)=>x?[i]:[]),strict:strict.total,empty:empty.total,unchanged:JSON.stringify(before)===JSON.stringify(Array.from(m))};
    ''')
    assert got=={'gray':[206],'dots':[206,606],'strict':0,'empty':0,'unchanged':True}
