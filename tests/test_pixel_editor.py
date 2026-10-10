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
      const disk=new Map(),nodes=new Map(),handlers={};
      function node(id=''){
        if(nodes.has(id))return nodes.get(id);
        const n={id,style:{},dataset:{},value:2,children:[],clientWidth:800,clientHeight:400,scrollTop:0,scrollLeft:0,
          after(){},setAttribute(){},classList:{toggle(){},add(){},remove(){}},append(x){this.children.push(x);},replaceChildren(){this.children=[];},addEventListener(){},setPointerCapture(){},
          click(){this.onclick?.();},getBoundingClientRect:()=>({left:0,top:0,width:10,height:10}),
          getContext:()=>({drawImage(){},putImageData(im){n.lastImage=im;},getImageData:(x,y,w,h)=>({width:w,height:h,data:new Uint8ClampedArray(w*h*4).fill(20)})})};
        if(id)nodes.set(id,n);return n;
      }
      const buttons=['add','keep','paint','erase','pan'].map(mode=>{const n=node(mode);n.dataset.mode=mode;return n;});
      node('review-pack').textContent=JSON.stringify(pack);node('review-draft').textContent='null';
      const doc={visibilityState:'visible',getElementById:node,createElement:()=>node(),addEventListener:(name,fn)=>handlers[name]=fn,
        querySelectorAll:q=>q==='[data-mode]'?buttons:buttons.slice(0,2),querySelector:q=>q==='header'?node('header'):buttons.find(b=>q.includes('\"'+b.dataset.mode+'\"'))};
      const ctx={document:doc,PixelReview:R,PixelViewports:Viewports,console,Uint8Array,Uint8ClampedArray,Map,Date,Math,JSON,Promise,setTimeout,clearTimeout,
        localStorage:{getItem:k=>disk.get(k)||null,setItem:(k,v)=>disk.set(k,v)},
        Image:class{async decode(){}},ImageData:class{constructor(data,width,height){Object.assign(this,{data,width,height});}},
        PixelOverview:class{constructor(c,handlers){this.handlers=handlers;}setImage(im,id,editing){this.selected=id;this.editing=editing;}draw(){}current(){return null;}fit(){}select(id){this.selected=id;}},
        fetch(){throw Error('Offline editor must not fetch');}};
      ctx.window={addEventListener(){}};vm.createContext(ctx);
      const code=fs.readFileSync(EDITOR,'utf8');vm.runInContext(code.slice(0,code.lastIndexOf('start().then')),ctx);
      async function run(){
        await vm.runInContext('start()',ctx);
        const loaded=vm.runInContext('({count:rows.length,note:state().note,confirmed:state().confirmed,mode})',ctx);
        await vm.runInContext('show(0)',ctx);
        const keyEvent=(key,tagName='BUTTON',extra={})=>({key,target:{tagName},preventDefault(){},...extra});
        handlers.keydown(keyEvent('e'));const eraseKey=vm.runInContext('mode',ctx);
        handlers.keydown(keyEvent('b','INPUT'));const textKey=vm.runInContext('mode',ctx);
        handlers.keydown(keyEvent('b','BUTTON',{isComposing:true}));const imeKey=vm.runInContext('mode',ctx);
        handlers.keydown(keyEvent('B'));const paintKey=vm.runInContext('mode',ctx);
        vm.runInContext("$('note').value='saved human note';$('note').oninput();",ctx);
        const saved=[...disk.values()].map(JSON.parse).find(s=>s.rows?.['p#r']?.note==='saved human note');
        vm.runInContext("deleteRegion('p#r')",ctx);await new Promise(r=>setTimeout(r,0));
        const deleted=vm.runInContext('exported().pages.p',ctx);
        await vm.runInContext('restore(history,future)',ctx);
        const restored=vm.runInContext('({count:rows.length,note:state().note,runs:state().runs})',ctx);
        node('review-draft').textContent=JSON.stringify(saved);disk.clear();await vm.runInContext('start()',ctx);
        const reopened=vm.runInContext('state().note',ctx);
        await vm.runInContext('show(0)',ctx);
        vm.runInContext("grey.fill(255);mask.fill(0);mask[44]=1;grey[44]=20;grey[45]=150;state().runs=R.encode(mask);",ctx);
        const before=vm.runInContext('JSON.stringify(state().runs)',ctx);
        vm.runInContext('remember();changed()',ctx);
        const applied=vm.runInContext('({runs:state().runs,confirmed:state().confirmed})',ctx);
        const automaticSaved=[...disk.values()].map(JSON.parse).some(s=>JSON.stringify(s.rows?.['p#r']?.runs)==='[[44,2]]');
        vm.runInContext("remember();$('radius').value=0;drawing=true;dab(5,4,false);finishPaint();",ctx);
        const erased=vm.runInContext('mask[45]===0&&R.keepPixels(state())[45]===1',ctx);
        vm.runInContext('state().runs=[[44,2]];cleanActive();',ctx);
        const protectedAgain=vm.runInContext('mask[45]===0',ctx);
        await vm.runInContext('restore(history,future)',ctx);

        await vm.runInContext('restore(history,future)',ctx);
        const undone=vm.runInContext('JSON.stringify(state().runs)',ctx);
        const stamp=vm.runInContext('JSON.stringify(state().autoCleanup)===JSON.stringify(R.cleanupStamp(state()))',ctx);
        vm.runInContext("state().region=[[0,0],[10,0],[10,3],[3,3],[3,10],[0,10]];state().runs=[];delete state().outsideMask;",ctx);
        await vm.runInContext('show(0)',ctx);
        const outsideWhite=vm.runInContext('original.data[(8*10+8)*4]===255',ctx);
        vm.runInContext('dab(8,8,true);changed()',ctx);
        const brushOutside=vm.runInContext('mask[88]',ctx);

        vm.runInContext("state().region=[[0,0],[10,0],[10,10],[0,10]];state().runs=[[36,1]];delete state().manualKeep;state().uncertainMask={box:[0,0,10,10],runs:[[34,1]]};",ctx);
        await vm.runInContext('show(0)',ctx);
        vm.runInContext("grey.fill(255);for(const i of [33,34,36,88])grey[i]=20;",ctx);
        handlers.keydown(keyEvent('2','TEXTAREA'));const modeTextGuard=vm.runInContext('state().annotationMode',ctx);
        handlers.keydown(keyEvent('2'));const keepMode=vm.runInContext('state().annotationMode',ctx);
        const emptyKeep=vm.runInContext('[mask[33],mask[34],mask[36],mask[88]]',ctx);
        vm.runInContext("remember();drawing=true;dab(3,3,'paint');dab(4,3,'paint');paint();",ctx);
        const liveKeep=vm.runInContext('mask[33]===0&&mask[34]===0',ctx);
        vm.runInContext('finishPaint()',ctx);
        handlers.keydown(keyEvent('e'));
        vm.runInContext("remember();drawing=true;dab(4,3,mode);paint();",ctx);
        const liveErase=vm.runInContext('mask[34]===1&&keepMarks[34]===0',ctx);
        vm.runInContext('finishPaint()',ctx);
        const keepResult=vm.runInContext('state().runs',ctx);
        const greenOnly=node('edit').lastImage.data[33*4+1]>node('edit').lastImage.data[33*4]&&node('edit').lastImage.data[36*4]===20;
        const cleanResult=node('result').lastImage.data[33*4]===20&&node('result').lastImage.data[34*4]===255;
        handlers.keydown(keyEvent('1'));
        const restoredMode=vm.runInContext('({choice:state().annotationMode,runs:state().runs,unknown:uncertain[34]})',ctx);
        const redOnly=node('edit').lastImage.data[36*4]>node('edit').lastImage.data[36*4+1]&&node('edit').lastImage.data[33*4]===20;
        handlers.keydown(keyEvent('2'));
        const retainedKeep=vm.runInContext('keepMarks[33]===1&&keepMarks[34]===0&&mask[34]===1',ctx);
        vm.runInContext("remember();drawing=true;dab(3,3,'erase');finishPaint();",ctx);
        const cleared=vm.runInContext('keepMarks[33]===0&&mask[33]===1',ctx);
        await vm.runInContext('restore(history,future)',ctx);
        const keepUndo=vm.runInContext("state().annotationMode==='keep'&&keepMarks[33]===1&&mask[33]===0",ctx);
        vm.runInContext("$('transcription').value='x + ?';$('transcription').oninput();$('textConfirmed').checked=true;$('textConfirmed').onchange();",ctx);
        const separateText=vm.runInContext('state().text_confirmed&&!state().confirmed',ctx);
        const labelled=vm.runInContext('exported()',ctx);
        node('review-draft').textContent=vm.runInContext('JSON.stringify(store)',ctx);disk.clear();await vm.runInContext('start()',ctx);
        const labelReload=vm.runInContext("({text:state().text,checked:state().text_confirmed,choice:state().annotationMode,green:keepMarks[33],erased:mask[34]})",ctx);
        await vm.runInContext('overview(0)',ctx);
        vm.runInContext("whole.handlers.reshape('p#r',[[0,0],[10,0],[10,3],[3,3],[3,10],[0,10]]);",ctx);
        const croppedExport=vm.runInContext('exported().pages.p.revisions[0].runs.reduce((n,r)=>n+r[1],0)',ctx);
        vm.runInContext("whole.handlers.reshape('p#r',[[0,0],[10,0],[10,10],[0,10]]);",ctx);
        const expandedExport=vm.runInContext('exported().pages.p.revisions[0].runs.reduce((n,r)=>n+r[1],0)',ctx);

        vm.runInContext("pack.task='region_selection';",ctx);await vm.runInContext('overview(0)',ctx);
        vm.runInContext('whole.handlers.create([[0,0],[3,0],[3,3],[0,3]]);',ctx);
        vm.runInContext("$('pageDone').onclick();",ctx);
        const frameMode=vm.runInContext("({detail:inDetail,task:exported().task,done:store.done[0],confirmed:rows.some(r=>ensure(r).confirmed),empty:rows.at(-1).runs.length===0})",ctx);
        node('review-pack').textContent=JSON.stringify({...pack,id:'empty-frames',task:'region_selection',rows:[],initial_export:undefined});
        node('review-draft').textContent='null';disk.clear();await vm.runInContext('start()',ctx);
        const blankFrames=vm.runInContext("({rows:rows.length,detail:inDetail,task:pack.task})",ctx);
        const libraryDisk=new Map(),libraryKey='pixel-review-library-fixture-offline-clean';
        ctx.ReviewLibraryStore={async get(k){return libraryDisk.get(k);},async put(k,v){libraryDisk.set(k,JSON.parse(JSON.stringify(v)));},async flush(){}};
        const auto={...base,id:'p#auto',detector_index:0,runs:[[3,1]]};
        const human=R.restoreRow(base,{confirmed:true,note:'protected human',runs:[[4,1]]});
        libraryDisk.set(libraryKey,{draft:{rows:{'p#r':human,'p#auto':R.restoreRow(auto,{runs:[[7,1]]})},extra:[],done:{},pageSeconds:{},imageRevision:'clean',modelRevision:'old'}});
        node('review-pack').textContent=JSON.stringify({...pack,id:'library-fixture',model_revision:'new',library:{id:'library',part:0,parts:1},rows:[base,auto]});
        node('review-draft').textContent='null';await vm.runInContext('start()',ctx);
        const library=vm.runInContext("({note:store.rows['p#r'].note,human:store.rows['p#r'].runs,automatic:store.rows['p#auto'].runs,model:store.modelRevision})",ctx);
        const librarySaved=libraryDisk.get(libraryKey).export.pages.p.revisions.length===2;
        const messages=[],windowHandlers={};ctx.window.parent={postMessage:m=>messages.push(m)};
        ctx.window.addEventListener=(name,fn)=>windowHandlers[name]=fn;ctx.location={hash:'#import=fixture'};ctx.URLSearchParams=URLSearchParams;
        await vm.runInContext('libraryImportSetup()',ctx);
        const imported={...base,id:'p#manual',ops:2,confirmed:false,runs:[[8,1]],note:'new correction'};
        await windowHandlers.message({source:ctx.window.parent,data:{type:'review-library-import',token:'fixture',pages:{p:{revisions:[{...base,confirmed:true,note:'must not replace'},imported]}}}});
        const libraryImport=vm.runInContext("({human:store.rows['p#r'].note,added:store.rows['p#manual'].note,suppressed:store.rows['p#auto'].deleted})",ctx);
        await windowHandlers.message({source:ctx.window.parent,data:{type:'review-library-import',token:'fixture',pages:{p:{revisions:[{...imported,id:'bad',box:[-1,0,10,10]}]}}}});
        const rejectedImport=!!messages.at(-1).error;
        console.log(JSON.stringify({libraryImport,rejectedImport,library,librarySaved,blankFrames,frameMode,erased,protectedAgain,loaded,saved:!!saved,deleted,restored,reopened,before,applied,undone,automaticSaved,stamp,outsideWhite,brushOutside,eraseKey,textKey,imeKey,paintKey,modeTextGuard,keepMode,emptyKeep,liveKeep,liveErase,keepResult,greenOnly,cleanResult,restoredMode,redOnly,retainedKeep,cleared,keepUndo,separateText,labelled,labelReload,croppedExport,expandedExport}));
      }run().catch(e=>{console.error(e);process.exit(1);});
    '''.replace('VIEWPORT',json.dumps(str(viewport))).replace('EDITOR',json.dumps(str(editor)))
    command=f'const R=require({json.dumps(str(RULES))});'+script
    got=json.loads(subprocess.check_output([NODE,'-e',command],text=True))
    assert got['libraryImport']=={'human':'protected human','added':'new correction','suppressed':True}
    assert got['rejectedImport']
    assert got['library']=={'note':'protected human','human':[[4,1]],'automatic':[[3,1]],'model':'new'}
    assert got['librarySaved']
    assert got['blankFrames']=={'rows':0,'detail':False,'task':'region_selection'}
    assert got['frameMode']=={'detail':False,'task':'region_selection','done':True,'confirmed':False,'empty':True}
    assert got['erased'] and got['protectedAgain']
    assert got['loaded']=={'count':1,'note':'my annotation','confirmed':True,'mode':'paint'}
    assert got['saved'] and got['deleted']['revisions']==[]
    assert got['deleted']['deleted_revisions'][0]['note']=='saved human note'
    assert got['restored']=={'count':1,'note':'saved human note','runs':[[2,3]]}
    assert got['reopened']=='saved human note'

    assert got['before']==got['undone'] and got['stamp'] and got['automaticSaved']
    assert got['applied']=={'runs':[[44,2]],'confirmed':False}
    assert got['outsideWhite'] and got['brushOutside']==0
    assert [got[k] for k in ['eraseKey','textKey','imeKey','paintKey']]==['erase','erase','erase','paint']
    assert got['modeTextGuard']=='erase' and got['keepMode']=='keep'
    assert got['emptyKeep']==[1,1,1,1]
    assert got['liveKeep'] and got['liveErase'] and got['greenOnly'] and got['redOnly'] and got['cleanResult']
    assert got['keepResult']==[[34,1],[36,1],[88,1]]
    assert got['restoredMode']=={'choice':'erase','runs':[[36,1]],'unknown':1}
    assert got['retainedKeep'] and got['cleared'] and got['keepUndo'] and got['separateText']
    assert got['labelReload']=={'text':'x + ?','checked':True,'choice':'keep','green':1,'erased':1}
    assert got['labelled']['annotation_version']==3
    assert got['labelled']['label_semantics']['runs']=='active_mode_erase_result'
    assert got['croppedExport']==51 and got['expandedExport']==99


def test_exclusive_modes_roundtrip_crop_and_export_without_overwriting_each_other():
    got=js('''
      const base={id:'p#r',page:0,box:[0,0,10,10],region:[[0,0],[10,0],[10,3],[3,3],[3,10],[0,10]],runs:[[25,1]]};
      const s=R.restoreRow(base),grey=new Uint8Array(100).fill(255),oldKeep=new Uint8Array(100);
      for(const i of [22,23,25,77])grey[i]=20;
      oldKeep[22]=1;R.saveKeep(s,oldKeep);
      let keep=R.changeMode(s,'keep',grey);const first=keep.runs;
      const marks=R.positivePixels(keep);marks[23]=1;R.savePositive(keep,marks);R.autoClean(keep,grey);
      const allKept=keep.runs;
      const small=R.reshape(keep,[[0,0],[3,0],[3,3],[0,3]],[10,10]);R.savePositive(small,R.positivePixels(small));
      const expanded=R.reshape(small,base.region,[10,10]);R.autoClean(expanded,grey);
      const pack={id:'fixture',pages:[{id:'p',size:[10,10]}],rows:[base]},store={rows:{'p#r':expanded},extra:[],done:{},pageSeconds:{}};
      const imported=R.importData(pack,{rows:{},extra:[],done:{},pageSeconds:{}},{schema:'paper2-pixel-review-v1',pack:pack.id,pages:R.exportPages(pack,store)});
      const reloaded=imported.rows['p#r'];
      const erase=R.changeMode(reloaded,'erase',grey),again=R.changeMode(erase,'keep',grey);
      const cropped=R.changeMode(s,'keep',grey);const newRegion=[[0,0],[3,0],[3,10],[0,10]];
      const eraseSmall=R.changeMode(R.reshape(cropped,newRegion,[10,10]),'erase',new Uint8Array(30).fill(20));
      const eraseExpanded=R.reshape(eraseSmall,base.region,[10,10]);
      return {first,allKept,expanded:expanded.runs,green:R.positivePixels(reloaded)[23],erase:erase.runs,again:again.runs,original:s.runs,geometry:eraseSmall.region,restoredOutside:eraseExpanded.runs};
    ''')
    assert got['first']==[[23,1],[25,1]]
    assert got['allKept']==got['expanded']==got['again']==[[25,1]]
    assert got['green']==1 and got['erase']==got['original']==got['restoredOutside']==[[25,1]]
    assert got['geometry']==[[0,0],[3,0],[3,10],[0,10]]


def test_legacy_cleaned_draft_is_upgraded_without_another_erase_pass():
    got=js('''
      const s={box:[0,0,20,20],runs:[[105,1]],confirmed:true,manualKeep:{box:[0,0,20,20],runs:[[110,1]]}};
      const stamp=R.cleanupStamp(s);delete stamp.labelVersion;delete stamp.manualKeep;delete stamp.uncertainMask;s.autoCleanup=stamp;
      const grey=new Uint8Array(400).fill(255);grey[105]=20;grey[106]=150;
      const added=R.autoClean(s,grey);
      return {added,runs:s.runs,confirmed:s.confirmed,version:s.autoCleanup.labelVersion};
    ''')
    assert got=={'added':0,'runs':[[105,1]],'confirmed':True,'version':1}


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
      const before=Array.from(m),a=R.residueCandidates(g,m,w,{threshold:180,radius:2,maxArea:3,dots:false});
      const b=R.residueCandidates(g,m,w,{threshold:180,radius:2,maxArea:3,gray:false});
      const strict=R.residueCandidates(g,m,w,{threshold:200,dots:false});
      const empty=R.residueCandidates(g,new Uint8Array(w*h),w);
      return {gray:Array.from(a.added).flatMap((x,i)=>x?[i]:[]),dots:Array.from(b.added).flatMap((x,i)=>x?[i]:[]),strict:strict.total,empty:empty.total,unchanged:JSON.stringify(before)===JSON.stringify(Array.from(m))};
    ''')
    assert got=={'gray':[206],'dots':[206,606],'strict':0,'empty':0,'unchanged':True}


def test_fixed_auto_cleanup_is_idempotent_and_keeps_approved_confirmation():
    got=js('''
      const grey=new Uint8Array(400).fill(255);grey[105]=20;grey[106]=150;grey[108]=170;grey[110]=190;
      const s={box:[0,0,20,20],runs:[[105,1]],confirmed:true};
      const first=R.autoClean(s,grey),once=JSON.stringify(s),second=R.autoClean(s,grey);
      const reload=JSON.parse(JSON.stringify(s)),third=R.autoClean(reload,grey);
      const negative={box:s.box,runs:[[105,1]],not_revision:true};
      const deleted={box:s.box,runs:[[105,1]],deleted:true};
      return {first,second,third,stable:once===JSON.stringify(s),confirmed:s.confirmed,
        negative:R.autoClean(negative,grey),deleted:R.autoClean(deleted,grey),config:R.AUTO_CLEANUP};
    ''')
    assert got['first']>0 and got['second']==got['third']==0 and got['stable'] and got['confirmed']
    assert got['negative']==got['deleted']==0
    assert got['config']=={'threshold':100,'radius':4,'maxArea':12,'gray':True,'dots':True}


def test_concave_polygon_clips_display_and_preserves_excluded_edits_on_expansion():
    got=js('''
      const region=[[10,20],[18,20],[18,23],[13,23],[13,28],[10,28]];
      const original={box:[10,20,18,28],region:[[10,20],[18,20],[18,28],[10,28]],runs:[[0,64]],note:'human'};
      const clipped=R.clipRegion(original,region,original.box),domain=R.regionPixels(region,original.box);
      const rgba=new Uint8ClampedArray(64*4).fill(25);for(let i=0;i<64;i++)rgba[i*4+3]=255;
      R.whiteOutside(rgba,domain);
      const notched=5*8+5,inside=5*8+1;
      const narrow=R.reshape(clipped,[[10,20],[13,20],[13,28],[10,28]],[40,40]);
      const expanded=R.reshape(JSON.parse(JSON.stringify(narrow)),original.region,[40,40]);
      const back=R.remap(expanded.runs,expanded.box,original.box);
      const pack={pages:[{id:'p'}],rows:[{...clipped,id:'r',page:0}]};
      const exported=R.exportPages(pack,{rows:{r:clipped},extra:[],done:{},pageSeconds:{}}).p.revisions[0];
      return {active:R.decode(clipped.runs,64).reduce((a,b)=>a+b,0),parked:R.decode(clipped.outsideMask.runs,64).reduce((a,b)=>a+b,0),
        outside:Array.from(rgba.slice(notched*4,notched*4+4)),inside:Array.from(rgba.slice(inside*4,inside*4+4)),back,exported:exported.runs,note:expanded.note};
    ''')
    assert got['active']==39 and got['parked']==25
    assert got['outside']==[255,255,255,255] and got['inside']==[25,25,25,255]
    assert got['back']==[[0,64]] and got['note']=='human'
    assert sum(n for _,n in got['exported'])==39


def test_auto_cleanup_cannot_reach_adjacent_line_outside_polygon():
    got=js('''
      const g=new Uint8Array(100).fill(255);g[22]=20;g[23]=150;g[33]=150;
      const s={box:[0,0,10,10],region:[[0,0],[10,0],[10,3],[3,3],[3,10],[0,10]],runs:[[22,1]]};
      R.autoClean(s,g);const m=R.decode(s.runs,100);
      return {inside:m[23],outside:m[33],runs:s.runs};
    ''')
    assert got['inside']==1 and got['outside']==0


def test_fit_whole_crop_respects_height_and_does_not_stretch():
    path=RULES.with_name('pixel_viewports.js')
    got=js('''
      const V=require(PATH),views=[0,1].map(()=>({clientWidth:800,clientHeight:300,scrollLeft:0,scrollTop:0,addEventListener(){}}));
      const canvases=[0,1].map(()=>({width:400,height:1200,style:{}}));
      const v=new V(views,canvases,()=>{});v.fit();return {scale:v.scale,width:parseFloat(canvases[0].style.width),height:parseFloat(canvases[0].style.height)};
    '''.replace('PATH',json.dumps(str(path))))
    assert got['height']==298 and got['width']==pytest.approx(298/3)


def test_manual_eraser_protection_survives_reload_crop_and_export():
    got=js('''
      const region=[[0,0],[20,0],[20,20],[0,20]],grey=new Uint8Array(400).fill(255);
      grey[105]=20;grey[106]=150;grey[107]=180;
      let s={box:[0,0,20,20],region,runs:[[105,1]]};R.autoClean(s,grey);
      const keep=R.keepPixels(s);keep[106]=1;R.saveKeep(s,keep);
      s.runs=[[105,1]];R.autoClean(s,grey);
      const first=R.decode(s.runs,400)[106];
      s=R.reshape(s,[[0,0],[5,0],[5,20],[0,20]],[20,20]);
      s=R.reshape(JSON.parse(JSON.stringify(s)),region,[20,20]);
      const pack={id:'p',rows:[{...s,id:'r',page:0}],pages:[{id:'page',size:[20,20]}]};
      const pages=R.exportPages(pack,{rows:{r:s},extra:[],done:{},pageSeconds:{}});
      const store=R.importData(pack,{rows:{},extra:[],done:{},pageSeconds:{}},{schema:'paper2-pixel-review-v1',pack:'p',pages});
      s=store.rows.r;s.runs=[[105,3]];R.autoClean(s,grey);
      const persisted=R.decode(s.runs,400)[106],neighbor=R.decode(s.runs,400)[107];
      const paint=R.keepPixels(s);paint[106]=0;R.saveKeep(s,paint);s.runs=[[105,3]];R.autoClean(s,grey);
      return {first,persisted,neighbor,repaint:R.decode(s.runs,400)[106]};
    ''')
    assert got=={'first':0,'persisted':0,'neighbor':1,'repaint':1}
