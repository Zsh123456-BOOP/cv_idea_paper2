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
