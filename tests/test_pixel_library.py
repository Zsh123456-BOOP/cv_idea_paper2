"""Full-corpus review: preserve human work, isolate validation and persist safely."""
import json
from argparse import Namespace
from pathlib import Path
import shutil
import subprocess

import pytest
from PIL import Image

from revrender.tools.annotate.pixel_library import manual_records, overlap, bundle
from revrender.tools.annotate.pixel_offline import EmbeddedReview

HERE=Path(__file__).parents[1]/'revrender/tools/annotate'


def test_only_human_actions_are_preserved_and_latest_export_wins(tmp_path):
    base={'schema':'paper2-pixel-review-v1','pages':{'p':{'revisions':[
        {'id':'confirmed','confirmed':True,'runs':[[0,1]]},
        {'id':'edited','ops':2,'confirmed':False},
        {'id':'untouched','ops':0,'confirmed':False}],
        'deleted_revisions':[{'id':'deleted','deleted':True}]}}}
    a=tmp_path/'a.json';a.write_text(json.dumps(base))
    b=tmp_path/'b.json';b.write_text(json.dumps({'schema':base['schema'],'pages':{'p':{'revisions':[{'id':'edited','ops':3}]}}}))
    result=manual_records([a,b])
    assert set(result['p'])=={'confirmed','edited','deleted'}
    assert result['p']['edited']['ops']==3
    assert overlap([0,0,10,10],[2,2,8,8])==1
    assert overlap([0,0,10,10],[20,20,30,30])==0


def test_bundle_keeps_all_pages_rows_splits_and_human_labels(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'images').mkdir()
    pages=[];rows=[]
    for i,split in enumerate(('train','val','test')):
        Image.new('L',(20,20),80).save(source/f'images/{i}.png')
        pages.append({'id':f'p{i}','image':f'images/{i}.png','question':'1','size':[20,20],'split':split})
        rows.append({'id':f'p{i}#r','page':i,'box':[0,0,10,10],'region':[[0,0],[10,0],[10,10],[0,10]],'runs':[[0,2]],'confirmed':True,'preserved_manual':True})
    data={'id':'full','schema':'paper2-pixel-review-pack-v1','task':'strike_only','image_revision':'review-clean-v1','model_revision':'model-A','image_source':'images','selected_checkpoint':'candidate','selection':{'selected':'manual','models':{}},'protected_validation_writers':['held'],'pages':pages,'rows':rows}
    (source/'pack.json').write_text(json.dumps(data))
    out=tmp_path/'out';bundle(Namespace(pack=source,output_dir=out,pages_per_part=2))
    files=sorted((out/'parts').glob('*.html'));assert len(files)==2
    packed=[json.loads(EmbeddedReview(p.read_text()).values['review-pack']) for p in files]
    assert sum(len(p['rows']) for p in packed)==3
    assert [p['split'] for group in packed for p in group['pages']]==['train','val','test']
    assert packed[1]['rows'][0]['page']==0
    assert packed[0]['initial_export']['pages']['p0']['revisions'][0]['runs']==[[0,2]]
    assert 'library_store.js' in files[0].read_text()
    assert len({p['id'] for p in packed})==2


def node(code):
    executable=shutil.which('node')
    if not executable:pytest.skip('needs node')
    return json.loads(subprocess.check_output([executable,'-e',code],text=True))


def test_corpus_export_uses_trusted_splits_and_excludes_unconfirmed():
    script=(HERE/'pixel_library.js').read_text()
    # Execute the real aggregation functions before the UI event handlers.
    script=script.split("$('resume').onclick")[0]
    code='''const vm=require('vm');const c={window:{addEventListener(){},REVIEW_LIBRARY:{id:'full',parts:[],initial_pages:{train:{revisions:[{id:'yes',confirmed:true},{id:'no',confirmed:false},{id:'unknown',confirmed:true,uncertainMask:{runs:[[0,1]]}}]},val:{split:'train',revisions:[{id:'held',confirmed:true}]}},page_splits:{train:'train',val:'val'}}},document:{getElementById:()=>({})},Map,Date,JSON};vm.createContext(c);vm.runInContext(SCRIPT,c);console.log(JSON.stringify(vm.runInContext('output(true)',c)));'''.replace('SCRIPT',json.dumps(script))
    got=node(code);assert set(got['pages'])=={'train'}
    assert [r['id'] for r in got['pages']['train']['revisions']]==['yes']


def test_indexeddb_serializes_snapshots_and_recovers_after_failed_write():
    code='''
    const vm=require('vm'),fs=require('fs'),disk=new Map();let fail=false;
    const fake={createObjectStore(){},transaction(){const tx={objectStore:()=>({
      put(value,key){setTimeout(()=>{if(fail){fail=false;tx.error=Error('quota');tx.onerror();}else{disk.set(key,value);tx.oncomplete();}},1);},
      get(key){const req={};setTimeout(()=>{req.result=disk.get(key);req.onsuccess();},1);return req;}
    })};return tx;}};
    const c={window:{},structuredClone,indexedDB:{open(){const req={result:fake};setTimeout(()=>req.onsuccess(),1);return req;}}};
    vm.createContext(c);vm.runInContext(SCRIPT,c);
    (async()=>{const s=c.window.ReviewLibraryStore,v={label:1};const first=s.put('a',v);v.label=2;await first;const original=await s.get('a');
      fail=true;let rejected=false;try{await s.put('a',{label:3});}catch(e){rejected=true;}
      await s.put('a',{label:4});await s.flush();console.log(JSON.stringify({original,rejected,final:await s.get('a')}));})();
    '''.replace('SCRIPT',json.dumps((HERE/'library_store.js').read_text()))
    assert node(code)=={'original':{'label':1},'rejected':True,'final':{'label':4}}


def test_embedded_editors_save_through_the_single_library_origin():
    code='''
    const vm=require('vm'),disk=new Map(),handlers=new Map();
    function windowObject(){const w={addEventListener:(k,f)=>{if(!handlers.has(w))handlers.set(w,new Set());handlers.get(w).add(f);},removeEventListener:(k,f)=>handlers.get(w)?.delete(f)};return w;}
    const host=windowObject(),child=windowObject();host.parent=host;child.parent=host;host.REVIEW_LIBRARY={id:'fixture'};
    host.postMessage=m=>setTimeout(()=>{for(const fn of handlers.get(host)||[])fn({source:child,data:m});},0);
    child.postMessage=m=>setTimeout(()=>{for(const fn of handlers.get(child)||[])fn({source:host,data:m});},0);
    const database={transaction(){const tx={objectStore:()=>({put(v,k){disk.set(k,v);setTimeout(()=>tx.oncomplete(),0);},get(k){const r={};setTimeout(()=>{r.result=disk.get(k);r.onsuccess();},0);return r;}})};return tx;}};
    const a={window:host,document:{querySelectorAll:()=>[{contentWindow:child}]},setTimeout,clearTimeout,structuredClone,indexedDB:{open(){const r={result:database};setTimeout(()=>r.onsuccess(),0);return r;}}};
    const b={window:child,setTimeout,clearTimeout,structuredClone,indexedDB:{open(){throw Error('child must not open its own storage');}}};
    for(const c of [a,b]){vm.createContext(c);vm.runInContext(SCRIPT,c);}
    (async()=>{await child.ReviewLibraryStore.put('pixel-review-fixture-part',{human:'keep'});await child.ReviewLibraryStore.flush();
      console.log(JSON.stringify({host:await host.ReviewLibraryStore.get('pixel-review-fixture-part'),child:await child.ReviewLibraryStore.get('pixel-review-fixture-part')}));})();
    '''.replace('SCRIPT',json.dumps((HERE/'library_store.js').read_text()))
    assert node(code)=={'host':{'human':'keep'},'child':{'human':'keep'}}
