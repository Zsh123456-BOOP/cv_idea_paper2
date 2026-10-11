"""Full-corpus review: preserve human work, isolate validation and persist safely."""
import json
from argparse import Namespace
from pathlib import Path
import shutil
import subprocess

import pytest
from PIL import Image

from revrender.tools.annotate.pixel_library import manual_records, overlap, bundle, duplicate_groups
from revrender.tools.annotate.pixel_offline import EmbeddedReview

HERE=Path(__file__).parents[1]/'revrender/tools/annotate'


def test_duplicate_crops_keep_mask_union_and_separate_neighbouring_lines():
    from revrender.data.pixel_masks import on_page
    import numpy as np
    def row(name,box,page=0,**extra):
        l,t,r,b=box
        return {'id':name,'page':page,'detector_index':0,'box':box,
                'region':[[l,t],[r,t],[r,b],[l,b]],'runs':[[0,1]],**extra}
    rows=[row('a',[0,0,40,10]),row('b',[0,0,38,10]),row('c',[1,0,40,10]),
          row('next-line',[0,8,40,18]),row('other-page',[0,0,40,10],1),
          row('human',[0,0,40,10],preserved_manual=True),row('distant',[50,0,60,10])]
    before=json.dumps(rows);groups=duplicate_groups(rows)
    assert len(groups)==1 and groups[0]['ids']==['a','b','c']
    expected=np.logical_or.reduce([on_page(r,[80,30]) for r in rows[:3]])
    assert np.array_equal(on_page(groups[0],[80,30]),expected)
    assert json.dumps(rows)==before
    # Do not transitively merge a long chain based only on consecutive overlap.
    chain=[row(str(i),[i*3,0,40+i*3,10]) for i in range(3)]
    assert duplicate_groups(chain)[0]['ids']==['0','1']


def test_dedup_migration_archives_aliases_and_preserves_human_edits():
    code='''
    const R=require(RULES);
    const a={id:'a',page:0,box:[0,0,10,10],region:[[0,0],[10,0],[10,10],[0,10]],runs:[[0,1]],detector_index:0};
    const b={...a,id:'b',runs:[[5,1]],detector_index:1};
    const pack={id:'x',pages:[{id:'p',size:[20,20]}],rows:[a,b],duplicate_groups:[{ids:['a','b'],box:a.box,region:a.region,runs:[[0,1],[5,1]]}]};
    const fresh=()=>({rows:{a:R.restoreRow(a),b:R.restoreRow(b)},extra:[],done:{},pageSeconds:{}});
    const store=fresh();store.rows.a.seconds=3;store.rows.b.seconds=5;
    const count=R.mergeDuplicates(pack,store);store.rows.a.confirmed=true;store.done[0]=true;
    const before=JSON.stringify(store),again=R.mergeDuplicates(pack,store);
    const pages=R.exportPages(pack,store);
    const imported=R.importData(pack,fresh(),{schema:'paper2-pixel-review-v1',pack:'x',pages});
    const cases=[];
    for(const edit of [{confirmed:true},{ops:1,runs:[]},{deleted:true},{note:'human'},{text_confirmed:true},{annotationMode:'keep'},{box:[1,0,10,10]}]){
      const s=fresh();Object.assign(s.rows.b,edit);const before=JSON.stringify(s);const n=R.mergeDuplicates(pack,s);cases.push(n===0&&before===JSON.stringify(s));
    }
    console.log(JSON.stringify({count,again,unchanged:before===JSON.stringify(store),pages,imported,cases}));
    '''.replace('RULES',json.dumps(str(HERE/'pixel_rules.js')))
    got=node(code)
    assert got['count']==1 and got['again']==0 and got['unchanged']
    page=got['pages']['p']
    assert len(page['revisions'])==1 and len(page['duplicate_revisions'])==1
    assert page['revisions'][0]['runs']==[[0,1],[5,1]]
    assert page['seconds']==8 and page['deleted_revisions']==[]
    assert got['imported']['rows']['b']['duplicate_of']=='a'
    assert got['imported']['rows']['a']['confirmed'] and got['imported']['done']['0']
    assert all(got['cases'])


def test_navigation_is_above_images_and_optional_fields_collapsed():
    from html.parser import HTMLParser
    class Tags(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.stack=[];self.parents={}
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if 'id' in attrs:self.ids.append(attrs['id']);self.parents[attrs['id']]=list(self.stack)
            if tag not in ('input','meta','br'):self.stack.append((tag,attrs))
        def handle_endtag(self,tag):
            for i in range(len(self.stack)-1,-1,-1):
                if self.stack[i][0]==tag:self.stack=self.stack[:i];break
    tree=Tags();tree.feed((HERE/'pixel_editor.html').read_text())
    assert len(tree.ids)==len(set(tree.ids))
    assert tree.ids.index('next')<tree.ids.index('edit')
    assert tree.ids.index('confirm')<tree.ids.index('edit')
    assert any(t=='details' and a.get('class')=='extra-details' and 'open' not in a for t,a in tree.parents['note'])


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
    assert 'window.ReviewLibraryStore=' in files[0].read_text()
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
