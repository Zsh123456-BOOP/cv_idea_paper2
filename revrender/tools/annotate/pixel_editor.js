'use strict';
const $=id=>document.getElementById(id), R=PixelReview, clone=x=>JSON.parse(JSON.stringify(x));
let pack,rows,key,backupKey,store,at=0,wholePage=0,inDetail=false,mode='add',zoom=3,hidden=false;
let mask,original,grey,history=[],future=[],drawing=false,lastPoint=null,serial=0,busy=false,lastTime=0;
const images=new Map();
const row=()=>rows[at], state=()=>ensure(row());
function ensure(r){return store.rows[r.id]||(store.rows[r.id]=R.restoreRow(r));}
function syncRows(){rows=[...pack.rows,...store.extra];rows.forEach(r=>store.rows[r.id]=R.restoreRow(r,store.rows[r.id]));}
function merged(r){return {...r,...ensure(r),page:r.page};}
function save(){try{localStorage.setItem(key,JSON.stringify(store));}catch(e){status('自动保存失败，请立即导出标注：'+e.message,true);}}
function touch(){const now=Date.now();if(lastTime&&document.visibilityState==='visible'){const seconds=Math.min((now-lastTime)/1000,60);if(inDetail&&row())state().seconds+=seconds;else store.pageSeconds[wholePage]=(store.pageSeconds[wholePage]||0)+seconds;}lastTime=now;}
function status(text,error=false){$('status').textContent=text;$('status').className='status'+(error?' error':'');}
function invalidate(r){const s=ensure(r);s.confirmed=false;s.ops++;store.done[r.page]=false;}
function changed(){state().runs=R.encode(mask);invalidate(row());save();nav();}
function remember(){history.push({store:clone(store),at,wholePage,inDetail,chosen:whole.selected,editing:whole.editing});if(history.length>50)history.shift();future=[];}
function cancelPending(){serial++;busy=false;drawing=false;$('edit').style.opacity=1;}
async function imageOf(pi){if(images.has(pi))return images.get(pi);const im=new Image();im.src=pack.pages[pi].image;await im.decode();images.set(pi,im);return im;}
function cropData(pi,box){const im=images.get(pi),[l,t,r,b]=box,c=document.createElement('canvas');c.width=r-l;c.height=b-t;const cx=c.getContext('2d');cx.drawImage(im,l,t,r-l,b-t,0,0,r-l,b-t);return cx.getImageData(0,0,r-l,b-t);}
async function cleanStore(candidate){
  let removed=0;
  for(const base of [...pack.rows,...candidate.extra]){
    const s=candidate.rows[base.id]=R.restoreRow(base,candidate.rows[base.id]);
    R.validBox(s.box,pack.pages[base.page].size);R.validRegion(s.region,pack.pages[base.page].size);
    await imageOf(base.page);const clean=R.filterInk(s.runs,cropData(base.page,s.box).data);
    if(clean.removed){s.runs=clean.runs;s.confirmed=false;candidate.done[base.page]=false;s.cleaned_pixels=(s.cleaned_pixels||0)+clean.removed;removed+=clean.removed;}
  }
  candidate.imageRevision=pack.image_revision||'original';return removed;
}
function nav(){
  $('progress').textContent=`${rows.filter(r=>ensure(r).confirmed).length} / ${rows.length} 处`;
  const nav=$('nav');nav.replaceChildren();
  pack.pages.forEach((p,i)=>{
    const h=document.createElement('button');h.className='question'+(!inDetail&&i===wholePage?' active':'');
    h.textContent=`第 ${i+1} 题 · 题号 ${p.question}${store.done[i]?' ✓':''}`;h.onclick=()=>overview(i);nav.append(h);
    let number=0;
    rows.forEach((r,k)=>{if(r.page!==i)return;number++;const b=document.createElement('button');b.textContent=`${ensure(r).confirmed?'✓':'○'} 划线 ${number}`;if(inDetail&&k===at)b.className='active';b.onclick=()=>show(k);nav.append(b);});
  });
  const pi=inDetail?row()?.page:wholePage;
  $('pageDone').textContent=store.done[pi]?'整题已检查 ✓':'确认整题检查完成';
  $('prevPage').disabled=wholePage===0;$('nextPage').disabled=wholePage===pack.pages.length-1;
}
async function show(index){
  if(!rows[index])return;touch();save();cancelPending();at=index;wholePage=row().page;inDetail=true;
  $('overviewPanel').hidden=true;$('detailPanel').hidden=false;$('overviewBtn').hidden=false;
  const ticket=++serial,s=state();original=null;await imageOf(row().page);if(ticket!==serial)return;
  original=cropData(row().page,s.box);grey=new Uint8Array(original.width*original.height);
  for(let i=0;i<grey.length;i++)grey[i]=Math.round((original.data[4*i]+original.data[4*i+1]+original.data[4*i+2])/3);
  mask=R.decode(s.runs,grey.length);for(const id of ['edit','result']){$(id).width=original.width;$(id).height=original.height;}
  const n=rows.filter((r,i)=>r.page===row().page&&i<=at).length;
  $('title').textContent=`第 ${wholePage+1} 题 · 划线 ${n}`;$('notRevision').checked=s.not_revision;$('note').value=s.note;
  $('previous').disabled=at===0;$('next').disabled=at===rows.length-1;
  paint();nav();lastTime=Date.now();status(s.confirmed?'这处已确认。继续修改会取消确认。':'先检查范围和右侧文字，再修正误擦、漏擦。');
}
function paint(){
  if(!original)return;
  const over=new ImageData(new Uint8ClampedArray(original.data),original.width,original.height),clean=new ImageData(new Uint8ClampedArray(original.data),original.width,original.height);
  for(let i=0;i<mask.length;i++){
    if(!mask[i]||state().not_revision)continue;
    for(let c=0;c<3;c++)clean.data[4*i+c]=255;
    if(!hidden){over.data[4*i]=Math.round(over.data[4*i]*.45+230*.55);over.data[4*i+1]*=.45;over.data[4*i+2]*=.45;}
  }
  $('edit').getContext('2d').putImageData(over,0,0);$('result').getContext('2d').putImageData(clean,0,0);
  for(const id of ['edit','result']){$(id).style.width=original.width*zoom+'px';$(id).style.height=original.height*zoom+'px';}
  $('zoomLabel').textContent=zoom+'×';
}
function point(e){const c=$('edit'),b=c.getBoundingClientRect();return [Math.max(0,Math.min(c.width-1,Math.round((e.clientX-b.left)*c.width/b.width))),Math.max(0,Math.min(c.height-1,Math.round((e.clientY-b.top)*c.height/b.height)))];}
function dab(x,y,add){const radius=Number($('radius').value),w=original.width,h=original.height;for(let yy=Math.max(0,y-radius);yy<=Math.min(h-1,y+radius);yy++)for(let xx=Math.max(0,x-radius);xx<=Math.min(w-1,x+radius);xx++)if((xx-x)**2+(yy-y)**2<=radius**2)mask[yy*w+xx]=add&&grey[yy*w+xx]<245?1:0;}
async function suggest(x,y,add){
  if(busy)return;touch();remember();const ticket=++serial;busy=true;$('edit').style.opacity=.65;status('正在按你的提示修正附近笔画…');
  try{
    const response=await fetch('/refine',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({page:row().page,box:state().box,runs:R.encode(mask),clicks:[[x,y,add]],radius:28})});
    const data=await response.json();if(!response.ok)throw Error(data.error||'服务未就绪');if(ticket!==serial)return;
    mask=R.decode(R.filterInk(data.runs,original.data).runs,mask.length);changed();paint();status('已局部修正。检查右侧是否伤字；不合适可撤销或用画笔。');
  }catch(e){if(ticket===serial){history.pop();status('模型修正未完成：'+e.message+'。可以继续用画笔和橡皮。',true);}}
  finally{if(ticket===serial){busy=false;$('edit').style.opacity=1;}}
}
$('edit').onpointerdown=e=>{if(busy||!original)return;e.preventDefault();touch();const [x,y]=point(e);if(mode==='add'||mode==='keep'){suggest(x,y,mode==='add');return;}remember();drawing=true;lastPoint=[x,y];$('edit').setPointerCapture(e.pointerId);dab(x,y,mode==='paint');paint();};
$('edit').onpointermove=e=>{if(!drawing)return;const p=point(e),distance=Math.max(Math.abs(p[0]-lastPoint[0]),Math.abs(p[1]-lastPoint[1]));for(let i=0;i<=distance;i++)dab(Math.round(lastPoint[0]+(p[0]-lastPoint[0])*i/Math.max(1,distance)),Math.round(lastPoint[1]+(p[1]-lastPoint[1])*i/Math.max(1,distance)),mode==='paint');lastPoint=p;paint();};
function finishPaint(){if(drawing){drawing=false;changed();status('已手动修改，确认前请检查右侧文字。');}}
$('edit').onpointerup=finishPaint;$('edit').onpointercancel=finishPaint;
const whole=new PixelOverview($('whole'),{
  rows:()=>rows.filter(r=>r.page===wholePage).map(merged),activity:touch,
  select:r=>{$('refineBtn').disabled=!r;$('wholeSelection').textContent=r?`${r.region.length} 个控制点 · ${r.confirmed?'已确认':'待复核'}`:'在完整答题图上检查有没有遗漏。';},
  zoom:scale=>$('wholeZoom').textContent=Math.round(scale*100)+'%',
  open:id=>show(rows.findIndex(r=>r.id===id)),
  reshape:(id,region)=>{
    const r=rows.find(r=>r.id===id);
    try{const next=R.reshape(ensure(r),region,pack.pages[r.page].size);remember();store.rows[id]=next;invalidate(r);save();nav();whole.select(id);status('范围已调整，重叠部分的像素修正按原坐标保留。点“放大精修”处理新增范围。');}
    catch(e){status(e.message,true);}
  },
  create:region=>{
    remember();const box=R.cropBox(region,pack.pages[wholePage].size);
    const r={id:pack.pages[wholePage].id+'#manual-'+Date.now(),page:wholePage,box,region,runs:[],text:''};
    store.extra.push(r);syncRows();invalidate(r);save();show(rows.length-1);
  }
});
function updateGeometryMode(){
  $('geometryBtn').classList.toggle('active',whole.editing);$('geometryBtn').textContent=whole.editing?'完成范围调整':'调整区域';
  $('wholeHint').textContent=whole.editing?'点选一个区域；拖白色顶点改范围，拖边中间的小点增加顶点，Alt / ⌥ 点顶点删除。改完点“放大精修”。':'点击涂改区域放大；漏了就在图上拖一个新框。滚轮缩放，空格或右键拖动平移。';
  whole.draw();
}
async function overview(pi=wholePage,options={}){
  touch();save();cancelPending();wholePage=pi;inDetail=false;
  $('detailPanel').hidden=true;$('overviewPanel').hidden=false;$('overviewBtn').hidden=true;
  $('title').textContent=`第 ${pi+1} 题 · 完整答题图`;const ticket=++serial;
  const im=await imageOf(pi);if(ticket!==serial)return;
  whole.setImage(im,options.selected||null,!!options.editing);
  if(options.focus&&whole.current())whole.focus(whole.current().region);
  updateGeometryMode();nav();lastTime=Date.now();status('检查整题上的涂改是否都被框住，点击一处即可放大。');
}
async function restore(from,to){
  if(!from.length)return;cancelPending();touch();
  to.push({store:clone(store),at,wholePage,inDetail,chosen:whole.selected,editing:whole.editing});
  const previous=from.pop(),latest=store;
  for(const [id,s] of Object.entries(previous.store.rows)){
    const now=latest.rows[id];if(!now)continue;
    s.seconds=Math.max(s.seconds,now.seconds);
    const different=JSON.stringify([s.box,s.region,s.runs,s.not_revision])!==JSON.stringify([now.box,now.region,now.runs,now.not_revision]);
    s.ops=Math.max(s.ops,now.ops)+(different?1:0);
  }
  for(const [pi,seconds] of Object.entries(latest.pageSeconds))previous.store.pageSeconds[pi]=Math.max(previous.store.pageSeconds[pi]||0,seconds);
  store=previous.store;at=previous.at;wholePage=previous.wholePage;syncRows();save();
  if(previous.inDetail)await show(at);else await overview(wholePage,{selected:previous.chosen,editing:previous.editing,focus:previous.editing});
  status('已恢复上一步的范围和像素修正。');
}
for(const id of ['undo','wholeUndo'])$(id).onclick=()=>restore(history,future);
for(const id of ['redo','wholeRedo'])$(id).onclick=()=>restore(future,history);
$('overviewBtn').onclick=()=>overview();$('regionBtn').onclick=()=>overview(row().page,{selected:row().id,editing:true,focus:true});
$('geometryBtn').onclick=()=>{whole.editing=!whole.editing;updateGeometryMode();};
$('refineBtn').onclick=()=>{if(whole.selected)show(rows.findIndex(r=>r.id===whole.selected));};
$('wholeFit').onclick=()=>whole.fit();$('wholeMore').onclick=()=>whole.zoom(1.25);$('wholeLess').onclick=()=>whole.zoom(.8);
$('wholeHide').onclick=()=>{whole.hidden=!whole.hidden;whole.draw();};
$('prevPage').onclick=()=>overview(Math.max(0,wholePage-1));$('nextPage').onclick=()=>overview(Math.min(pack.pages.length-1,wholePage+1));
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===b));});
$('more').onclick=()=>{zoom=Math.min(10,zoom+.5);paint();};$('less').onclick=()=>{zoom=Math.max(.5,zoom-.5);paint();};$('hide').onclick=()=>{hidden=!hidden;paint();};
$('notRevision').onchange=()=>{touch();remember();state().not_revision=$('notRevision').checked;changed();paint();};
$('note').oninput=()=>{touch();state().note=$('note').value;save();};
$('previous').onclick=()=>show(at-1);$('next').onclick=()=>show(at+1);
$('confirm').onclick=()=>{if(busy||!original)return;touch();state().runs=R.encode(mask);state().confirmed=true;save();nav();if(rows[at+1]?.page===row().page)show(at+1);else overview(row().page);};
$('pageDone').onclick=()=>{
  const pi=inDetail?row().page:wholePage;
  if(rows.some(r=>r.page===pi&&!ensure(r).confirmed)){status('这题还有未确认的区域，请逐处复核后再确认整题。',true);return;}
  touch();store.done[pi]=true;save();nav();status(`第 ${pi+1} 题已确认检查完成。可以导出标注。`);
};
function exported(source=store){
  const pages={};pack.pages.forEach((p,pi)=>{
    const selected=[...pack.rows,...source.extra].filter(r=>r.page===pi).map(r=>({...r,...R.restoreRow(r,source.rows[r.id]),page:pi}));
    pages[p.id]={done:!!source.done[pi],overview_seconds:source.pageSeconds?.[pi]||0,seconds:(source.pageSeconds?.[pi]||0)+selected.reduce((s,r)=>s+r.seconds,0),revisions:selected};
  });
  return {schema:'paper2-pixel-review-v1',pack:pack.id,exported_at:new Date().toISOString(),task:'strike_only',image_revision:source.imageRevision||'original',image_source:source.imageRevision===pack.image_revision?pack.image_source:undefined,pages};
}
function download(value,name){const blob=new Blob([JSON.stringify(value)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('exportBtn').onclick=()=>{touch();save();download(exported(),'划线精细标注_'+pack.id+'_'+new Date().toISOString().slice(0,10));status('已导出。未确认区域保留为草稿，不进入人工金标。');};
$('backupBtn').onclick=()=>{const old=JSON.parse(localStorage.getItem(backupKey));if(old)download(exported(old),'更新前标注备份_'+pack.id);};
$('importBtn').onclick=()=>$('import').click();
$('import').onchange=async e=>{
  try{
    const value=JSON.parse(await e.target.files[0].text()),next=R.importData(pack,store,value);
    const removed=await cleanStore(next);remember();cancelPending();store=next;syncRows();save();await overview(0);
    status('标注已导入。'+(removed?`已去掉 ${removed} 个清洁后变白的像素，涉及的区域需要重新确认。`:''));
  }catch(error){status('导入失败，现有标注未变：'+error.message,true);}e.target.value='';
};
document.addEventListener('keydown',e=>{
  if(['INPUT','TEXTAREA','BUTTON'].includes(e.target.tagName))return;const k=e.key.toLowerCase();
  if((e.ctrlKey||e.metaKey)&&k==='z'){e.preventDefault();restore(e.shiftKey?future:history,e.shiftKey?history:future);}
  else if(k==='0'||k==='escape'){e.preventDefault();if(inDetail)overview();else whole.fit();}
  else if(k==='h'){if(inDetail)$('hide').click();else $('wholeHide').click();}
  else if(inDetail&&k==='n')show(at+1);else if(inDetail&&k==='p')show(at-1);
  else if(e.code==='Space'){e.preventDefault();if(inDetail)$('confirm').click();else whole.space=true;}
});
document.addEventListener('keyup',e=>{if(e.code==='Space')whole.space=false;});
window.addEventListener('blur',()=>whole.space=false);
document.addEventListener('visibilitychange',()=>{if(document.hidden){if(pack&&store){touch();save();}lastTime=0;}else lastTime=Date.now();});
window.addEventListener('beforeunload',()=>{if(pack&&store){touch();save();}});
async function start(){
  pack=await (await fetch('pack.json')).json();key='pixel-review-'+pack.id;backupKey=key+'-before-clean';
  const previous=localStorage.getItem(key);store=JSON.parse(previous||'null')||{rows:{},done:{},extra:[]};store.pageSeconds||={};
  if(previous&&store.imageRevision!==(pack.image_revision||'original')&&!localStorage.getItem(backupKey))localStorage.setItem(backupKey,previous);
  syncRows();const removed=await cleanStore(store);save();$('backupBtn').hidden=!localStorage.getItem(backupKey);$('cleanBadge').hidden=!pack.image_revision;
  await overview(0);
  if(removed)status(`已保留旧标注，并去掉 ${removed} 个清洁后变白的像素；涉及区域需要重新确认。更新前备份可随时导出。`);
}
start().catch(e=>status('载入失败：'+e.message,true));
