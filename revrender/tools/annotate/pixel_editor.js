'use strict';
const $=id=>document.getElementById(id), R=PixelReview, clone=x=>JSON.parse(JSON.stringify(x));
let pack,rows,key,backupKey,store,at=0,wholePage=0,inDetail=false,mode='add',zoom=3,hidden=false;
let mask,original,grey,allowed,manualKeep,uncertain,keepMarks,history=[],future=[],drawing=false,lastPoint=null,serial=0,busy=false,lastTime=0;
const images=new Map();
let lastAutoCleaned=0;
let backupHandle=null,backupPending=null,backupWriting=false,brushSaveTimer=null,space=false;
const row=()=>rows[at], state=()=>ensure(row());
const selectingRegions=()=>pack?.task==='region_selection';
const keeping=()=>state().annotationMode==='keep';
function ensure(r){return store.rows[r.id]||(store.rows[r.id]=R.restoreRow(r));}
function syncRows(){
  const focused=rows?.[at]?.id,all=[...pack.rows,...store.extra];
  all.forEach(r=>store.rows[r.id]=R.restoreRow(r,store.rows[r.id]));
  rows=all.filter(r=>!store.rows[r.id].deleted&&!store.rows[r.id].duplicate_of);
  if(pack.library)rows.sort((a,b)=>a.page-b.page||ensure(a).box[1]-ensure(b).box[1]||ensure(a).box[0]-ensure(b).box[0]);
  const position=rows.findIndex(r=>r.id===focused);
  at=position>=0?position:Math.min(at,Math.max(0,rows.length-1));
}
function merged(r){return {...r,...ensure(r),page:r.page};}
function save(){
  if(!store)return;
  store.savedAt=Date.now();store.cursor={page:wholePage,id:inDetail?row()?.id:null};
  if(pack.library){
    const stamp=store.savedAt;
    $('saveState').textContent='正在保存…';
    ReviewLibraryStore.put(key,{draft:store,export:exported()}).then(()=>{
      if(store.savedAt===stamp){$('saveState').textContent='全部修改已自动保存 · '+new Date(stamp).toLocaleTimeString();$('saveState').classList.remove('error');}
    }).catch(e=>{$('saveState').textContent='保存失败，请立即导出：'+e.message;$('saveState').classList.add('error');});
  }else{
    // Offline page: localStorage and IndexedDB (read back), so one of them failing is seen and the other still holds the work.
    let local=false;try{localStorage.setItem(key,JSON.stringify(store));local=true;}catch(e){}
    const stamp=store.savedAt,shown=(db)=>{if(store.savedAt!==stamp)return;const where=[db&&'浏览器数据库',local&&'浏览器本地'].filter(Boolean);
      $('saveState').textContent=where.length?'已自动保存 · '+new Date(stamp).toLocaleTimeString()+' · '+where.join(' + '):'保存失败！请立即导出标注或保存可续标副本';
      $('saveState').classList.toggle('error',!where.length);};
    ReviewLibraryStore.put(key,{draft:store}).then(()=>ReviewLibraryStore.get(key)).then(v=>shown(v?.draft?.savedAt===stamp)).catch(()=>shown(false));
  }
  if(backupHandle){backupPending=JSON.stringify(exported());flushBackup();}
}
async function flushBackup(){
  if(backupWriting)return;backupWriting=true;
  try{while(backupPending){const value=backupPending;backupPending=null;const stream=await backupHandle.createWritable();await stream.write(value);await stream.close();} $('saveState').textContent+=' · 文件备份已写入';}
  catch(e){$('saveState').textContent='文件备份失败：'+e.message+'；请导出备份';$('saveState').classList.add('error');backupHandle=null;}
  finally{backupWriting=false;}
}
function saveLabels(){if(keeping())R.savePositive(state(),keepMarks);else{R.saveKeep(state(),manualKeep);R.saveUncertain(state(),uncertain);}}
function scheduleBrushSave(){if(!brushSaveTimer)brushSaveTimer=setTimeout(()=>{brushSaveTimer=null;if(drawing){state().runs=R.encode(mask);saveLabels();save();}},200);}

function touch(){const now=Date.now();if(lastTime&&document.visibilityState==='visible'){const seconds=Math.min((now-lastTime)/1000,60);if(inDetail&&row())state().seconds+=seconds;else store.pageSeconds[wholePage]=(store.pageSeconds[wholePage]||0)+seconds;}lastTime=now;}
function status(text,error=false){$('status').textContent=text;$('status').className='status'+(error?' error':'');}
function invalidate(r){const s=ensure(r);s.confirmed=false;s.ops++;store.done[r.page]=false;}
function cleanActive(){saveLabels();if(!keeping())for(let i=0;i<mask.length;i++)if(!allowed[i]||manualKeep[i]||uncertain[i])mask[i]=0;state().runs=R.encode(mask);const added=R.autoClean(state(),grey);mask=R.decode(state().runs,grey.length);return added;}
function changed(){cleanActive();invalidate(row());save();nav();}
function remember(){history.push({store:clone(store),at,wholePage,inDetail,chosen:whole.selected,editing:whole.editing});if(history.length>50)history.shift();future=[];}
function cancelPending(){finishPaint();serial++;busy=false;drawing=false;$('edit').style.opacity=1;}
async function imageOf(pi){if(images.has(pi))return images.get(pi);const im=new Image();im.src=pack.pages[pi].image;await im.decode();images.set(pi,im);return im;}
function cropData(pi,box){const im=images.get(pi),[l,t,r,b]=box,c=document.createElement('canvas');c.width=r-l;c.height=b-t;const cx=c.getContext('2d');cx.drawImage(im,l,t,r-l,b-t,0,0,r-l,b-t);return cx.getImageData(0,0,r-l,b-t);}
function cropGrey(pi,box){const rgba=cropData(pi,box).data,pixels=new Uint8Array(rgba.length/4);for(let i=0;i<pixels.length;i++)pixels[i]=Math.round((rgba[4*i]+rgba[4*i+1]+rgba[4*i+2])/3);return pixels;}
async function cleanStore(candidate){
  let removed=0;lastAutoCleaned=0;
  for(const base of [...pack.rows,...candidate.extra]){
    if(candidate.rows[base.id]?.duplicate_of)continue;
    let s=candidate.rows[base.id]=R.restoreRow(base,candidate.rows[base.id]);
    R.validBox(s.box,pack.pages[base.page].size);R.validRegion(s.region,pack.pages[base.page].size);
    s=candidate.rows[base.id]=R.clipRegion(s,s.region,R.cropBox(s.region,pack.pages[base.page].size));
    await imageOf(base.page);const rgba=cropData(base.page,s.box).data,clean=R.filterInk(s.runs,rgba);
    if(clean.removed){s.needs_clean_review=s.confirmed||s.needs_clean_review;s.runs=clean.runs;s.confirmed=false;candidate.done[base.page]=false;s.cleaned_pixels=(s.cleaned_pixels||0)+clean.removed;removed+=clean.removed;}
    const pixels=new Uint8Array(rgba.length/4);for(let i=0;i<pixels.length;i++)pixels[i]=Math.round((rgba[4*i]+rgba[4*i+1]+rgba[4*i+2])/3);
    lastAutoCleaned+=R.autoClean(s,pixels);
  }
  candidate.imageRevision=pack.image_revision||'original';return removed;
}
function nav(){
  $('progress').textContent=TextMode.on()?`已完成 ${rows.filter(r=>TextMode.done(ensure(r),r)).length} / ${rows.length} · 像素 ${rows.filter(r=>ensure(r).confirmed).length} · 文字 ${rows.filter(r=>ensure(r).text_confirmed).length}${TextMode.judging()?' · 对错 '+rows.filter(TextMode.judged).length:''}`:selectingRegions()?`已框选 ${rows.length} 处 · ${Object.values(store.done).filter(Boolean).length} 题完成`:`像素 ${rows.filter(r=>ensure(r).confirmed).length} / ${rows.length} · 文字 ${rows.filter(r=>ensure(r).text_confirmed).length} / ${rows.length}`;
  const nav=$('nav');nav.replaceChildren();const filter=TextMode.filterBox();if(filter)nav.append(filter);
  pack.pages.forEach((p,i)=>{
    if(TextMode.filtering()&&!rows.some(r=>r.page===i&&TextMode.matches(r)))return;
    const h=document.createElement('button');h.className='question'+(!inDetail&&i===wholePage?' active':'');
    h.textContent=`第 ${i+1} 题 · 题号 ${p.question}${p.split?' · '+({train:'训练',val:'验证（不训练）',test:'测试（不训练）'}[p.split]||p.split):''}${store.done[i]?' ✓':''}`;h.onclick=()=>overview(i);nav.append(h);
    let number=0;
    rows.forEach((r,k)=>{if(r.page!==i)return;number++;if(!TextMode.matches(r))return;const b=document.createElement('button');b.textContent=`${TextMode.done(ensure(r),r)?'✓':'○'} 划线 ${number}${TextMode.tag(r)}${ensure(r).needs_clean_review?' · 清洁后复核':''}`;if(inDetail&&k===at)b.className='active';b.onclick=()=>show(k);nav.append(b);});
  });
  const deleted=[...pack.rows,...store.extra].filter(r=>ensure(r).deleted&&!ensure(r).duplicate_of);
  if(deleted.length){const group=document.createElement('details'),summary=document.createElement('summary');summary.textContent=`已删除 ${deleted.length} 处（可恢复）`;group.append(summary);
    deleted.forEach(r=>{const b=document.createElement('button');b.textContent=`恢复第 ${r.page+1} 题 · ${r.id.split('#').pop()}`;b.onclick=()=>restoreDeleted(r);group.append(b);});nav.append(group);}
  const pi=inDetail?row()?.page:wholePage;
  $('pageDone').textContent=selectingRegions()?(store.done[pi]?'本题框选完成 ✓':'本题框选完成'):(store.done[pi]?'整题已检查 ✓':'确认整题检查完成');
  $('prevPage').disabled=wholePage===0;$('nextPage').disabled=wholePage===pack.pages.length-1;
}
async function show(index){
  if(!rows[index])return;if(selectingRegions())return overview(rows[index].page,{selected:rows[index].id,editing:true,focus:true});touch();save();cancelPending();at=index;wholePage=row().page;inDetail=true;
  $('overviewPanel').hidden=true;$('detailPanel').hidden=false;$('overviewBtn').hidden=false;
  const ticket=++serial,s=state();original=null;await imageOf(row().page);if(ticket!==serial)return;
  Object.assign(s,R.clipRegion(s,s.region,R.cropBox(s.region,pack.pages[row().page].size)));allowed=R.regionPixels(s.region,s.box);
  original=cropData(row().page,s.box);R.whiteOutside(original.data,allowed);grey=new Uint8Array(original.width*original.height);
  for(let i=0;i<grey.length;i++)grey[i]=Math.round((original.data[4*i]+original.data[4*i+1]+original.data[4*i+2])/3);
  manualKeep=R.keepPixels(s);uncertain=R.uncertainPixels(s);keepMarks=R.positivePixels(s);R.autoClean(s,grey);mask=R.decode(s.runs,grey.length);for(const id of ['edit','result']){$(id).width=original.width;$(id).height=original.height;}
  const n=rows.filter((r,i)=>r.page===row().page&&i<=at).length;
  $('title').textContent=`第 ${wholePage+1} 题 · 划线 ${n} · 多边形裁切 ${original.width} × ${original.height}`;$('notRevision').checked=s.not_revision;$('note').value=s.note;
  $('transcription').value=s.text||'';$('unreadable').checked=!!s.unreadable;$('textConfirmed').checked=!!s.text_confirmed;updateTextState();TextMode.onShow();
  $('previous').disabled=TextMode.step(at,-1)<0;$('next').disabled=TextMode.step(at,1)<0;
  updateModeControls();paint();viewports.fit();nav();lastTime=Date.now();save();status(s.confirmed?'这处已确认。继续修改会取消确认。':'仅显示、编辑多边形内部；框外已遮白。');
}
function updateModeControls(){
  const keep=keeping();
  for(const [id,active] of [['eraseMode',!keep],['keepMode',keep]]){$(id).classList.toggle('active',active);$(id).setAttribute('aria-pressed',String(active));}
  $('detailPanel').dataset.annotationMode=keep?'keep':'erase';
  $('modeHint').textContent=keep?'模式 2：画笔涂绿要保留的数字、文字和符号；橡皮撤掉绿色。框内未涂绿的墨迹全部擦白，下图实时更新。':'模式 1：画笔涂红要去除的划线；橡皮撤掉红色、恢复原图。未标红的墨迹保留，下图实时更新。';
  $('editTitle').textContent=keep?'原图 · 只显示绿色保留标记':'原图 · 只显示红色删除标记';
  for(const tool of ['add','keep'])document.querySelector('[data-mode="'+tool+'"]').hidden=!!pack.standalone||keep;
  if(keep&&['add','keep'].includes(mode))document.querySelector('[data-mode="paint"]').click();
}
function chooseAnnotationMode(choice){
  if(!original||busy||(state().annotationMode||'erase')===choice)return;
  finishPaint();touch();remember();store.rows[row().id]=R.changeMode(state(),choice,grey);invalidate(row());
  manualKeep=R.keepPixels(state());uncertain=R.uncertainPixels(state());keepMarks=R.positivePixels(state());
  R.autoClean(state(),grey);mask=R.decode(state().runs,grey.length);
  document.querySelector('[data-mode="paint"]').click();updateModeControls();paint();save();nav();
  status(`已切换为模式 ${keeping()?2:1}；另一种模式的标记已单独保存。`);
}
function paint(){
  if(!original)return;
  const over=new ImageData(new Uint8ClampedArray(original.data),original.width,original.height),clean=new ImageData(new Uint8ClampedArray(original.data),original.width,original.height);
  const keep=keeping();let marked=0,unknown=0;
  for(let i=0;i<mask.length;i++){
    if(!allowed[i]||grey[i]>=245||state().not_revision)continue;
    const selected=keep?keepMarks[i]:mask[i],color=selected?(keep?[0,155,91]:[230,35,55]):null;
    if(selected)marked++;
    if(!keep&&uncertain[i])unknown++;
    if(mask[i])for(let c=0;c<3;c++)clean.data[4*i+c]=255;
    if(color&&!hidden)for(let c=0;c<3;c++)over.data[4*i+c]=Math.round(over.data[4*i+c]*.38+color[c]*.62);
  }
  $('edit').getContext('2d').putImageData(over,0,0);$('result').getContext('2d').putImageData(clean,0,0);
  $('labelStats').textContent=`模式 ${keep?2:1} · ${keep?'绿色保留':'红色删除'} ${marked} 像素${unknown?' · 旧不确定 '+unknown+' 像素仍保留，可用画笔或橡皮重新标注':''}`;
  $('resultTitle').textContent=keep?'实时结果 · 只保留涂绿的笔画':'实时结果 · 擦除涂红的划线';
}
function point(e){const c=$('edit'),b=c.getBoundingClientRect();return [Math.max(0,Math.min(c.width-1,Math.round((e.clientX-b.left)*c.width/b.width))),Math.max(0,Math.min(c.height-1,Math.round((e.clientY-b.top)*c.height/b.height)))];}
function dab(x,y,tool){
  if(typeof tool==='boolean')tool=tool?'paint':'erase';
  const radius=Number($('radius').value),w=original.width,h=original.height;
  for(let yy=Math.max(0,y-radius);yy<=Math.min(h-1,y+radius);yy++)for(let xx=Math.max(0,x-radius);xx<=Math.min(w-1,x+radius);xx++)if(allowed[yy*w+xx]&&(xx-x)**2+(yy-y)**2<=radius**2){
    const i=yy*w+xx;
    if(keeping()){
      keepMarks[i]=tool==='paint'&&grey[i]<245?1:0;
      mask[i]=grey[i]<245&&!keepMarks[i]?1:0;
    }else{
      mask[i]=tool==='paint'&&grey[i]<245?1:0;manualKeep[i]=tool==='erase'?1:0;uncertain[i]=0;
    }
  }
}
async function suggest(x,y,add){
  if(busy)return;touch();remember();const ticket=++serial;busy=true;$('edit').style.opacity=.65;status('正在按你的提示修正附近笔画…');
  try{
    const response=await fetch('/refine',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({page:row().page,box:state().box,domain_runs:R.encode(allowed),runs:R.encode(mask),clicks:[[x,y,add]],radius:28})});
    const data=await response.json();if(!response.ok)throw Error(data.error||'服务未就绪');if(ticket!==serial)return;
    mask=R.decode(R.filterInk(data.runs,original.data).runs,mask.length);changed();paint();status('已局部修正。检查下方是否伤字；不合适可撤销或用画笔。');
  }catch(e){if(ticket===serial){history.pop();status('模型修正未完成：'+e.message+'。可以继续用画笔和橡皮。',true);}}
  finally{if(ticket===serial){busy=false;$('edit').style.opacity=1;}}
}
const viewports=new PixelViewports([$('editViewport'),$('resultViewport')],[$('edit'),$('result')],scale=>{zoom=scale;$('zoomLabel').textContent=Number(scale.toFixed(1))+'×';});
for(const id of ['editViewport','resultViewport']){
  const view=$(id);view.oncontextmenu=e=>e.preventDefault();
  view.onpointerdown=e=>{
    if(!original||busy)return;
    if(id==='resultViewport'||mode==='pan'||space||e.button===2||e.button===1||e.shiftKey){touch();viewports.down(e,view);return;}
    if(e.target!==$('edit'))return;
    e.preventDefault();touch();const [x,y]=point(e);if(!allowed[y*original.width+x])return;
    if(mode==='add'||mode==='keep'){suggest(x,y,mode==='add');return;}
    remember();invalidate(row());drawing=true;lastPoint=[x,y];view.setPointerCapture(e.pointerId);dab(x,y,mode);paint();scheduleBrushSave();
  };
  view.onpointermove=e=>{
    if(viewports.drag){viewports.move(e);return;}
    if(!drawing)return;
    const p=point(e),distance=Math.max(Math.abs(p[0]-lastPoint[0]),Math.abs(p[1]-lastPoint[1]));
    for(let i=0;i<=distance;i++)dab(Math.round(lastPoint[0]+(p[0]-lastPoint[0])*i/Math.max(1,distance)),Math.round(lastPoint[1]+(p[1]-lastPoint[1])*i/Math.max(1,distance)),mode);
    lastPoint=p;paint();scheduleBrushSave();
  };
  view.onpointerup=view.onpointercancel=()=>{viewports.up();finishPaint();};
  view.addEventListener('wheel',e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();const rect=view.getBoundingClientRect();viewports.zoom(zoom*(e.deltaY<0?1.15:1/1.15),[e.clientX-rect.left,e.clientY-rect.top]);}},{passive:false});
}
function finishPaint(){clearTimeout(brushSaveTimer);brushSaveTimer=null;if(drawing){drawing=false;const added=cleanActive();save();nav();paint();status(`已修改并自动保存${added?'，自动清理 '+added+' 个残留像素':''}。`);}}
const whole=new PixelOverview($('whole'),{
  rows:()=>rows.filter(r=>r.page===wholePage).map(merged),activity:touch,
  select:r=>{$('wholeDelete').disabled=!r;$('refineBtn').disabled=!r;$('wholeSelection').textContent=r?`${r.region.length} 个控制点 · ${r.confirmed?'已确认':'待复核'}`:'在完整答题图上检查有没有遗漏。';},
  zoom:scale=>$('wholeZoom').textContent=Math.round(scale*100)+'%',
  open:id=>show(rows.findIndex(r=>r.id===id)),
  reshape:(id,region)=>{
    const r=rows.find(r=>r.id===id);
    try{const next=R.reshape(ensure(r),region,pack.pages[r.page].size);if(next.annotationMode==='keep')R.syncKeepResult(next,cropGrey(r.page,next.box));remember();store.rows[id]=next;invalidate(r);save();nav();whole.select(id);status('范围已调整，重叠部分的像素修正按原坐标保留。点“放大精修”处理新增范围。');}
    catch(e){status(e.message,true);}
  },
  create:region=>{
    remember();const box=R.cropBox(region,pack.pages[wholePage].size);
    const r={id:pack.pages[wholePage].id+'#manual-'+Date.now(),page:wholePage,box,region,runs:[],text:''};
    store.extra.push(r);syncRows();invalidate(r);save();
    if(selectingRegions()){whole.editing=true;whole.select(r.id);nav();status('已添加框，拖顶点或边中点调整形状；继续拖动可添加下一个框。');}else show(rows.length-1);
  }
});
function updateGeometryMode(){
  $('geometryBtn').classList.toggle('active',whole.editing);$('geometryBtn').textContent=whole.editing?'完成范围调整':'调整区域';
  $('wholeHint').textContent=selectingRegions()?'拖框选中需要去划线的行；点击已有框后拖顶点调整，拖边中点加点，⌥ 点顶点删点。滚轮放大，空格 / 右键平移。框好约 80 处后导出框选。':whole.editing?'点选一个区域；拖白色顶点改范围，拖边中间的小点增加顶点，Alt / ⌥ 点顶点删除。改完点“放大精修”。':'点击涂改区域放大；漏了就在图上拖一个新框。滚轮缩放，空格或右键拖动平移。';
  whole.draw();
}
async function overview(pi=wholePage,options={}){
  touch();save();cancelPending();wholePage=pi;inDetail=false;
  $('detailPanel').hidden=true;$('overviewPanel').hidden=false;$('overviewBtn').hidden=true;
  $('title').textContent=`第 ${pi+1} 题 · 完整答题图`;const ticket=++serial;
  const im=await imageOf(pi);if(ticket!==serial)return;
  whole.setImage(im,options.selected||null,selectingRegions()||!!options.editing);
  if(options.focus&&whole.current())whole.focus(whole.current().region);
  updateGeometryMode();nav();lastTime=Date.now();save();status(selectingRegions()?'本阶段只框选，不需要画像素。导出后交给模型批量预填，再回来精修。':'检查整题上的涂改是否都被框住，点击一处即可放大。');
}
async function restore(from,to){
  if(!from.length)return;cancelPending();touch();
  to.push({store:clone(store),at,wholePage,inDetail,chosen:whole.selected,editing:whole.editing});
  const previous=from.pop(),latest=store;
  for(const [id,s] of Object.entries(previous.store.rows)){
    const now=latest.rows[id];if(!now)continue;
    s.seconds=Math.max(s.seconds,now.seconds);
    const different=JSON.stringify([s.box,s.region,s.runs,s.not_revision,s.manualKeep,s.uncertainMask,s.annotationMode,s.keepMarks])!==JSON.stringify([now.box,now.region,now.runs,now.not_revision,now.manualKeep,now.uncertainMask,now.annotationMode,now.keepMarks]);
    s.ops=Math.max(s.ops,now.ops)+(different?1:0);
    // An explicit undo must survive reload instead of immediately being auto-cleaned again.
    s.autoCleanup=R.cleanupStamp(s);
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
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{finishPaint();mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===b));});
$('eraseMode').onclick=()=>chooseAnnotationMode('erase');$('keepMode').onclick=()=>chooseAnnotationMode('keep');
$('more').onclick=()=>viewports.zoom(zoom*1.4);$('less').onclick=()=>viewports.zoom(zoom/1.4);$('fitDetail').onclick=()=>viewports.fit();$('hide').onclick=()=>{hidden=!hidden;paint();};
$('notRevision').onchange=()=>{finishPaint();touch();remember();state().not_revision=$('notRevision').checked;changed();paint();};
$('note').oninput=()=>{touch();state().note=$('note').value;save();};
function updateTextState(){$('textState').textContent=state().text_confirmed?'已校对 ✓':'未校对';$('transcription').disabled=!!state().unreadable;}
function textChanged(){touch();state().text=$('transcription').value;state().unreadable=$('unreadable').checked;state().text_confirmed=false;$('textConfirmed').checked=false;updateTextState();save();nav();}
$('transcription').oninput=textChanged;$('unreadable').onchange=textChanged;
$('textConfirmed').onchange=()=>{
  if($('textConfirmed').checked&&!state().unreadable&&!state().text?.trim()){$('textConfirmed').checked=false;status('请先填写被划内容，或勾选“整处无法辨认”。',true);return;}
  touch();state().text_confirmed=$('textConfirmed').checked;updateTextState();save();nav();
};
$('previous').onclick=()=>show(TextMode.step(at,-1));$('next').onclick=()=>show(TextMode.step(at,1));
$('confirm').onclick=()=>{if(busy||!original)return;finishPaint();touch();if(!TextMode.confirmText())return;state().runs=R.encode(mask);state().confirmed=true;state().needs_clean_review=false;save();nav();
  if(TextMode.on()){const k=TextMode.step(at,1);if(k>=0)show(k);else status(TextMode.filtering()?'筛选出的这一类已看完。':'全部区域都看完了，记得导出。');}else if(rows[at+1]?.page===row().page)show(at+1);else overview(row().page);};
$('pageDone').onclick=()=>{
  const pi=inDetail?row().page:wholePage;
  if(!selectingRegions()&&rows.some(r=>r.page===pi&&!ensure(r).confirmed)){status('这题还有未确认的区域，请逐处复核后再确认整题。',true);return;}
  touch();store.done[pi]=true;save();nav();status(`第 ${pi+1} 题已确认检查完成。可以导出标注。`);
};
function exported(source=store){
  const pages=R.exportPages(pack,source);
  if(pack.library)for(const p of pack.pages)pages[p.id].split=p.split;
  return {schema:'paper2-pixel-review-v1',annotation_version:3,label_semantics:{runs:'active_mode_erase_result',annotationMode:'erase_or_keep',keepMarks:'mode_2_retained_ink',eraseDraft:'inactive_mode_1_draft',manualKeep:'mode_1_eraser_protection',uncertainMask:'unknown_not_ground_truth',text_confirmed:'independent_text_review'},pack:pack.id,exported_at:new Date().toISOString(),task:pack.task||'strike_only',image_revision:source.imageRevision||'original',image_source:source.imageRevision===pack.image_revision?pack.image_source:undefined,pages};
}
function download(value,name){const blob=new Blob([JSON.stringify(value)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function deleteRegion(id){
  const r=rows.find(r=>r.id===id);if(!r)return;
  finishPaint();touch();remember();const pi=r.page;ensure(r).deleted=true;store.done[pi]=false;
  cancelPending();inDetail=false;syncRows();save();overview(pi).then(()=>status('框已移到左侧“已删除”列表；可恢复，也可撤销。原图和像素修正仍保留。'));
}
function restoreDeleted(r){
  finishPaint();touch();remember();ensure(r).deleted=false;store.done[r.page]=false;
  syncRows();save();overview(r.page,{selected:r.id,editing:true});
}
$('deleteRegion').onclick=()=>deleteRegion(row().id);
$('wholeDelete').onclick=()=>deleteRegion(whole.selected);
$('snapshotBtn').onclick=()=>{
  finishPaint();touch();save();
  const doc=document.documentElement.cloneNode(true);
  doc.querySelector('#review-draft').textContent=JSON.stringify(store).replace(/</g,'\\u003c');
  const blob=new Blob(['<!doctype html>\n'+doc.outerHTML],{type:'text/html;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download='划线标注_续标_'+new Date().toISOString().replace(/[:.]/g,'-')+'.html';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  status('已下载含当前图像和进度的 HTML 副本，直接打开副本即可接着标。');
};
$('bindBackup').onclick=async()=>{
  try{backupHandle=await window.showSaveFilePicker({suggestedName:'划线标注_自动备份.json',types:[{description:'标注 JSON',accept:{'application/json':['.json']}}]});finishPaint();save();}
  catch(e){if(e.name!=='AbortError')status('无法绑定备份文件：'+e.message,true);}
};
$('exportBtn').onclick=()=>{finishPaint();touch();save();download(exported(),'划线精细标注_'+pack.id+'_'+new Date().toISOString().slice(0,10));status(selectingRegions()?'已导出框选 JSON，交给模型批量预填；这些框不作为像素金标。':'已导出。未确认区域保留为草稿，不进入人工金标。');};
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
  if(e.isComposing)return;
  const editingText=['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)||e.target.isContentEditable;
  if(inDetail&&!editingText&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&['1','2'].includes(e.key)){e.preventDefault();chooseAnnotationMode(e.key==='1'?'erase':'keep');return;}
  const brushes={b:'paint',e:'erase'};
  if(inDetail&&!editingText&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&brushes[e.key.toLowerCase()]){e.preventDefault();document.querySelector('[data-mode="'+brushes[e.key.toLowerCase()]+'"]').click();return;}
  if(['INPUT','TEXTAREA','BUTTON','SELECT'].includes(e.target.tagName))return;const k=e.key.toLowerCase();
  if((e.ctrlKey||e.metaKey)&&k==='z'){e.preventDefault();restore(e.shiftKey?future:history,e.shiftKey?history:future);}
  else if(k==='0'||k==='escape'){e.preventDefault();if(inDetail)overview();else whole.fit();}
  else if(k==='h'){if(inDetail)$('hide').click();else $('wholeHide').click();}
  else if(inDetail&&k==='n')show(TextMode.step(at,1));else if(inDetail&&k==='p')show(TextMode.step(at,-1));
  else if(inDetail&&e.key==='Enter'){$('confirm').click();}
  else if(e.code==='Space'){e.preventDefault();space=true;whole.space=true;}
});
document.addEventListener('keyup',e=>{if(e.code==='Space'){whole.space=false;space=false;}});
window.addEventListener('blur',()=>{whole.space=false;space=false;viewports.up();finishPaint();});
document.addEventListener('visibilitychange',()=>{if(document.hidden){if(pack&&store){finishPaint();touch();save();}lastTime=0;}else lastTime=Date.now();});
window.addEventListener('beforeunload',()=>{if(pack&&store){finishPaint();touch();save();}});
async function start(){
  const embedded=$('review-pack');pack=embedded?JSON.parse(embedded.textContent):await (await fetch('pack.json')).json();
  key='pixel-review-'+pack.id+(pack.standalone?'-offline-'+pack.image_revision:'');backupKey=key+'-before-clean';
  let previous=null,warning='';
  try{
    if(pack.library)previous=JSON.stringify((await ReviewLibraryStore.get(key))?.draft||null);
    else{
      // The newer of the offline page's two copies.
      let local=null,db=null;try{local=JSON.parse(localStorage.getItem(key)||'null');}catch(e){}try{db=(await ReviewLibraryStore.get(key))?.draft||null;}catch(e){}
      const newest=[local,db].filter(Boolean).sort((a,b)=>(b.savedAt||0)-(a.savedAt||0))[0];previous=newest?JSON.stringify(newest):null;
    }
  }catch(e){warning='浏览器不允许本地保存，请使用可续标副本。';}
  let cached=null;try{cached=JSON.parse(previous||'null');}catch(e){warning='浏览器中的草稿格式异常，已改用文件内进度。';try{localStorage.setItem(key+'-unreadable-'+Date.now(),previous);}catch(e){}}
  const embeddedDraft=$('review-draft')?JSON.parse($('review-draft').textContent):null;
  store=(embeddedDraft?.savedAt||0)>(cached?.savedAt||0)?embeddedDraft:cached;
  if(!store){store={rows:{},done:{},extra:[],pageSeconds:{}};if(pack.initial_export)store=R.importData(pack,store,pack.initial_export);}
  store.pageSeconds||={};
  if(pack.library&&store.modelRevision!==pack.model_revision){
    // the draft as it was, kept before a new prefill reaches it (a region of it the new pack no longer lists stays recoverable)
    if(cached)await ReviewLibraryStore.put(key+'-before-'+String(pack.model_revision).slice(0,12),{draft:cached,savedAt:Date.now()}).catch(()=>{});
    // A new prefill reaches every region nobody touched; a record the annotator confirmed elsewhere (preserved_manual) arrives as it is.
    const untouched=old=>!old.confirmed&&!old.ops&&!old.deleted&&!old.duplicate_of&&!old.text_confirmed;
    for(const base of pack.rows){const old=store.rows[base.id];
      if(base.preserved_manual){if(!old||untouched(old))store.rows[base.id]=R.restoreRow(base,{...base});}
      else if(old&&untouched(old))store.rows[base.id]=R.restoreRow(base);}
    store.modelRevision=pack.model_revision;
  }else if(pack.model_revision&&store.modelRevision!==pack.model_revision){
    // A struck-text page given a new pixel prefill (struck_text update --runs): only the pixels of a region nobody confirmed, edited or
    // deleted change; its text, marks and notes stay.  The draft as it was is kept first.
    if(cached)await ReviewLibraryStore.put(key+'-before-'+String(pack.model_revision).slice(0,12),{draft:cached,savedAt:Date.now()}).catch(()=>{});
    for(const base of pack.rows){const old=store.rows[base.id];if(old&&!old.confirmed&&!old.ops&&!old.deleted)old.runs=base.runs;}
    store.modelRevision=pack.model_revision;
  }
  if(pack.library&&pack.duplicate_groups?.length){
    const next=clone(store),merged=R.mergeDuplicates(pack,next);
    if(merged){
      // Save the exact pre-migration draft before altering the visible candidates.
      await ReviewLibraryStore.put(key+'-before-dedup-'+Date.now(),{draft:store,export:exported()});
      store=next;
      if(store.cursor?.id&&store.rows[store.cursor.id]?.duplicate_of)store.cursor.id=store.rows[store.cursor.id].duplicate_of;
    }
  }
  if(previous&&cached?.imageRevision!==(pack.image_revision||'original')){try{if(!localStorage.getItem(backupKey))localStorage.setItem(backupKey,previous);}catch(e){}}
  if(pack.library){
    const bar=document.createElement('div');bar.className='toolbar library-bar';
    const go=async url=>{finishPaint();touch();save();try{await ReviewLibraryStore.flush();if(url==='../index.html'&&window.parent!==window)window.parent.postMessage({type:'review-library-home'},'*');else location.href=url;}catch(e){status('保存失败，请先导出当前进度。',true);}};
    const button=(text,url)=>{const b=document.createElement('button');b.textContent=text;b.onclick=()=>go(url);bar.append(b);};
    button('← 全量进度 / 导出所有已筛选结果','../index.html');
    const part=pack.library.part;
    if(part>0)button('上一组题',String(part-1).padStart(3,'0')+'.html');
    if(part+1<pack.library.parts)button('下一组题',String(part+1).padStart(3,'0')+'.html');
    const label=document.createElement('span');label.textContent=`第 ${part+1} / ${pack.library.parts} 组，仅用于分页，没有标注数量限制`;bar.append(label);
    document.querySelector('header').after(bar);$('exportBtn').textContent='导出当前组';
    $('appSubtitle').textContent='全量模型预标：好的按 Enter 确认，需修正的直接改，难的按 N 跳过。';
  }
  const cursor=store.cursor;syncRows();const beforeClean=clone(store),removed=await cleanStore(store);
  if(lastAutoCleaned)history.push({store:beforeClean,at:0,wholePage:cursor?.page||0,inDetail:false,chosen:null,editing:false});
  try{$('backupBtn').hidden=!localStorage.getItem(backupKey);}catch(e){}$('cleanBadge').hidden=!pack.image_revision;
  if(pack.standalone){
    $('snapshotBtn').hidden=false;
    document.querySelectorAll('[data-mode="add"],[data-mode="keep"]').forEach(b=>b.hidden=true);
    document.querySelector('[data-mode="paint"]').click();
    $('detailHint').textContent+=' 模型预填已保留；不需要启动服务。';
  }
  if(selectingRegions()){
    $('appTitle').textContent='划线区域框选';$('appSubtitle').textContent='先框选需要去划线的区域，模型预填后再精修。';
    $('exportBtn').textContent='导出框选';$('geometryBtn').hidden=true;$('refineBtn').hidden=true;
  }
  $('bindBackup').hidden=!window.showSaveFilePicker;
  await TextMode.setup();
  const first=rows.find(r=>!TextMode.done(ensure(r),r)),pi=cursor?.page??first?.page??0;
  // Text mode opens at the first region not yet done (pixels, text and, with drafts, the judgment); otherwise where the annotator left off.
  const target=TextMode.on()&&first?rows.indexOf(first):cursor?.id?rows.findIndex(r=>r.id===cursor.id):-1;
  if(target>=0)await show(target);else await overview(pi);
  const toReview=rows.filter(r=>ensure(r).needs_clean_review).length;
  if(toReview)status(`已载入 ${rows.length} 个区域。${toReview} 处原已确认标注去除了浅灰噪点，请看一眼后重新确认，已有修正仍保留。`);
  if(removed)status(`已去掉 ${removed} 个清洁后变白的像素；涉及区域需要重新确认。`);
  if(lastAutoCleaned)status(`已按固定配置自动清理 ${lastAutoCleaned} 个残留像素并保存；可撤销。`);
  if(warning)status(warning,true);
}
async function libraryImportSetup(){
  if(!pack.library)return;
  $('snapshotBtn').hidden=true;
  $('saveHelp').textContent='本组修改自动保存至 IndexedDB；回全量入口可一次导出所有已查看进度。清理浏览器数据前请导出备份。';
  const token=new URLSearchParams(location.hash.slice(1)).get('import');
  if(!token&&window.parent!==window)window.parent.postMessage({type:'review-library-part',part:pack.library.part},'*');
  if(!token||window.parent===window)return;
  const overlap=(a,b)=>Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]))/Math.max(1,Math.min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])));
  window.addEventListener('message',async ev=>{
    if(ev.source!==window.parent||ev.data?.type!=='review-library-import'||ev.data.token!==token)return;
    try{
      const value={schema:'paper2-pixel-review-v1',pack:pack.id,pages:clone(ev.data.pages)};
      for(const q of Object.values(value.pages))for(const list of ['revisions','deleted_revisions'])q[list]=(q[list]||[]).filter(r=>{
        const old=store.rows[r.id];return !(old&&(old.confirmed||old.ops>0||old.deleted));
      });
      const next=R.importData(pack,store,value);
      for(const [page,q] of Object.entries(value.pages)){
        const pi=pack.pages.findIndex(p=>p.id===page),human=[...(q.revisions||[]),...(q.deleted_revisions||[])].filter(r=>r.confirmed||r.ops>0||r.deleted);
        for(const base of pack.rows.filter(r=>r.page===pi&&r.detector_index!==undefined)){
          const state=next.rows[base.id];
          if(!state.confirmed&&!state.ops&&!state.deleted&&human.some(r=>r.id!==base.id&&overlap(r.box,state.box)>=.5)){state.deleted=true;state.superseded_by_manual=true;}
        }
      }
      await cleanStore(next);store=next;syncRows();save();await ReviewLibraryStore.flush();
      window.parent.postMessage({type:'review-import-done',token},'*');
    }catch(e){window.parent.postMessage({type:'review-import-done',token,error:e.message},'*');}
  });
  window.parent.postMessage({type:'review-import-ready',token},'*');
}
start().then(libraryImportSetup).catch(e=>status('载入失败：'+e.message,true));
