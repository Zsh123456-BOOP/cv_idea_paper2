'use strict';
const L=window.REVIEW_LIBRARY,S=window.ReviewLibraryStore,$=id=>document.getElementById(id);
let saved=new Map();
const key=p=>'pixel-review-'+p.id+'-offline-review-clean-v1';
const copied=x=>JSON.parse(JSON.stringify(x));
function count(pages){let confirmed=0,total=0;for(const p of Object.values(pages||{}))for(const r of p.revisions||[]){total++;if(r.confirmed)confirmed++;}return {confirmed,total};}
function combined(){const pages=copied(L.initial_pages);for(const x of saved.values())Object.assign(pages,x.export?.pages||{});for(const [id,p] of Object.entries(pages))p.split=L.page_splits[id];return pages;}
function output(training=false){const pages=combined();if(training)for(const [id,p] of Object.entries(pages)){
  if(p.split!=='train'){delete pages[id];continue;}
  p.revisions=(p.revisions||[]).filter(r=>r.confirmed&&!r.deleted&&!r.uncertainMask?.runs?.length);p.deleted_revisions=[];
  if(!p.revisions.length)delete pages[id];
}
return {schema:'paper2-pixel-review-v1',annotation_version:3,pack:L.id,task:'strike_only',exported_at:new Date().toISOString(),image_source:L.image_source,image_revision:'review-clean-v1',selected_checkpoint:L.selected_checkpoint,protected_validation_writers:L.protected_validation_writers,training_only:training,pages};}
function download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=name+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function refresh(){
  saved=new Map();for(const p of L.parts){const v=await S.get(key(p));if(v)saved.set(p.id,v);}
  const c=count(combined()),total=L.parts.reduce((n,p)=>n+p.regions,0);$('summary').textContent=`${Object.keys(L.page_to_part).length} 题 · ${total} 处 · 已确认 ${c.confirmed} 处 · ${L.parts.length} 组分页`;
  const cards=$('cards');cards.replaceChildren();
  for(const p of L.parts){const v=saved.get(p.id),n=v?count(v.export.pages).confirmed:p.confirmed;
    if($('filter').value==='pending'&&n>=p.regions||$('filter').value==='started'&&!v)continue;
    const a=document.createElement('article'),h=document.createElement('strong'),t=document.createElement('p'),bar=document.createElement('progress'),link=document.createElement('a');
    h.textContent=`第 ${L.parts.indexOf(p)+1} 组 · ${p.pages} 题`;t.textContent=`确认 ${n} / ${p.regions} 处 · `+Object.entries(p.splits).map(([s,n])=>`${({train:'训练',val:'验证',test:'测试'})[s]} ${n} 题`).join(' / ');bar.max=p.regions;bar.value=n;link.href=p.file;link.textContent=v?'继续这组 →':'打开这组 →';a.append(h,t,bar,link);cards.append(a);
  }
}
$('resume').onclick=()=>{const p=L.parts.find(p=>{const v=saved.get(p.id);return (v?count(v.export.pages).confirmed:p.confirmed)<p.regions;})||L.parts[0];location.href=p.file;};
$('filter').onchange=refresh;
$('export').onclick=async()=>{await refresh();download(output(),'全量划线标注_'+new Date().toISOString().slice(0,10));};
$('train').onclick=async()=>{await refresh();const value=output(true);download(value,'已确认训练划线_'+new Date().toISOString().slice(0,10));$('status').textContent=`导出 ${count(value.pages).confirmed} 处训练标签，验证、测试和未确认区域已排除。`;};
$('import').onclick=()=>$('file').click();
// Reuse the editor's import validation in an invisible part frame. It retains
// current human work on ID conflicts; imported JSON never becomes executable code.
$('file').onchange=async e=>{
  try{
    const value=JSON.parse(await e.target.files[0].text());if(value.schema!=='paper2-pixel-review-v1')throw Error('不是像素标注 JSON');
    for(const id of Object.keys(value.pages||{}))if(!(id in L.page_to_part))throw Error('包含本库以外的题目：'+id);
    const groups=new Map();for(const [id,page] of Object.entries(value.pages)){
      const n=L.page_to_part[id];if(!groups.has(n))groups.set(n,{});groups.get(n)[id]=page;
    }
    download(output(),'导入前全量备份_'+Date.now());
    for(const [n,pages] of groups){
      $('status').textContent=`正在导入第 ${n+1} 组…`;
      await new Promise((resolve,reject)=>{
        const frame=document.createElement('iframe');frame.hidden=true;const token=crypto.randomUUID();let timer;
        const done=error=>{clearTimeout(timer);window.removeEventListener('message',handler);frame.remove();error?reject(Error(error)):resolve();};
        const handler=ev=>{if(ev.source!==frame.contentWindow||ev.data?.token!==token)return;if(ev.data.type==='review-import-ready')frame.contentWindow.postMessage({type:'review-library-import',token,pages},'*');else if(ev.data.type==='review-import-done')done(ev.data.error);};
        window.addEventListener('message',handler);timer=setTimeout(()=>done('导入超时，已保存部分仍保留，可重试'),120000);
        frame.src=L.parts[n].file+'#import='+token;document.body.append(frame);
      });
    }
    await refresh();$('status').textContent='导入完成；已有人工确认或修改的同编号区域优先保留。';
  }catch(error){$('status').textContent='导入失败：'+error.message;$('status').className='error';}e.target.value='';
};
const table=document.createElement('p');table.textContent=Object.entries(L.validation.models).map(([n,m])=>`${n}${n===L.validation.selected?'（本轮选定）':''}：F1 ${(m.metrics.f1*100).toFixed(2)}%，误擦 ${(m.metrics.valid_ink_erased*100).toFixed(2)}%`).join('；');$('models').append(table);
refresh().catch(e=>{$('status').textContent='无法读取自动保存：'+e.message+'。请保留并导出标注备份。';$('status').className='error';});
