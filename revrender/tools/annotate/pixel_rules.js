/* Native-pixel state rules shared by the editor and regression tests. */
'use strict';
(function(root, factory) {
  if (typeof module !== 'undefined') module.exports = factory(require('./page_rules.js'));
  else root.PixelReview = factory({bounds, polygonArea});
})(typeof window !== 'undefined' ? window : this, function(G) {
  function decode(runs, size) {
    const a = new Uint8Array(size); let end = 0;
    for (const [s, n] of runs) {
      if (!Number.isInteger(s) || !Number.isInteger(n) || s < end || n <= 0 || s+n > size) throw Error('掩膜数据不合法');
      a.fill(1, s, s+n); end = s+n;
    }
    return a;
  }
  function encode(a) {
    const out = []; let start = -1;
    for (let i = 0; i <= a.length; i++) {
      if (a[i] && start < 0) start = i;
      if (!a[i] && start >= 0) { out.push([start, i-start]); start = -1; }
    }
    return out;
  }
  function validBox(box, size) {
    if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isInteger)) throw Error('区域坐标不合法');
    const [l,t,r,b] = box, [w,h] = size;
    if (l<0 || t<0 || r>w || b>h || r<=l || b<=t) throw Error('区域超出原图');
    return box;
  }
  function validRegion(region, size) {
    if (!Array.isArray(region) || region.length<3 || region.length>200 || region.some(p => !Array.isArray(p) || p.length!==2 ||
        p.some((v,i) => !Number.isFinite(v) || v<0 || v>size[i])) || G.polygonArea(region)<1) throw Error('区域至少需要三个有效顶点');
    return region.map(p => p.map(Math.round));
  }
  function cropBox(region, size) {
    const [l,t,r,b] = G.bounds(validRegion(region,size));
    return validBox([Math.max(0,l-6),Math.max(0,t-6),Math.min(size[0],r+6),Math.min(size[1],b+6)],size);
  }
  function remap(runs, from, to) {
    const oldWidth=from[2]-from[0], width=to[2]-to[0], next=new Uint8Array(width*(to[3]-to[1]));
    const old=decode(runs,oldWidth*(from[3]-from[1]));
    for (let y=Math.max(from[1],to[1]); y<Math.min(from[3],to[3]); y++) {
      for (let x=Math.max(from[0],to[0]); x<Math.min(from[2],to[2]); x++) next[(y-to[1])*width+x-to[0]]=old[(y-from[1])*oldWidth+x-from[0]];
    }
    return encode(next);
  }
  function reshape(state, region, size) {
    region=validRegion(region,size); const box=cropBox(region,size);
    return {...state, region, box, runs:remap(state.runs,state.box,box), confirmed:false};
  }
  function filterInk(runs, rgba) {
    const mask=decode(runs,rgba.length/4); let removed=0;
    for (let i=0;i<mask.length;i++) if(mask[i] && Math.round((rgba[4*i]+rgba[4*i+1]+rgba[4*i+2])/3)>=245) { mask[i]=0;removed++; }
    return {runs:encode(mask), removed};
  }
  function restoreRow(base, saved) {
    return {runs:base.runs,box:base.box,region:base.region,confirmed:false,not_revision:false,deleted:false,ops:0,seconds:0,note:'',...saved};
  }
  function importData(pack, current, value) {
    if(value.schema!=='paper2-pixel-review-v1'||value.pack!==pack.id)throw Error('文件不属于这批题');
    const next=JSON.parse(JSON.stringify(current));const extra=new Map(next.extra.map(r=>[r.id,r]));const seen=new Set();
    for(const [id,page] of Object.entries(value.pages)) {
      const pi=pack.pages.findIndex(q=>q.id===id);if(pi<0)throw Error('题目不匹配');
      const records=[...(page.revisions||[]).map(r=>({...r,deleted:false})),...(page.deleted_revisions||[]).map(r=>({...r,deleted:true}))];
      for(const r of records) {
        if(seen.has(r.id))throw Error('重复的区域编号');seen.add(r.id);
        const base=pack.rows.find(b=>b.id===r.id)||extra.get(r.id);
        if(base&&base.page!==pi)throw Error('区域与题目不匹配');
        validBox(r.box,pack.pages[pi].size);validRegion(r.region,pack.pages[pi].size);
        decode(r.runs,(r.box[2]-r.box[0])*(r.box[3]-r.box[1]));
        if(!pack.rows.some(b=>b.id===r.id))extra.set(r.id,{...r,page:pi});
        next.rows[r.id]=restoreRow(r,{...r,page:pi});
      }
      next.done[pi]=!!page.done;next.pageSeconds[pi]=page.overview_seconds||0;
    }
    next.extra=[...extra.values()];
    for(let pi=0;pi<pack.pages.length;pi++)if([...pack.rows,...next.extra].some(r=>r.page===pi&&!next.rows[r.id]?.deleted&&!next.rows[r.id]?.confirmed))next.done[pi]=false;
    return next;
  }
  function exportPages(pack, source) {
    const pages={};pack.pages.forEach((p,pi)=>{
      const selected=[...pack.rows,...source.extra].filter(r=>r.page===pi).map(r=>({...r,...restoreRow(r,source.rows[r.id]),page:pi}));
      pages[p.id]={done:!!source.done[pi],overview_seconds:source.pageSeconds?.[pi]||0,
        seconds:(source.pageSeconds?.[pi]||0)+selected.reduce((s,r)=>s+r.seconds,0),
        revisions:selected.filter(r=>!r.deleted),deleted_revisions:selected.filter(r=>r.deleted)};
    });return pages;
  }
  return {decode,encode,validBox,validRegion,cropBox,remap,reshape,filterInk,restoreRow,importData,exportPages};
});
