/* Native-pixel state rules shared by the editor and regression tests. */
'use strict';
(function(root, factory) {
  if (typeof module !== 'undefined') module.exports = factory(require('./page_rules.js'));
  else root.PixelReview = factory({bounds, polygonArea, pointInPolygon});
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
    return validBox([l,t,r,b],size);
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
    return {...clipRegion(state,region,box), confirmed:false};
  }
  function regionPixels(region,box){
    const [l,t,r,b]=box,width=r-l,out=new Uint8Array(width*(b-t));
    for(let y=t;y<b;y++)for(let x=l;x<r;x++)out[(y-t)*width+x-l]=G.pointInPolygon([x+.5,y+.5],region)?1:0;
    return out;
  }
  function clipRegion(state,region=state.region,box=state.box){
    // Park excluded edits at native coordinates so expanding a polygon can recover them.
    const boxes=[state.box,box,...(state.outsideMask?[state.outsideMask.box]:[])];
    const all=[Math.min(...boxes.map(b=>b[0])),Math.min(...boxes.map(b=>b[1])),Math.max(...boxes.map(b=>b[2])),Math.max(...boxes.map(b=>b[3]))];
    const width=all[2]-all[0],size=width*(all[3]-all[1]),combined=decode(remap(state.runs,state.box,all),size);
    if(state.outsideMask){const parked=decode(remap(state.outsideMask.runs,state.outsideMask.box,all),size);for(let i=0;i<size;i++)combined[i]|=parked[i];}
    const allowed=regionPixels(region,box),active=new Uint8Array(allowed.length),cropWidth=box[2]-box[0];
    for(let y=box[1];y<box[3];y++)for(let x=box[0];x<box[2];x++){
      const i=(y-box[1])*cropWidth+x-box[0],j=(y-all[1])*width+x-all[0];
      if(allowed[i]){active[i]=combined[j];combined[j]=0;}
    }
    return {...state,region,box,runs:encode(active),outsideMask:{box:all,runs:encode(combined)}};
  }
  function keepPixels(state){
    const size=(state.box[2]-state.box[0])*(state.box[3]-state.box[1]);
    return state.manualKeep?decode(remap(state.manualKeep.runs,state.manualKeep.box,state.box),size):new Uint8Array(size);
  }
  function saveKeep(state,active){
    // Keep native coordinates outside a temporarily shrunken crop as well.
    const old=state.manualKeep,box=old?[Math.min(old.box[0],state.box[0]),Math.min(old.box[1],state.box[1]),Math.max(old.box[2],state.box[2]),Math.max(old.box[3],state.box[3])]:state.box.slice();
    const width=box[2]-box[0],size=width*(box[3]-box[1]),all=old?decode(remap(old.runs,old.box,box),size):new Uint8Array(size);
    const [l,t,r,b]=state.box;
    for(let y=t;y<b;y++)for(let x=l;x<r;x++)all[(y-box[1])*width+x-box[0]]=active[(y-t)*(r-l)+x-l];
    state.manualKeep={box,runs:encode(all)};
  }
  function whiteOutside(rgba,allowed){
    for(let i=0;i<allowed.length;i++)if(!allowed[i])for(let c=0;c<3;c++)rgba[i*4+c]=255;
    return rgba;
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
  const AUTO_CLEANUP=Object.freeze({threshold:100,radius:4,maxArea:12,gray:true,dots:true});
  const CLEANUP_POLICY='residue-100-4-12-v1';
  function cleanupStamp(state){return {policy:CLEANUP_POLICY,box:state.box.slice(),region:state.region?.map(p=>p.slice()),runs:state.runs.map(r=>r.slice())};}
  function autoClean(state,grey){
    if(state.deleted||state.not_revision)return 0;
    const stamp=cleanupStamp(state);
    if(state.autoCleanup?.policy===stamp.policy&&JSON.stringify(state.autoCleanup.box)===JSON.stringify(stamp.box)&&JSON.stringify(state.autoCleanup.region)===JSON.stringify(stamp.region)&&JSON.stringify(state.autoCleanup.runs)===JSON.stringify(stamp.runs))return 0;
    const width=state.box[2]-state.box[0],mask=decode(state.runs,grey.length);
    const allowed=state.region?regionPixels(state.region,state.box):new Uint8Array(grey.length).fill(1),cleanGrey=grey.slice();
    for(let i=0;i<mask.length;i++)if(!allowed[i]){mask[i]=0;cleanGrey[i]=255;}
    const keep=keepPixels(state);for(let i=0;i<mask.length;i++)if(keep[i])mask[i]=0;
    const candidate=residueCandidates(cleanGrey,mask,width,AUTO_CLEANUP);let added=0;
    for(let i=0;i<mask.length;i++)if(candidate.added[i]&&allowed[i]&&!keep[i]){mask[i]=1;added++;}
    state.runs=encode(mask);state.autoCleanup=cleanupStamp(state);
    state.autoCleanedPixels=(state.autoCleanedPixels||0)+added;
    return added;
  }
  function residueCandidates(grey, mask, width, options={}) {
    // Suggestions only: never mutate the human mask or the scan.
    const {threshold,radius,maxArea,gray,dots}={...AUTO_CLEANUP,...options};
    const size=grey.length,height=size/width,added=new Uint8Array(size);
    if(mask.length!==size||!Number.isInteger(height))throw Error('图像尺寸不匹配');
    if(threshold<0||threshold>245||radius<1||radius>4||maxArea<1||maxArea>12)throw Error('清理参数超出范围');
    function dilate(source,distance){
      const out=new Uint8Array(size);
      for(let i=0;i<size;i++)if(source[i]){
        const x=i%width,y=Math.floor(i/width);
        for(let yy=Math.max(0,y-distance);yy<=Math.min(height-1,y+distance);yy++)
          for(let xx=Math.max(0,x-distance);xx<=Math.min(width-1,x+distance);xx++)out[yy*width+xx]=1;
      }return out;
    }
    const near=dilate(mask,radius),touch=dilate(mask,1),dark=new Uint8Array(size),labels=new Int32Array(size),queue=new Int32Array(size);
    for(let i=0;i<size;i++)dark[i]=!mask[i]&&grey[i]<threshold?1:0;
    const protectedInk=dilate(dark,2);let countGray=0,countDots=0,groups=0,label=0;
    for(let start=0;start<size;start++){
      if(labels[start]||mask[start]||grey[start]>=245)continue;
      label++;let head=0,tail=1,border=false,allNear=true,wasConnected=false;
      queue[0]=start;labels[start]=label;
      while(head<tail){
        const i=queue[head++],x=i%width,y=Math.floor(i/width);
        border ||= x===0||y===0||x===width-1||y===height-1;
        allNear &&= !!near[i];wasConnected ||= !!touch[i];
        for(let yy=Math.max(0,y-1);yy<=Math.min(height-1,y+1);yy++)for(let xx=Math.max(0,x-1);xx<=Math.min(width-1,x+1);xx++){
          const k=yy*width+xx;if(!labels[k]&&!mask[k]&&grey[k]<245){labels[k]=label;queue[tail++]=k;}
        }
      }
      // Crop boundaries may truncate real characters. Never suggest cleaning those components.
      if(border)continue;
      let isolated=dots&&tail<=maxArea&&allNear&&wasConnected;
      if(isolated)for(let j=0;j<tail;j++){
        const i=queue[j],x=i%width,y=Math.floor(i/width);
        for(let yy=Math.max(0,y-3);yy<=Math.min(height-1,y+3);yy++)for(let xx=Math.max(0,x-3);xx<=Math.min(width-1,x+3);xx++){
          const k=yy*width+xx;if(!mask[k]&&grey[k]<245&&labels[k]!==label)isolated=false;
        }
      }
      if(isolated){groups++;for(let j=0;j<tail;j++){added[queue[j]]=1;countDots++;}}
      else if(gray)for(let j=0;j<tail;j++){
        const i=queue[j];if(near[i]&&grey[i]>=threshold&&!protectedInk[i]){added[i]=1;countGray++;}
      }
    }
    return {added,gray:countGray,dots:countDots,groups,total:countGray+countDots};
  }
  return {decode,encode,validBox,validRegion,cropBox,remap,reshape,filterInk,restoreRow,importData,exportPages,residueCandidates,AUTO_CLEANUP,cleanupStamp,autoClean,regionPixels,clipRegion,whiteOutside,keepPixels,saveKeep};
});
