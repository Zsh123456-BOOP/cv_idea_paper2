/* Whole-question navigation and polygon handles; hit testing reuses Claude's page_rules. */
'use strict';
class PixelOverview {
  constructor(canvas, callbacks) {
    this.canvas=canvas; this.cb=callbacks; this.scale=1; this.x=0; this.y=0;
    this.selected=null; this.editing=false; this.hidden=false; this.drag=null; this.space=false;
    canvas.addEventListener('contextmenu',e=>e.preventDefault());
    canvas.addEventListener('pointerdown',e=>this.down(e));
    canvas.addEventListener('pointermove',e=>this.move(e));
    canvas.addEventListener('pointerup',e=>this.up(e));
    canvas.addEventListener('pointercancel',()=>{this.drag=null;this.draw();});
    canvas.addEventListener('wheel',e=>{
      e.preventDefault();this.cb.activity();const [x,y]=this.screen(e);this.zoom(Math.exp(-e.deltaY*.0015),x,y);
    },{passive:false});
    new ResizeObserver(()=>{if(this.image && this.canvas.clientWidth) this.resize(false);}).observe(canvas.parentElement);
  }
  rows(){return this.cb.rows();}
  current(){return this.rows().find(r=>r.id===this.selected);}
  screen(e){const r=this.canvas.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top];}
  point(e){const [x,y]=this.screen(e);return [Math.max(0,Math.min(this.image.width,Math.round((x-this.x)/this.scale))),Math.max(0,Math.min(this.image.height,Math.round((y-this.y)/this.scale)))];}
  setImage(image, selected=null, editing=false){this.image=image;this.selected=selected;this.editing=editing;this.drag=null;this.resize(true);this.cb.select(this.current());}
  resize(fit){this.canvas.width=this.canvas.clientWidth;this.canvas.height=this.canvas.clientHeight;if(fit)this.fit();else this.draw();}
  fit(){if(!this.image)return;this.scale=Math.min(this.canvas.width/this.image.width,this.canvas.height/this.image.height)*.96;this.x=(this.canvas.width-this.image.width*this.scale)/2;this.y=(this.canvas.height-this.image.height*this.scale)/2;this.draw();}
  focus(region){const [l,t,r,b]=bounds(region),w=Math.max(220,(r-l)*3),h=Math.max(140,(b-t)*4);this.scale=Math.min(4,this.canvas.width/w,this.canvas.height/h);this.x=this.canvas.width/2-(l+r)/2*this.scale;this.y=this.canvas.height/2-(t+b)/2*this.scale;this.draw();}
  zoom(factor,x=this.canvas.width/2,y=this.canvas.height/2){const nx=(x-this.x)/this.scale,ny=(y-this.y)/this.scale;this.scale=Math.min(16,Math.max(.08,this.scale*factor));this.x=x-nx*this.scale;this.y=y-ny*this.scale;this.draw();}
  select(id){this.selected=id;this.cb.select(this.current());this.draw();}
  handleAt(point){
    const r=this.current();if(!this.editing||this.hidden||!r)return null;
    for(let i=0;i<r.region.length;i++) if(Math.hypot(r.region[i][0]-point[0],r.region[i][1]-point[1])*this.scale<=8)return {index:i,midpoint:false};
    for(let i=0;i<r.region.length;i++){const a=r.region[i],b=r.region[(i+1)%r.region.length],m=[(a[0]+b[0])/2,(a[1]+b[1])/2];if(Math.hypot(m[0]-point[0],m[1]-point[1])*this.scale<=7)return {index:i+1,midpoint:true};}
    return null;
  }
  down(e){
    if(!this.image)return;e.preventDefault();this.cb.activity();this.canvas.focus();
    this.canvas.setPointerCapture(e.pointerId);const p=this.point(e),screen=this.screen(e);
    if(e.button===1||e.button===2||e.shiftKey||this.space){this.drag={kind:'pan',screen,x:this.x,y:this.y};return;}
    const handle=this.handleAt(p);
    if(handle){
      const region=this.current().region.map(p=>p.slice());
      if(e.altKey&&!handle.midpoint){if(region.length>3){region.splice(handle.index,1);this.cb.reshape(this.selected,region);}return;}
      if(handle.midpoint)region.splice(handle.index,0,p);
      this.drag={kind:'handle',index:handle.index,region,screen,midpoint:handle.midpoint,moved:false};return;
    }
    this.drag={kind:'draw',from:p,to:p,screen,moved:false};
  }
  move(e){
    const d=this.drag;if(!d)return;const screen=this.screen(e);
    if(d.kind==='pan'){this.x=d.x+screen[0]-d.screen[0];this.y=d.y+screen[1]-d.screen[1];this.draw();return;}
    if(Math.hypot(screen[0]-d.screen[0],screen[1]-d.screen[1])>3)d.moved=true;
    if(d.kind==='handle')d.region[d.index]=this.point(e);else d.to=this.point(e);
    this.draw();
  }
  up(){
    const d=this.drag;this.drag=null;if(!d)return;
    if(d.kind==='handle'){if(d.moved||d.midpoint)this.cb.reshape(this.selected,d.region);this.draw();return;}
    if(d.kind==='pan')return;
    if(!d.moved){
      const hit=this.hidden?null:revisionAt({revisions:this.rows()},d.from,7/this.scale);
      this.select(hit?.id||null);if(hit&&!this.editing)this.cb.open(hit.id);return;
    }
    const [x0,y0]=d.from,[x1,y1]=d.to,l=Math.min(x0,x1),t=Math.min(y0,y1),r=Math.max(x0,x1),b=Math.max(y0,y1);
    if(r-l>=5&&b-t>=5)this.cb.create([[l,t],[r,t],[r,b],[l,b]]);this.draw();
  }
  draw(){
    if(!this.image)return;const c=this.canvas,cx=c.getContext('2d');
    cx.setTransform(1,0,0,1,0,0);cx.clearRect(0,0,c.width,c.height);cx.fillStyle='#e6e9ee';cx.fillRect(0,0,c.width,c.height);
    cx.setTransform(this.scale,0,0,this.scale,this.x,this.y);cx.drawImage(this.image,0,0);
    const path=points=>{cx.beginPath();points.forEach((p,i)=>i?cx.lineTo(...p):cx.moveTo(...p));cx.closePath();};
    if(!this.hidden)this.rows().forEach((r,i)=>{
      const selected=r.id===this.selected,region=selected&&this.drag?.kind==='handle'?this.drag.region:r.region;
      const colour=this.cb.colour?this.cb.colour(r,selected):selected?'#235cca':r.not_revision?'#89939d':r.confirmed?'#137660':'#d34a3d';
      path(region);cx.lineWidth=(selected?2.5:1.5)/this.scale;cx.strokeStyle=colour;cx.stroke();
      const [l,t]=bounds(region);cx.font=`${12/this.scale}px sans-serif`;cx.fillStyle=colour;cx.fillRect(l,t-18/this.scale,23/this.scale,18/this.scale);cx.fillStyle='white';cx.fillText(String(i+1),l+5/this.scale,t-4/this.scale);
      if(selected&&this.editing)region.forEach((p,j)=>{
        const q=region[(j+1)%region.length],mid=[(p[0]+q[0])/2,(p[1]+q[1])/2];
        for(const [dot,radius] of [[p,5],[mid,3]]){cx.beginPath();cx.arc(...dot,radius/this.scale,0,Math.PI*2);cx.fillStyle='white';cx.fill();cx.strokeStyle=colour;cx.stroke();}
      });
    });
    if(this.cb.overlay)this.cb.overlay(cx,this.scale);       // drawn in image coordinates over the polygons (the line page's regions)
    if(this.drag?.kind==='draw'&&this.drag.moved){cx.strokeStyle='#235cca';cx.lineWidth=2/this.scale;const a=this.drag.from,b=this.drag.to;cx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);}
    cx.setTransform(1,0,0,1,0,0);this.cb.zoom(this.scale);
  }
}
