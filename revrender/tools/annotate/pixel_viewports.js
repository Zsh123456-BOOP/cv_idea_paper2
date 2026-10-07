/* Both native-pixel views share one pan/zoom position. Independent of editor state. */
'use strict';
class PixelViewports {
  constructor(views,canvases,onzoom){
    this.views=views;this.canvases=canvases;this.onzoom=onzoom;this.scale=1;this.drag=null;
    views.forEach(v=>v.addEventListener('scroll',()=>this.sync(v)));
  }
  sync(source){for(const v of this.views)if(v!==source){if(v.scrollLeft!==source.scrollLeft)v.scrollLeft=source.scrollLeft;if(v.scrollTop!==source.scrollTop)v.scrollTop=source.scrollTop;}}
  zoom(scale,anchor=null){
    const v=this.views[0],x=anchor?.[0]??v.clientWidth/2,y=anchor?.[1]??v.clientHeight/2;
    const nx=(v.scrollLeft+x)/this.scale,ny=(v.scrollTop+y)/this.scale;
    this.scale=Math.max(.5,Math.min(64,scale));
    for(const c of this.canvases){c.style.width=c.width*this.scale+'px';c.style.height=c.height*this.scale+'px';}
    for(const view of this.views){view.scrollLeft=nx*this.scale-x;view.scrollTop=ny*this.scale-y;}
    this.onzoom(this.scale);
  }
  fit(){this.zoom((this.views[0].clientWidth-2)/this.canvases[0].width);for(const v of this.views){v.scrollLeft=0;v.scrollTop=0;}}
  down(e,view){e.preventDefault();view.setPointerCapture(e.pointerId);this.drag={x:e.clientX,y:e.clientY,left:view.scrollLeft,top:view.scrollTop,view};}
  move(e){if(!this.drag)return;const d=this.drag;d.view.scrollLeft=d.left+d.x-e.clientX;d.view.scrollTop=d.top+d.y-e.clientY;this.sync(d.view);}
  up(){this.drag=null;}
}
if(typeof module!=='undefined')module.exports=PixelViewports;
