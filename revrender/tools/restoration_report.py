"""Create a local, self-contained comparison of frozen restoration pilot results."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from ..eval.revision_boxes import held
from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest


def paired_interval(differences):
    d=np.asarray(differences,np.float64)
    require(len(d)>0,'empty paired comparison')
    rng=np.random.default_rng(SEED)
    means=d[rng.integers(0,len(d),(2000,len(d)))].mean(1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def build(args):
    root=args.evaluation;require((root/'COMPLETED').is_file(),'evaluation is not complete')
    out=make_output_dir(args.output_dir)
    rows=json.loads((root/'pack/pack.json').read_text())['rows']
    reads={r['id']:r for r in [json.loads(s) for s in (root/'qwen8b/predictions.jsonl').read_text().splitlines()]}
    require(set(reads)=={r['id'] for r in rows},'missing or extra OCR outputs')
    scores=json.loads((root/'read_qwen8b.json').read_text())['held']
    def score(model):return scores[f'{model}/erased_neutral/line']
    base={r['line_id']:held(r['text'],reads[r['id']]['text']) for r in rows if r['model']=='scan' and r['kind']=='line'}
    table=[];inputs=[root/'read_qwen8b.json',root/'pack/pack.json',root/'qwen8b/predictions.jsonl']
    for path in sorted(root.glob('*/pixels.json')):
        pixels=json.loads(path.read_text());name=pixels['model']
        manifest=json.loads((path.parent/'run_manifest.json').read_text());run=Path(manifest['config']['run'])
        selected=json.loads((run/'selection.json').read_text())
        validation=json.loads((run/f'evaluation_{selected["step"]}.json').read_text())
        inputs.extend([path,path.parent/'run_manifest.json',run/'selection.json',run/f'evaluation_{selected["step"]}.json'])
        differences=[held(r['text'],reads[r['id']]['text'])-base[r['line_id']] for r in rows if r['model']==name and r['kind']=='line']
        table.append({'model':name,'step':selected['step'],'validation':validation,'pixels':pixels,'ocr':score(name),
                      'paired_held_delta':float(np.mean(differences)),'paired_held_delta_ci95':paired_interval(differences)})
    require(len(table)==9,'expected nine matched experiments')
    truncated=sum(r['finish_reason']=='length' for r in reads.values())
    summary={'baseline':score('scan'),'models':table,'truncated_readings':truncated,
             'protocol':'4000-step task adaptation; checkpoint selection on 16 human validation regions; fixed test oracle regions',
             'limitations':['not original-paper benchmark reproduction','test box bootstrap is exploratory; regions from the same writer are correlated',
                            'pixel labels cannot establish whether hidden letters were reconstructed correctly','one seed and short training budget']}
    write_json(out/'summary.json',summary)
    samples={}
    for r in rows:
        key=r['line_id'];s=samples.setdefault(key,{'id':key,'truth':r['text'],'kind':r['kind'],'variants':{}})
        s['variants'][r['model']]={'image':'data:image/png;base64,'+base64.b64encode(Path(r['images']['erased']).read_bytes()).decode(),
                                  'read':reads[r['id']]['text'],'held':held(r['text'],reads[r['id']]['text'])}
    payload=json.dumps({'summary':summary,'samples':list(samples.values())},ensure_ascii=False).replace('<','\\u003c')
    (out/'index.html').write_text(HTML.replace('__DATA__',payload),encoding='utf-8')
    write_manifest(out,config={'evaluation':str(root)},inputs={str(p):digest(p) for p in inputs})


HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>划线恢复基线：首轮对照</title>
<style>body{font:16px/1.65 system-ui;margin:32px auto;padding:0 20px;max-width:1160px;color:#182637;background:#f7f9fc}h1{font-size:27px}small,.note{color:#536477}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:9px;border-bottom:1px solid #dfe6ee;text-align:left}button,select{font:inherit;padding:7px;margin:4px;border:1px solid #acbacc;border-radius:6px;background:white}.card{background:white;border:1px solid #dbe4ef;border-radius:9px;padding:20px;margin:16px 0}.card img{display:block;width:100%;image-rendering:auto;margin:12px 0}.text{white-space:pre-wrap;overflow-wrap:anywhere}.toolbar{position:sticky;top:0;background:#edf2f9;padding:8px;z-index:1}.good{color:#08754a}.bad{color:#a33e2d}</style>
<h1>划线恢复基线：首轮对照</h1>
<p>三种模型 × 三种素材，各 4,000 步。按独立人工验证集选版本后，统一测试。图中范围来自人工框选，属于恢复模块评测。</p>
<p class="note">金标像素：原有人工掩膜素材；预测像素：预测掩膜与墨迹相交并筛选；预测重绘：同批路径重新绘制。三组底稿、位置和划分相同。指标均为百分比。</p>
<div id="table"></div><p id="baseline"></p>
<p class="note">“读回”是被划掉字符的保留率，可能包含多余字符，须与“完全正确”一起看。括号区间是逐区域配对 bootstrap 的探索性区间；同一书写者的区域相关，不能当作严格显著性结论。像素金标只标可分离的划线，不能证明被遮住的笔画已恢复正确。</p>
<h2>逐处对比</h2><div class="toolbar"><select id="model"></select><button id="prev">上一处</button><select id="sample"></select><button id="next">下一处</button><span id="pos"></span></div>
<div class="card"><strong>人工转写</strong><div id="truth" class="text"></div><small id="id"></small></div>
<div class="card"><strong>原图</strong><img id="raw"><div id="rawread" class="text"></div></div>
<div class="card"><strong id="restitle">恢复图</strong><img id="restored"><div id="restread" class="text"></div></div>
<script>const D=__DATA__;const $=id=>document.getElementById(id);const pct=x=>(100*x).toFixed(1);const labels={gold_pixels:'金标像素',predicted_pixels:'预测像素',predicted_traced:'预测重绘'};
const name=m=>{for(const [k,v] of Object.entries(labels))if(m.endsWith('_'+k))return m.slice(0,-k.length-1)+' · '+v;return m};
const models=D.summary.models;const headers=['模型 / 素材','选中步数','验证 F1','测试擦除精度','测试划线召回','测试有效墨迹误擦','字符读回','完全正确','相比原图读回差值（95%）'];
const table=document.createElement('table');const head=table.insertRow();headers.forEach(x=>{const e=document.createElement('th');e.textContent=x;head.append(e)});
models.forEach(m=>{const row=table.insertRow();const ci=m.paired_held_delta_ci95;[name(m.model),m.step,pct(m.validation.human_removal.f1),pct(m.pixels.precision),pct(m.pixels.recall),pct(m.pixels.valid_ink_damage),pct(m.ocr.held),m.ocr.exact==null?'—':pct(m.ocr.exact),pct(m.paired_held_delta)+' ['+pct(ci[0])+', '+pct(ci[1])+']'].forEach(x=>row.insertCell().textContent=x)});$('table').append(table);
$('baseline').textContent='原图：'+D.summary.baseline.revisions+' 处可读划线，字符读回 '+pct(D.summary.baseline.held)+'%，完全正确 '+pct(D.summary.baseline.exact)+'%。达到生成长度上限的读数：'+D.summary.truncated_readings+'。';
models.forEach(m=>$('model').add(new Option(name(m.model),m.model)));D.samples.forEach((s,i)=>$('sample').add(new Option((i+1)+' · '+s.truth.slice(0,45),i)));
function show(){const i=Number($('sample').value),s=D.samples[i],m=$('model').value,a=s.variants.scan,b=s.variants[m];$('truth').textContent=s.truth;$('id').textContent=s.id;$('raw').src=a.image;$('restored').src=b.image;$('rawread').textContent='VLM：'+a.read;$('restread').textContent='VLM：'+b.read;$('restitle').textContent=name(m)+' · 恢复图';$('pos').textContent=(i+1)+' / '+D.samples.length}
$('model').onchange=show;$('sample').onchange=show;$('prev').onclick=()=>{$('sample').selectedIndex=Math.max(0,$('sample').selectedIndex-1);show()};$('next').onclick=()=>{$('sample').selectedIndex=Math.min(D.samples.length-1,$('sample').selectedIndex+1);show()};show();
</script></html>'''


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    build(p.parse_args(argv))


if __name__=='__main__':main()
