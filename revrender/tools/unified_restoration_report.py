"""Audit complete paired outputs and publish a self-contained comparison."""
import argparse
import base64
import json
from pathlib import Path

from ..eval.unified_restoration import verify_files
from ..util import digest, make_output_dir, require, write_json, write_manifest


def audit(rows, readings, specs):
    ids = [r['id'] for r in rows]; read_ids = [r['id'] for r in readings]
    require(len(set(ids))==len(ids) and len(set(read_ids))==len(read_ids), 'duplicate request or reading')
    require(set(ids)==set(read_ids), 'missing or extra OCR outputs')
    require(all(r['condition']=='erased_neutral' for r in readings), 'mixed OCR prompts')
    for split, expected in [('val',41),('test',81)]:
        populations = []
        for name in ['scan', *[s['name'] for s in specs]]:
            group = [r for r in rows if r['split']==split and r['model']==split+'__'+name and r['kind']=='line']
            require(len(group)==expected, 'incorrect OCR denominator')
            populations.append({r['line_id']:(r['text'],tuple(r['size'])) for r in group})
        require(all(p==populations[0] for p in populations), 'different OCR examples')


def build(args):
    root = args.evaluation; inputs = root/'inputs'
    verify_files(json.loads((inputs/'prepared_sha256.json').read_text()))
    specs = json.loads((inputs/'models.json').read_text())
    rows = json.loads((root/'pack/pack.json').read_text())['rows']
    readings = [json.loads(s) for s in (root/'qwen8b/predictions.jsonl').read_text().splitlines()]
    audit(rows, readings, specs); lookup = {r['id']:r for r in readings}
    scores = json.loads((root/'read_qwen8b.json').read_text())['held']
    table = []
    for spec in specs:
        pixels = json.loads((root/'runs'/spec['name']/'pixels.json').read_text())
        verify_files(spec['sha256'])
        table.append(dict(name=spec['name'],spec=spec,pixels=pixels['splits'],
                          ocr={s:scores[s+'__'+spec['name']+'/erased_neutral/line'] for s in ('val','test')}))
    for split in ('val','test'):
        denominators = {(r['pixels'][split]['tp']+r['pixels'][split]['fn'],r['pixels'][split]['valid_ink']) for r in table}
        require(len(denominators)==1, 'different pixel scoring domains')
    summary = dict(models=table,scan={s:scores[s+'__scan/erased_neutral/line'] for s in ('val','test')},
                   counts=json.loads((inputs/'counts.json').read_text()),protocol=json.loads((inputs/'protocol.json').read_text()),
                   ocr_run=json.loads((root/'qwen8b/run_manifest.json').read_text()),
                   truncated_readings=sum(r.get('finish_reason')=='length' for r in readings),
                   limitations=['same evaluation, unequal training data and compute; not an architecture-only comparison',
                                'oracle region crops, not end-to-end whole-page detection',
                                '16 validation regions, 7 pixel test pages; limited independent writers',
                                'character readback can include extra characters; strict exact match also reported',
                                'human removal labels do not certify hallucinated or fully occluded strokes',
                                'fixed test set has been inspected historically; this is not a new unseen holdout'])
    samples = {}
    for r in rows:
        key = r['split']+'|'+r['line_id']
        sample = samples.setdefault(key,dict(id=r['line_id'],split=r['split'],truth=r['text'],kind=r['kind'],variants={}))
        sample['variants'][r['model'].split('__',1)[1]] = dict(read=lookup[r['id']]['text'],
            image='data:image/png;base64,'+base64.b64encode(Path(r['images']['erased']).read_bytes()).decode())
    out = make_output_dir(args.output_dir); write_json(out/'summary.json',summary)
    payload = json.dumps(dict(summary=summary,samples=list(samples.values())),ensure_ascii=False).replace('<','\\u003c')
    (out/'index.html').write_text(HTML.replace('__DATA__',payload).replace('__COUNT__',str(len(specs))),encoding='utf-8')
    lines = ['# 统一验证与测试评测','',
             f'权重在推理前冻结；原图与{len(specs)}个模型在同一次 Qwen3-VL-8B（TP2）中识别。无重新训练、无按测试成绩选版。',
             '','像素统一：原图灰度 <200；亮度增加 >0.2 计为擦除；人工目标同样限制到该墨迹范围。测试多边形重叠按页去重。',
             '','我们的旧模型是人工微调 step800，本轮是上一轮验证选出的 step400。七个 baseline 均为任务适配后选定版本。',
             '','**评测相同，训练数据不相同。** 人工微调版含真实精修监督；七基线使用同一批合成训练对。不能据此独立证明架构优劣。']
    if any(s['name']=='Ours-synthetic' for s in specs):
        lines += ['', '**Ours-synthetic 是原始框内 step8000：** 从 ImageNet 骨干开始，用47,202条合成行训练，再用70,789个合成框训练；没有用真实精修区域训练。step8000历史上由真实OCR验证集选择，素材也使用第一篇金标，因此不是完全无人工标注。与七baseline的合成数据量及训练预算仍不同。',
                  '', '旧九模型的恢复图及像素结果经过输入、权重和文件哈希核验后原样复用；本轮重新识别全部模型，未拼接历史OCR分数。']
    for split,title in [('val','验证集：16 处像素／41 处可读划线'),('test','测试集：7 题 28 处像素／81 处可读划线')]:
        lines += ['', '## '+title,'','|模型|擦除精度%|划线召回%|像素F1%|原字误擦%|字符读回%|严格全对|','|---|---:|---:|---:|---:|---:|---:|']
        base=summary['scan'][split]
        lines.append(f'|原图|—|0|0|0|{100*base["held"]:.2f}|{round(base["exact"]*base["exact_revisions"])}/{base["exact_revisions"]}|')
        for row in sorted(table,key=lambda r:-r['ocr'][split]['held']):
            p=row['pixels'][split];o=row['ocr'][split]
            lines.append('|'+row['name']+'|'+ '|'.join(f'{100*p[k]:.2f}' for k in ('precision','recall','f1','valid_ink_damage'))+f'|{100*o["held"]:.2f}|{round(o["exact"]*o["exact_revisions"])}/{o["exact_revisions"]}|')
    lines += ['', '达到生成长度上限的读数：'+str(summary['truncated_readings'])+'。', '',
              '结果是人工已知区域的恢复比较，不含检测漏检；像素标签不能证明完全遮挡笔画恢复正确。读回率按区域等权平均，会容忍多读字符，因此同时报告严格全对。固定测试集历史上已经看过，仍需新的独立测试集支撑论文最终结论。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_manifest(out,config={'evaluation':str(root)},inputs={str(p):digest(p) for p in
        [inputs/'prepared_sha256.json',root/'pack/pack.json',root/'qwen8b/predictions.jsonl',root/'read_qwen8b.json']})


HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>统一验证与测试比较</title>
<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:24px auto;padding:0 16px;background:#f4f7fa;color:#17283a}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #dce3eb;text-align:left}.card{background:white;padding:16px;border-radius:10px;margin:14px 0}img{display:block;max-width:100%;height:auto;margin:12px auto}select,button{font:inherit;padding:8px;margin:3px}.bar{position:sticky;top:0;background:#e7eef8;padding:8px}.text{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#576879}</style>
<h1>相同数据、相同指标：__COUNT__ 个模型比较</h1><p>仅评测，权重预先冻结。人工微调旧版为 step800；本轮微调为 step400；若表中有纯合成版，则为原始框内 step8000，没有真实精修训练，但使用过真实验证集选版和第一篇标注素材。所有图像使用同一次 Qwen3-VL-8B 识别。</p>
<p><b>训练数据仍不相同：</b>三个自有版本的数据来源、训练量与七个 baseline 不同。这张表比较现有模型的实际表现，不能单独证明架构更好。区域来自人工框选，不是整页自动检测成绩。</p>
<div id="tables"></div><p id="notes"></p>
<div class="bar"><select id="split"><option value="test">测试集</option><option value="val">验证集</option></select><select id="model"></select><button id="prev">上一处</button><select id="sample"></select><button id="next">下一处</button></div>
<div class="card"><b>人工转写</b><div class="text" id="truth"></div><small id="id"></small></div>
<div class="card"><b>原图</b><img id="raw"><div class="text" id="rawread"></div></div>
<div class="card"><b id="title"></b><img id="restored"><div class="text" id="read"></div></div>
<script>const D=__DATA__,$=id=>document.getElementById(id),pct=x=>(100*x).toFixed(2),names={'Ours-old':'真实50处微调','Ours-round1':'真实90处微调','Ours-synthetic':'纯合成训练 · 原始框内模型'};const name=n=>names[n]||n;
for(const s of ['val','test']){const h=document.createElement('h2');h.textContent=s==='val'?'验证集：像素16处／识别41处':'测试集：像素7题28处／识别81处';$('tables').append(h);const t=document.createElement('table');let tr=t.insertRow();['模型','擦除精度%','划线召回%','像素F1%','原字误擦%','字符读回%','严格全对'].forEach(v=>{const e=document.createElement('th');e.textContent=v;tr.append(e)});const add=(n,p,o)=>{tr=t.insertRow();[name(n),...(p?['precision','recall','f1','valid_ink_damage'].map(k=>pct(p[k])):['—','0','0','0']),pct(o.held),Math.round(o.exact*o.exact_revisions)+'/'+o.exact_revisions].forEach(v=>tr.insertCell().textContent=v)};add('原图',null,D.summary.scan[s]);[...D.summary.models].sort((a,b)=>b.ocr[s].held-a.ocr[s].held).forEach(m=>add(m.name,m.pixels[s],m.ocr[s]));$('tables').append(t)}
$('notes').textContent='像素：灰度 <200，亮度增加 >0.2 算擦除。所有模型范围与分母一致。读回率会容忍多余字符，须结合严格全对看。生成截断：'+D.summary.truncated_readings+'。';D.summary.models.forEach(m=>$('model').add(new Option(name(m.name),m.name)));let samples=[];
function show(){const s=samples[Number($('sample').value)];if(!s)return;const n=$('model').value,a=s.variants.scan,b=s.variants[n];$('truth').textContent=s.truth;$('id').textContent=s.id;$('raw').src=a.image;$('restored').src=b.image;$('rawread').textContent='VLM：'+a.read;$('read').textContent='VLM：'+b.read;$('title').textContent=name(n)}
function populate(){samples=D.samples.filter(s=>s.split===$('split').value&&s.kind==='line');$('sample').replaceChildren();samples.forEach((s,i)=>$('sample').add(new Option((i+1)+' / '+samples.length+' · '+s.truth.slice(0,25),i)));show()}
$('split').onchange=populate;$('sample').onchange=show;$('model').onchange=show;$('prev').onclick=()=>{$('sample').selectedIndex=Math.max(0,$('sample').selectedIndex-1);show()};$('next').onclick=()=>{$('sample').selectedIndex=Math.min(samples.length-1,$('sample').selectedIndex+1);show()};populate();</script></html>'''


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    build(p.parse_args(argv))


if __name__=='__main__':main()
