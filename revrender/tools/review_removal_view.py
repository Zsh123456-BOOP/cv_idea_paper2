"""Offline native-pixel comparison of an incremental remover round on fixed validation."""
import argparse
import base64
import html
import io
import json

from pathlib import Path
import numpy as np
from PIL import Image

from ..util import digest, make_output_dir, write_manifest


def build(dataset, evaluation, output_dir):
    summary=json.loads((evaluation/'summary.json').read_text())
    records=json.loads((dataset/'validation_native.json').read_text())
    native=np.load(dataset/'validation_native.npz',allow_pickle=False)
    # If no candidate passed the replacement rule, still show the best new F1
    # for diagnosis; explicitly label that it has not replaced the old model.
    eligible=[n for n in summary['models'] if n!='baseline']
    candidate=summary['selected'] if summary['selected']!='baseline' else max(eligible,key=lambda n:summary['models'][n]['metrics']['f1'])
    before=np.load(evaluation/'baseline.npz',allow_pickle=False)
    alternatives={name:np.load(evaluation/(name+'.npz'),allow_pickle=False) for name in eligible}
    def image(array):
        stream=io.BytesIO();Image.fromarray(array).save(stream,format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()
    cards=[]
    for i,r in enumerate(records):
        scan=native[f'image_{i}'];gold=native[f'mask_{i}'].astype(bool)
        old=scan.copy();old[before[f'mask_{i}'].astype(bool)]=255
        target=scan.copy();target[gold]=255
        def figure(name,im,version=None):
            attrs=(' class="candidate" data-version="'+html.escape(version)+'"'+(' hidden' if version!=candidate else '')) if version else ''
            return '<figure'+attrs+'><figcaption>'+html.escape(name)+'</figcaption><img alt="'+html.escape(name)+'" src="'+image(im)+'" style="width:'+str(min(im.shape[1]*3,1000))+'px"></figure>'
        figures=figure('原图',scan)+figure('旧模型',old)
        for name,prediction in alternatives.items():
            restored=scan.copy();restored[prediction[f'mask_{i}'].astype(bool)]=255
            figures+=figure('本轮 '+name,restored,name)
        cards.append('<section><h2>验证区域 '+str(i+1)+'</h2><p class="id">'+html.escape(r['id'])+'</p>'+figures+figure('人工精修参考',target)+'</section>')
    rows=[]
    for name,v in summary['models'].items():
        m=v['metrics'];rows.append('<tr><td>'+html.escape('旧模型' if name=='baseline' else name)+'</td>'+''.join('<td>'+f'{m[k]*100:.2f}%</td>' for k in ('precision','recall','f1','valid_ink_erased'))+'</tr>')
    verdict='本轮通过替换规则：'+summary['selected'] if summary['selected']!='baseline' else '本轮未通过替换规则，保留旧模型；下方展示新版本中 F1 最高者，仅供比较。'
    selector='<label>查看新版本：<select id="version">'+''.join('<option value="'+html.escape(n)+'"'+(' selected' if n==candidate else '')+'>'+html.escape(n)+'</option>' for n in eligible)+'</select></label>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>本轮微调验证对比</title><style>
[hidden]{display:none!important}select{font:inherit;padding:6px}label{position:sticky;top:0;display:block;background:#f3f5f8;padding:10px}body{margin:0;background:#f3f5f8;color:#203044;font:16px/1.6 system-ui}main{max-width:1100px;margin:auto;padding:24px}h1{font-size:25px}h2{font-size:18px;margin:0}.id{font-size:12px;color:#65758a;overflow-wrap:anywhere}section{background:white;margin:22px 0;padding:18px;border:1px solid #dce2ea;border-radius:10px}figure{margin:12px 0;border:1px solid #e4e8ed;padding:8px;overflow:auto}figcaption{font-size:13px;color:#526275}img{display:block;max-width:100%;height:auto;image-rendering:pixelated}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}.summary{background:#e7eefc;padding:14px;border-radius:8px}
</style><main><h1>本轮微调 · 固定验证集对比</h1><p class="summary">'''+html.escape(verdict)+'''</p><p>同一份原生验证图、同一阈值，不叠加标注页的自动残留清理。擦除精确率、召回、F1 越高越好；原字误擦率越低越好。人工参考是像素标注，不是划线前扫描图。</p><table><tr><th>版本</th><th>擦除精确率</th><th>划线召回</th><th>F1</th><th>原字误擦率</th></tr>'''+''.join(rows)+'</table>'+selector+''.join(cards)+'''</main><script>document.getElementById('version').onchange=e=>document.querySelectorAll('.candidate').forEach(f=>f.hidden=f.dataset.version!==e.target.value);</script></html>'''
    out=make_output_dir(output_dir);(out/'index.html').write_text(page)
    inputs=[dataset/'validation_native.json',dataset/'validation_native.npz',evaluation/'summary.json',evaluation/'baseline.npz',*(evaluation/(n+'.npz') for n in eligible)]
    write_manifest(out,config={'candidate_shown':candidate,'selected':summary['selected']},inputs={str(p):digest(p) for p in inputs},regions=len(records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','evaluation','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.dataset,a.evaluation,a.output_dir)
