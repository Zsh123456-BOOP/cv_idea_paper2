import numpy as np
from revrender.baselines.restoration import restore_native, removal_counts, removal_metrics


def test_native_tiling_identity_preserves_geometry_and_all_edge_pixels():
    rng=np.random.default_rng(42)
    for shape in [(36,99),(290,533),(256,256)]:
        x=rng.random(shape,dtype=np.float32)
        assert np.array_equal(restore_native(lambda a:a,x),x)


def test_removal_counts_do_not_reward_all_white_or_identity():
    image=np.ones((30,40),np.float32);image[4:20,8:12]=0;image[10:13,3:35]=0
    gold=np.zeros(image.shape,bool);gold[10:13,3:35]=True;gold[10:13,8:12]=False
    perfect=image.copy();perfect[gold]=1
    result=removal_metrics(removal_counts(image,perfect,gold))
    assert result['f1']==1 and result['valid_ink_damage']==0
    white=removal_metrics(removal_counts(image,np.ones_like(image),gold))
    assert white['f1']<1 and white['valid_ink_damage']==1
    assert removal_metrics(removal_counts(image,image,gold))['f1']==0


def test_added_ink_is_reported_separately_from_removal():
    image=np.ones((8,8),np.float32);pred=image.copy();pred[0,0]=0
    result=removal_metrics(removal_counts(image,pred,np.zeros((8,8),bool)))
    assert result['new_dark_pixels']==1 and result['tp']==0


def test_export_merge_deduplicates_only_identical_scans(tmp_path):
    import json
    from argparse import Namespace
    from PIL import Image
    from revrender.baselines.restoration_export import merge
    dirs=[]
    for name in ['NAFNet','DocDiff']:
        directory=tmp_path/name;(directory/'pack').mkdir(parents=True);dirs.append(directory)
        Image.fromarray(np.zeros((8,8),np.uint8)).save(directory/'pack/image.png')
        rows=[{'id':f'{model}|sample','model':model,'images':{'erased':'image.png'}} for model in ['scan',name]]
        (directory/'pack/pack.json').write_text(json.dumps({'rows':rows}))
        (directory/'run_manifest.json').write_text('{}')
    merge(Namespace(inputs=dirs,output_dir=tmp_path/'merged'))
    rows=json.loads((tmp_path/'merged/pack.json').read_text())['rows']
    assert len(rows)==3 and all(__import__('pathlib').Path(r['images']['erased']).is_file() for r in rows)
