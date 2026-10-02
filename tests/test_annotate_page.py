import json

from revrender.tools.annotate import single_file as m


def package(tmp_path):
    (tmp_path/'crops').mkdir()
    (tmp_path/'refs').mkdir()
    (tmp_path/'crops'/'a.jpg').write_bytes(b'\xff\xd8jpeg')
    (tmp_path/'refs'/'a.png').write_bytes(b'\x89PNGpng')
    pack = {'schema_version': 's', 'rows': [{'line_id': 'x:L1', 'file': 'a.jpg', 'reference': 'refs/a.png', 'versions': [{'marks': []}]}]}
    (tmp_path/'manifest.js').write_text('window.PACK = '+json.dumps(pack)+';\n')
    return tmp_path


def test_the_page_needs_no_file_beside_it(tmp_path):
    page, pack = m.build(package(tmp_path), m.EDITOR.read_text(encoding='utf-8'))
    assert 'src="manifest.js"' not in page and 'window.PACK = ' in page
    row = pack['rows'][0]
    assert row['image_data'].startswith('data:image/jpeg;base64,') and row['reference_data'].startswith('data:image/png;base64,')
    assert row['versions'] == [{'marks': []}] and row['file'] == 'a.jpg'


def test_the_editor_reads_inline_images_before_the_folder():
    text = m.EDITOR.read_text(encoding='utf-8')
    assert 'r.image_data || ("crops/" + r.file)' in text and 'r.reference_data || r.reference' in text
    assert text.count(m.BEGIN) == 1 and text.count(m.END) == 1


def test_a_closing_script_tag_in_the_data_cannot_end_the_script(tmp_path):
    folder = package(tmp_path)
    pack = m.read_manifest(folder)
    pack['rows'][0]['line_id'] = '</script><b>'
    (folder/'manifest.js').write_text('window.PACK = '+json.dumps(pack)+';\n')
    page, _ = m.build(folder, m.EDITOR.read_text(encoding='utf-8'))
    assert '</script><b>' not in page
