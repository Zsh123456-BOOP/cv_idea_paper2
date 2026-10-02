"""The OCR model that cuts the lines: Unlimited-OCR (Baidu, MIT), run exactly as it was for the catalogue.

``UnlimitedOCR.read(page_image_file)`` returns the model's text for one whole-question image: its layout regions with their
``<|det|>type [x0, y0, x1, y1]<|/det|>`` boxes (thousandths of the page), which ``data.ocr_lines.strict_lines`` cuts into lines.
Needs ``transformers`` (4.57) and the model folder (6.4 GB, trust_remote_code); a page takes about 8 s on an RTX 4090.
"""
import contextlib
import io
import tempfile

SETTINGS = dict(prompt='<image>document parsing.', base_size=1024, image_size=640, crop_mode=True, max_length=8192,
                no_repeat_ngram_size=35, ngram_window=128, save_results=False)


class UnlimitedOCR:
    def __init__(self, directory, device='cuda'):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(directory, trust_remote_code=True, use_safetensors=True,
                                               dtype=torch.bfloat16).eval().to(device)

    def read(self, image_file):
        """The model's text for one page image file."""
        import torch
        printed = io.StringIO()                       # the model prints its answer when it does not return it
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(printed):
            result = self.model.infer(self.tokenizer, image_file=str(image_file), output_path=scratch, **SETTINGS)
        torch.cuda.empty_cache()
        return (result if isinstance(result, str) and result.strip() else printed.getvalue()).strip()
