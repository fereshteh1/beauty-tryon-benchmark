"""Validate synchronized source, notebook schema and a source-only distribution."""
import ast
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PHOTO_WEIGHT_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.pth', '.pt', '.ckpt', '.safetensors', '.dat', '.npz', '.zip'}


def main():
    import nbformat
    files = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts]
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in PHOTO_WEIGHT_SUFFIXES:
            raise ValueError('Photo/weight/archive found in source release: '+str(relative))
        if path.stat().st_size > 2 * 2**20:
            raise ValueError('Unexpected large release file: '+str(relative))
        if path.suffix == '.py':
            ast.parse(path.read_text())
        if path.suffix in {'.py', '.md', '.ipynb', '.json', '.yml'}:
            text = path.read_text()
            if re.search(r'\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,})\b', text):
                raise ValueError('Potential credential in '+str(relative))
    notebook_path = ROOT/'notebooks/Beauty_TryOn_Phase0_T4.ipynb'
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    if len(code_cells) != 23:
        raise ValueError('Expected 23 numbered code cells')
    matched = set()
    for cell in code_cells:
        if cell.outputs or cell.execution_count is not None:
            raise ValueError('Notebook outputs/execution state must be cleared before release')
        source = cell.source
        if source.startswith('%%writefile '):
            header, body = source.split('\n', 1)
            target = Path(header.split(' ', 1)[1])
            relative = Path('tests')/target.name if 'tests' in target.parts else Path('engine')/target.name
            if (ROOT/relative).read_text() != body:
                raise ValueError('Notebook source differs from '+str(relative))
            matched.add(relative.as_posix())
            ast.parse(body)
        else:
            ast.parse(source)
    if len(matched) != 12:
        raise ValueError('Expected 12 visible source-definition cells')
    worker = (ROOT/'engine/core.py').read_text()+'\n\n'+(ROOT/'engine/worker_tail.py').read_text()
    if worker != (ROOT/'engine/worker.py').read_text():
        raise ValueError('Worker differs from its assembled sources')
    cases = json.loads((ROOT/'examples/manifest.json').read_text())
    sys.path.insert(0, str(ROOT/'engine'))
    import controller
    controller.validate_manifest(cases)
    ids = {case['id'] for case in cases}
    if ids != {f'{group}{i:02d}' for group in 'ABC' for i in range(1, 11)}:
        raise ValueError('Manifest template must contain 10 cases per group')
    print('PASS: Python sources, notebook schema/source parity, 23 cells, empty outputs, 30-case template, source-only release.')


if __name__ == '__main__':
    main()
