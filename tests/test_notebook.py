"""Verify fresh-kernel source deployment/imports without models or a GPU."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROBE = r'''
import ast,json,pathlib,sys
notebook=json.loads(pathlib.Path(sys.argv[1]).read_text())
root=pathlib.Path(sys.argv[2])
code=root/'benchmark_code/engine'
code.mkdir(parents=True)
(code.parent/'tests').mkdir()
cells=[c for c in notebook['cells'] if c['cell_type']=='code']
for cell in cells:
    source=''.join(cell['source'])
    if source.startswith('%%writefile '):
        header,body=source.split('\n',1)
        target=pathlib.Path(header.split(' ',1)[1].replace('/content/beauty_phase0',str(root)))
        target.write_text(body)
source=''.join(cells[15]['source']).replace('/content/beauty_phase0',str(root))
tree=ast.parse(source)
assert isinstance(tree.body[-1],ast.With)
tree.body=tree.body[:-1]
exec(compile(tree,'<fresh-notebook-assembly>','exec'))
import runtime,controller,photo_preview,hair_engine,color_engine
assert runtime.CODE_ROOT==code
assert runtime.TEST_ROOT==code.parent/'tests'
assert runtime.ROOT==root
assert (code/'worker.py').is_file()
assert all((runtime.TEST_ROOT/n).is_file() for n in ('test_core.py','test_runtime.py','test_operators.py'))
print('fresh notebook deployment passed; no inference performed')
'''


class NotebookTests(unittest.TestCase):
    def test_fresh_kernel_has_no_old_globals_or_paths(self):
        notebook = Path(__file__).resolve().parents[1]/'notebooks/Beauty_TryOn_Phase0_T4.ipynb'
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, '-c', PROBE, str(notebook), folder],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('deployment passed', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
