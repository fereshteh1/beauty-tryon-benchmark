"""CPU regression tests for notebook state, diagnostics and framed IPC; no model inference."""
import contextlib, io, json, os, pathlib, sys, tempfile, time, unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1] / 'engine'))
import controller as api
import runtime as runtime

FAKE_WORKER = r'''
import sys, struct, pickle
def send(value):
    body=pickle.dumps(value,protocol=5)
    sys.stdout.buffer.write(struct.pack('!Q',len(body))+body)
    sys.stdout.buffer.flush()
send({'status':'ready'})
while True:
    header=sys.stdin.buffer.read(8)
    if not header: break
    length=struct.unpack('!Q',header)[0]
    request=pickle.loads(sys.stdin.buffer.read(length))
    send({'metrics':{'status':'ok','fixture':True}})
'''

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=pathlib.Path(self.temp.name)
        self.saved=runtime.CODE_ROOT
        runtime.CODE_ROOT=self.root
        api.configure(self.root,self.root,sys.executable,dict(os.environ))

    def tearDown(self):
        api.stop_engine()
        runtime.CODE_ROOT=self.saved
        self.temp.cleanup()

    def test_stage_failure_has_type_without_private_message(self):
        with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):
            with runtime.stage(8,'fixture'):
                raise ValueError('PRIVATE_INPUT_FILENAME')
        raw=(self.root/'run_status.json').read_text()
        self.assertNotIn('PRIVATE_INPUT_FILENAME',raw)
        self.assertEqual(json.loads(raw)['8']['status'],'FAILED')

    def test_stage_success_persisted(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with runtime.stage(4,'fixture'): pass
        self.assertEqual(json.loads((self.root/'run_status.json').read_text())['4']['status'],'PASSED')

    def test_pipe_deadline(self):
        readfd,writefd=os.pipe()
        try:
            with os.fdopen(readfd,'rb',buffering=0) as stream:
                with self.assertRaises(TimeoutError):
                    api.recv_exact(stream,8,time.monotonic()+.03)
        finally:
            os.close(writefd)

    def test_transport_consumes_inputs_and_closes_worker(self):
        (self.root/'worker.py').write_text(FAKE_WORKER)
        client=api.ModelClient('hairstyle')
        api.CLIENT=client
        face,reference=bytearray(b'fixture_face'),bytearray(b'fixture_ref')
        result=api.evaluate(face,reference)
        self.assertTrue(result['fixture'])
        self.assertEqual(set(face),{0})
        self.assertEqual(set(reference),{0})
        api.stop_engine()
        self.assertIsNotNone(client.process.poll())

    def test_initialization_failure_contains_original_diagnostic(self):
        fake=FAKE_WORKER.split("send({'status':'ready'})")[0]
        fake+="print('fixture backend error',file=sys.stderr,flush=True)\nsend({'status':'init_failed'})\n"
        (self.root/'worker.py').write_text(fake)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'fixture backend error'):
                api.ModelClient('hairstyle')

    def test_installer_failure_contains_actual_output(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'fixture_install_failure'):
                runtime.run(sys.executable,'-c',"print('fixture_install_failure');raise SystemExit(2)")

    def test_initialization_crash_contains_original_diagnostic(self):
        (self.root/'worker.py').write_text("import sys\nprint('fixture fatal import',file=sys.stderr,flush=True)\nraise SystemExit(3)\n")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'fixture fatal import'):
                api.ModelClient('hairstyle')

    def test_summary_rejects_stale_metrics(self):
        import types
        from unittest.mock import patch, Mock
        display_module=types.ModuleType('IPython.display')
        display_module.display=Mock()
        display_module.Markdown=lambda value:value
        (self.root/'hairstyle_demo_metrics.json').write_text(json.dumps({
            'status':'ok','inference_s':1,'peak_allocated_mib':2}))
        with patch.dict(sys.modules,{'IPython.display':display_module}),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'17'):
                runtime.summary()

    def test_fresh_kernel_module_configuration(self):
        import importlib.util
        path=pathlib.Path(__file__).resolve().parents[1]/'engine/runtime.py'
        first=importlib.util.spec_from_file_location('fixture_runtime_one',path)
        second=importlib.util.spec_from_file_location('fixture_runtime_two',path)
        a,b=importlib.util.module_from_spec(first),importlib.util.module_from_spec(second)
        first.loader.exec_module(a);second.loader.exec_module(b)
        self.assertEqual(a.ROOT,b.ROOT)
        self.assertEqual(a.PYTHON,b.PYTHON)
        self.assertEqual(a.ENV['HF_HUB_OFFLINE'],'1')

if __name__=='__main__': unittest.main(verbosity=2)
