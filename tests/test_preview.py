"""No GPU: validate private preview transport, cancellation and buffer cleanup."""
import base64
import contextlib
import io
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
import controller
import photo_preview


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.saved = controller.CLIENT, controller.PRIVATE_IMAGES
        controller.PRIVATE_IMAGES = True

    def tearDown(self):
        controller.CLIENT, controller.PRIVATE_IMAGES = self.saved

    def test_private_callback_receives_image_then_wipes_it(self):
        image, extra = bytearray(b'comparison_fixture'), bytearray(b'output_fixture')
        received = []
        def call(request):
            self.assertTrue(request['return_images'])
            self.assertFalse(request['debug_public'])
            return {'metrics': {'status': 'ok'}, 'comparison_png': image, 'output_png': extra}
        controller.CLIENT = types.SimpleNamespace(call=call)
        face, ref = bytearray(b'face'), bytearray(b'ref')
        result = controller.evaluate(face, ref, preview_fn=lambda data: received.append(bytes(data)))
        self.assertEqual(received, [b'comparison_fixture'])
        self.assertEqual(result, {'status': 'ok'})
        for value in (image, extra, face, ref):
            self.assertFalse(any(value))
        self.assertTrue(controller.PRIVATE_IMAGES)

    def test_callback_failure_still_cleans_every_buffer(self):
        image = bytearray(b'comparison_fixture')
        controller.CLIENT = types.SimpleNamespace(call=lambda req: {
            'metrics': {'status': 'ok'}, 'comparison_png': image})
        face = bytearray(b'face')
        def failing(data):
            raise RuntimeError('browser_fixture_failure')
        with self.assertRaisesRegex(RuntimeError, 'browser_fixture_failure'):
            controller.evaluate(face, preview_fn=failing)
        self.assertFalse(any(image))
        self.assertFalse(any(face))

    def test_private_batch_can_collect_human_scores_with_preview(self):
        previews, buffers = [], []
        def call(request):
            image = bytearray(b'preview')
            buffers.append(image)
            return {'metrics': {'status': 'ok', 'inference_s': 2., 'pipeline_s': 3.,
                                'peak_allocated_mib': 100.}, 'comparison_png': image}
        def loader(name):
            buffer = bytearray(b'input')
            buffers.append(buffer)
            return buffer
        controller.CLIENT = types.SimpleNamespace(service='hairstyle', call=call)
        cases = [{'id': f'{g}{i:02d}', 'group': g, 'target': 'target.jpg', 'reference': 'ref.jpg'}
                 for g in 'ABC' for i in range(1, 11)]
        with contextlib.redirect_stdout(io.StringIO()):
            rows = controller.run_batch(cases, loader,
                preview_fn=lambda data: previews.append(len(data)),
                score_fn=lambda case, metrics: {'identity': 4, 'hairline': 4, 'reference': 4})
        self.assertEqual(len(previews), 30)
        self.assertEqual(controller.quality_gate(rows)['gate'], 'PASS')
        self.assertTrue(all(not any(buffer) for buffer in buffers))
        self.assertTrue(all('comparison_png' not in row for row in rows))

    def test_private_export_stays_blocked_with_preview(self):
        with self.assertRaises(PermissionError):
            controller.run_batch([{'id': 'A01', 'group': 'A', 'target': 'x.jpg'}],
                                 lambda name: bytearray(), save_dir='forbidden', preview_fn=lambda x: None)

    def test_public_export_is_available_before_human_rating(self):
        controller.PRIVATE_IMAGES = False
        comparison, result = bytearray(b'comparison'), bytearray(b'result')
        controller.CLIENT = types.SimpleNamespace(service='hairstyle', call=lambda req: {
            'metrics': {'status': 'ok'}, 'comparison_png': comparison, 'output_png': result})
        with tempfile.TemporaryDirectory() as folder:
            def scores(case, metrics):
                self.assertEqual((Path(folder)/'A01_comparison.png').read_bytes(), b'comparison')
                return {'identity': 4, 'hairline': 4, 'reference': 4}
            with contextlib.redirect_stdout(io.StringIO()):
                rows = controller.run_batch([{'id': 'A01', 'group': 'A', 'target': 'x.jpg'}],
                    lambda name: bytearray(b'input'), save_dir=folder, allow_public_export=True, score_fn=scores)
            self.assertEqual(rows[0]['identity'], 4)
        self.assertFalse(any(comparison))
        self.assertFalse(any(result))

    def test_picker_returns_owned_bytes_without_display_data(self):
        google, colab, output = (types.ModuleType(name) for name in ('google', 'google.colab', 'google.colab.output'))
        response = {'data': base64.b64encode(b'fixture').decode()}
        scripts = []
        def evaluate(script, timeout_sec=None):
            scripts.append(script)
            return response
        output.eval_js = evaluate
        with patch.dict(sys.modules, {'google': google, 'google.colab': colab, 'google.colab.output': output}):
            result = photo_preview.pick_image('fixture')
        self.assertEqual(result, bytearray(b'fixture'))
        self.assertEqual(response, {})
        self.assertNotIn(base64.b64encode(b'fixture').decode(), scripts[0])
        controller.clear_buffer(result)

    def test_picker_cancel_is_explicit(self):
        output = types.ModuleType('google.colab.output')
        output.eval_js = lambda script, timeout_sec=None: {'error': 'cancelled'}
        with patch.dict(sys.modules, {'google.colab.output': output}):
            with self.assertRaisesRegex(ValueError, 'لغو'):
                photo_preview.pick_image('fixture')

    def test_preview_does_not_create_saved_image_output(self):
        output = types.ModuleType('google.colab.output')
        scripts = []
        output.eval_js = lambda script, timeout_sec=None: scripts.append(script) or True
        with patch.dict(sys.modules, {'google.colab.output': output}):
            photo_preview.show_transient(bytearray(b'public_fixture'))
        self.assertIn('URL.createObjectURL', scripts[0])
        self.assertIn('URL.revokeObjectURL', scripts[0])
        self.assertIn('300000', scripts[0])
        self.assertNotIn('localStorage', scripts[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
