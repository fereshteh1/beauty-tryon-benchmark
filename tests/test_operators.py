import importlib.util, os, pathlib, unittest
os.environ['BEAUTY_OP_BACKEND']='torch_reference'
torch = None
if os.environ.get('BEAUTY_TEST_REPO'):
    import torch

class Operators(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('BEAUTY_TEST_REPO'), 'GPU setup assets are not configured; CPU CI does not download models.')
    def test_all_six_pinned_operator_copies(self):
        count=0
        for path in pathlib.Path(os.environ.get('BEAUTY_TEST_REPO', 'research/repo_patched')).rglob('op/*.py'):
            if path.name not in {'fused_act.py','upfirdn2d.py'}:continue
            spec=importlib.util.spec_from_file_location('operator_'+str(count),path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            x=torch.arange(-8.,8.).reshape(1,1,4,4)
            if path.name=='fused_act.py':
                y=module.fused_leaky_relu(x,torch.tensor([1.]),negative_slope=.1,scale=2.)
                expected=torch.nn.functional.leaky_relu(x+1.,negative_slope=.1)*2.
                torch.testing.assert_close(y,expected)
                layer=module.FusedLeakyReLU(1)
                self.assertIn('bias',layer.state_dict())
                self.assertIsNone(module.fused)
            else:
                y=module.upfirdn2d(x,torch.ones(1,1),up=2)
                self.assertEqual(y.shape,(1,1,8,8))
                torch.testing.assert_close(y[:,:,::2,::2],x)
                self.assertEqual(float(y[:,:,1::2,:].abs().sum()),0.)
                z=module.upfirdn2d(y,torch.ones(1,1),down=2)
                torch.testing.assert_close(z,x)
                self.assertIsNone(module.upfirdn2d_op)
            count+=1
        self.assertEqual(count,6)

if __name__=='__main__':unittest.main(verbosity=2)
