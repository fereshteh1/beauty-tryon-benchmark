import io, json, os, sys, ast, unittest, zipfile, copy
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
import core
import controller as c


def png(color=(80, 90, 100), size=(32, 32)):
    im = Image.new('RGB', size, color)
    stream = io.BytesIO(); im.save(stream, format='PNG')
    data = bytearray(stream.getvalue()); core.wipe(im); core.wipe(stream)
    return data


def good_records():
    return [{'id':f'{g}{i:02d}', 'group':g, 'engine':'hairstyle', 'status':'ok',
             'identity':4,'hairline':4,'reference':4,'pipeline_s':7.,'inference_s':5.}
            for g in 'ABC' for i in range(1,11)]


class Tests(unittest.TestCase):
    def test_decode_jpeg_png_webp_cleanup_and_pixels(self):
        for fmt in ('JPEG', 'PNG', 'WEBP'):
            with self.subTest(format=fmt):
                image=Image.new('RGB',(40,30),(80,90,100))
                stream=io.BytesIO()
                image.save(stream,format=fmt)
                blob=bytearray(stream.getvalue())
                clean=core.decode_clean(blob)
                try:
                    self.assertEqual(clean.size,(40,30))
                    self.assertFalse(clean.info)
                    self.assertFalse(clean.getexif())
                    self.assertLessEqual(max(abs(v-w) for v,w in zip(clean.getpixel((20,15)),(80,90,100))),2)
                finally:
                    core.wipe(clean);core.wipe(clean)
                    core.wipe(blob);stream.close();image.close()

    def test_encoded_buffer_wiped_before_external_close(self):
        stream=core.WipingBytesIO(b'private encoded bytes')
        # A caller-held view permits inspection after zeroing and before release.
        view=io.BytesIO.getbuffer(stream)
        try:
            with self.assertRaises(BufferError):
                stream.close()
            self.assertEqual(bytes(view),b'\0'*len(view))
        finally:
            view.release()
            stream.close()
        core.wipe(stream);core.wipe(stream)
        closed=io.BytesIO(b'fixture');closed.close()
        core.wipe(closed);c.clear_buffer(closed)

    def test_official_public_demo_pngs_and_comparison_roundtrip(self):
        root=Path(os.environ.get('BEAUTY_TEST_DEMO','/content/beauty_phase0/public_demo'))
        if not all((root/name).is_file() for name in ('6.png','7.png')):
            self.skipTest('Official public demo assets have not been downloaded.')
        for name in ('6.png','7.png'):
            with self.subTest(public_demo=name):
                blob=bytearray((root/name).read_bytes())
                image=frame=decoded=None
                encoded=None
                try:
                    image=core.decode_clean(blob)
                    self.assertGreater(min(image.size),100)
                    self.assertFalse(image.info)
                    frame=core.comparison_frame(image,image,image,image)
                    encoded=core.encode_png(frame)
                    decoded=core.decode_clean(encoded)
                    self.assertEqual(decoded.size,frame.size)
                finally:
                    for value in (image,frame,decoded,encoded,blob):
                        if value is not None:core.wipe(value)

    def test_rejected_image_does_not_decode_during_cleanup(self):
        from unittest.mock import patch
        source=png(size=(32,32))
        image=Image.open(io.BytesIO(source))
        old=core.MAX_PIXELS;core.MAX_PIXELS=10
        try:
            with patch.object(core.Image,'open',return_value=image),patch.object(image,'load',side_effect=AssertionError('rejected image decoded')):
                with self.assertRaisesRegex(ValueError,'image_limits_exceeded'):
                    core.decode_clean(source)
        finally:
            core.MAX_PIXELS=old;core.wipe(source);image.close()

    def test_metadata_and_orientation(self):
        im = Image.new('RGB',(40,20),(50,100,150))
        exif = Image.Exif(); exif[274]=6; exif[270]='PRIVATE_METADATA'
        stream=io.BytesIO(); im.save(stream,format='JPEG',exif=exif)
        clean=core.decode_clean(bytearray(stream.getvalue()))
        self.assertEqual(clean.size,(20,40));self.assertFalse(clean.info);self.assertFalse(clean.getexif())
        result=core.encode_png(clean)
        self.assertNotIn(b'PRIVATE_METADATA',result)
        self.assertNotIn(b'eXIf',result)
        core.wipe(clean);core.wipe(result)

    def test_rgb_limits(self):
        with self.assertRaises(ValueError):core.decode_clean(bytearray())
        with self.assertRaises(Exception):core.decode_clean(bytearray(b'not an image'))
        old=core.MAX_PIXELS;core.MAX_PIXELS=10
        try:
            with self.assertRaises(ValueError):core.decode_clean(png())
        finally:core.MAX_PIXELS=old

    def test_mask_and_zero_strength(self):
        rgb=np.zeros((32,32,3),np.uint8);rgb[:]=[80,65,55]
        mask=np.zeros((32,32),np.float32);mask[8:24,8:24]=1
        out=core.lab_recolor(rgb,mask,(180,35,20),1.)
        np.testing.assert_array_equal(out[mask==0],rgb[mask==0])
        self.assertGreater(np.abs(out[mask>0].astype(float)-rgb[mask>0]).mean(),5.)
        unchanged=core.lab_recolor(rgb,mask,(180,35,20),0.)
        self.assertLessEqual(np.abs(unchanged.astype(int)-rgb.astype(int)).max(),1)
        with self.assertRaises(ValueError):core.lab_recolor(rgb,np.zeros_like(mask),(1,2,3))
        with self.assertRaises(ValueError):core.lab_recolor(rgb,mask,(999,2,3))

    def test_luminance_preserved_for_in_gamut_target(self):
        import cv2
        rgb=np.empty((32,32,3),np.uint8)
        for y in range(32):rgb[y]=[100+y,90+y,80+y]
        out=core.lab_recolor(rgb,np.ones((32,32),np.float32),(130,100,100),.5)
        before=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[...,0]
        after=cv2.cvtColor(out.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[...,0]
        self.assertLess(np.abs(before-after).max(),.5)

    def test_gate_21_and_failures_in_denominator(self):
        rows=good_records()
        for r in rows[21:]:r['status']='error'
        g=c.quality_gate(rows)
        self.assertEqual(g['gate'],'PASS');self.assertEqual(g['joint_passes'],21)
        self.assertEqual(g['denominator'],30)
        rows[20]['status']='error';self.assertEqual(c.quality_gate(rows)['gate'],'FAIL')

    def test_gate_missing_duplicate_and_mixed(self):
        rows=good_records();rows[0]['identity']=None
        self.assertEqual(c.quality_gate(rows)['gate'],'UNASSESSED')
        rows=good_records();rows[0]['id']=rows[1]['id']
        self.assertEqual(c.quality_gate(rows)['gate'],'INCOMPLETE')
        rows=good_records();rows[0]['engine']='color'
        self.assertEqual(c.quality_gate(rows)['gate'],'INCOMPLETE')
        self.assertEqual(c.quality_gate(good_records()[:21])['gate'],'INCOMPLETE')

    def test_gate_latency_boundary_and_tail(self):
        rows=good_records()
        for row in rows:row['inference_s']=8.
        self.assertEqual(c.quality_gate(rows)['gate'],'FAIL')
        rows=good_records();rows[0]['inference_s']=40.;rows[1]['inference_s']=40.
        self.assertEqual(c.quality_gate(rows)['gate'],'PASS')
        rows=good_records();rows[0]['inference_s']=float('nan')
        self.assertNotEqual(c.quality_gate(rows)['gate'],'PASS')

    def test_private_guards(self):
        c.PRIVATE_IMAGES=True
        with self.assertRaises(PermissionError):c.show_comparison(bytearray())
        with self.assertRaises(PermissionError):c.batch_folder('/irrelevant')
        case={'id':'A01','group':'A','target':'x.png','reference':'r.png'}
        with self.assertRaises(PermissionError):c.run_batch([case],None,save_dir='/irrelevant')

    def test_manifest_validation(self):
        case={'id':'A01','group':'A','target':'x.png','reference':'r.png'}
        c.validate_manifest([case])
        for path in ['/absolute.png','../secret.png','x/../../secret.png','x.exe','x\\y.png']:
            bad=dict(case,target=path)
            with self.assertRaises(ValueError):c.validate_manifest([bad])
        with self.assertRaises(ValueError):c.validate_manifest([case,case])

    def test_zip_no_extract_and_zero_after_last_use(self):
        case={'id':'A01','group':'A','target':'x.png','reference':'r.png'}
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('manifest.json',json.dumps([case]));archive.writestr('x.png',png());archive.writestr('r.png',png((10,20,30)))
        blob=bytearray(stream.getvalue())
        with c.zip_cases(blob) as (cases,loader,release):
            self.assertEqual(set(blob),{0})
            b=loader('x.png');self.assertTrue(b.startswith(b'\x89PNG'))
            release('x.png')
            with self.assertRaises(Exception):loader('x.png')
            self.assertTrue(loader('r.png').startswith(b'\x89PNG'));release('r.png')
        core.wipe(b)

    def test_batch_failure_consumes_buffers(self):
        class Dummy:
            service='hairstyle'
            def call(self,req):raise RuntimeError('simulated')
        old=c.CLIENT;c.CLIENT=Dummy();c.PRIVATE_IMAGES=True
        buffers=[]
        def loader(name):
            b=png();buffers.append(b);return b
        case={'id':'A01','group':'A','target':'x.png','reference':'r.png'}
        try:
            rows=c.run_batch([case],loader)
            self.assertEqual(rows[0]['status'],'error')
            for b in buffers:self.assertEqual(set(b),{0})
        finally:c.CLIENT=old

    def test_audit_guard_disk_network_inference(self):
        ns={'__name__':'not_main'}
        exec((Path(__file__).resolve().parents[1]/'engine/worker_tail.py').read_text(),ns)
        ns['GUARD_ACTIVE']=True
        for event,args in [('open',('/private','wb',0)),('open',('/private',None,os.O_WRONLY)),
                           ('subprocess.Popen',('x',)),('socket.connect',('x',))]:
            with self.assertRaises(PermissionError):ns['audit_guard'](event,args)
        ns['audit_guard']('open',('/public','rb',0))


if __name__=='__main__':
    unittest.main(verbosity=2)
