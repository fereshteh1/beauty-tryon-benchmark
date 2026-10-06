"""Setup and public-data benchmark commands for a Linux CUDA workstation."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/content/beauty_phase0', help='Code/assets root; no private images are written here')
    parser.add_argument('--backend', choices=('auto', 'torch_reference'), default='auto')
    subs = parser.add_subparsers(dest='command', required=True)
    for command in ('setup', 'verify', 'demo', 'status'):
        item = subs.add_parser(command)
        if command == 'demo':
            item.add_argument('--service', choices=('hairstyle', 'color'), default='hairstyle')
    batch = subs.add_parser('batch-public')
    batch.add_argument('folder', help='Existing public/synthetic dataset folder with manifest.json')
    batch.add_argument('--public-data', action='store_true', required=True, help='Confirm that every image is public/synthetic and permitted for this use')
    batch.add_argument('--service', choices=('hairstyle', 'color'), default='hairstyle')
    batch.add_argument('--save-outputs', action='store_true', help='Export public image results under the assets root')
    options = parser.parse_args()
    os.environ['BEAUTY_ROOT'] = str(Path(options.root).resolve())
    os.environ['BEAUTY_OP_BACKEND'] = options.backend
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
    import runtime
    runtime.ROOT.mkdir(parents=True, exist_ok=True)
    if options.command == 'setup':
        runtime.preflight()
        runtime.install()
        runtime.assets()
        runtime.verify()
    elif options.command == 'verify':
        runtime.verify()
    elif options.command == 'demo':
        stage = 17 if options.service == 'hairstyle' else 18
        with runtime.stage(stage, 'Public ' + options.service + ' demo'):
            runtime.demo(options.service, show=False)
    elif options.command == 'status':
        print(json.dumps({
            'model_python_exists': Path(runtime.PYTHON).is_file(),
            'repo_exists': (runtime.REPO / 'hair_swap.py').is_file(),
            'assets_manifest_exists': (runtime.ROOT / 'weight_manifest.json').is_file(),
            'public_warmup_exists': all((runtime.ROOT / 'public_demo' / name).is_file() for name in ('6.png', '7.png')),
            'full_30_image_gate': 'NOT_ESTABLISHED_BY_STATUS',
        }, indent=2))
    elif options.command == 'batch-public':
        api = runtime.api()
        previous = api.PRIVATE_IMAGES
        try:
            api.PRIVATE_IMAGES = False
            api.start_engine(options.service)
            destination = str(runtime.ROOT/'public_benchmark_outputs') if options.save_outputs else None
            rows = api.batch_folder(options.folder, show=False,
                save_dir=destination, allow_public_export=options.save_outputs,
                score_fn=api.record_scores if options.save_outputs else None)
            limit = 8. if options.service == 'hairstyle' else .5
            api.export_metrics(rows, path=str(runtime.CODE_ROOT/(options.service+'_benchmark.csv')), seconds=limit)
            print(json.dumps(api.quality_gate(rows, seconds=limit), indent=2))
        finally:
            api.PRIVATE_IMAGES = previous
            api.stop_engine()


if __name__ == '__main__':
    main()
