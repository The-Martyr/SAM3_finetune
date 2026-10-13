"""Install the reviewed patch onto a clean, pinned upstream checkout."""
import argparse
from pathlib import Path
import shutil
import subprocess

REVISION = '660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7'
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=ROOT / 'vendor' / 'sam3')
args = parser.parse_args()
source = args.source.resolve()
if not source.exists():
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/sam3.git', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', REVISION], check=True)
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
if revision != REVISION:
    raise SystemExit('Upstream revision mismatch; use a fresh checkout at ' + REVISION)
marker = source / '.sam3_sft_installed'
if marker.exists():
    raise SystemExit('Already installed; use a fresh checkout to update the patch')
if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
    raise SystemExit('Refusing to patch a dirty upstream checkout')
patch = ROOT / 'patches' / 'training.patch'
subprocess.run(['git', '-C', str(source), 'apply', '--check', str(patch)], check=True)
subprocess.run(['git', '-C', str(source), 'apply', str(patch)], check=True)
shutil.copyfile(ROOT / 'overlay' / 'freeze_policy.py', source / 'sam3/train/freeze_policy.py')
marker.write_text(REVISION + '\n')
print('Patched upstream:', source)
