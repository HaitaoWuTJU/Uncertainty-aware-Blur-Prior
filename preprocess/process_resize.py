import os
from PIL import Image
from torchvision import transforms
import argparse

# Source THINGS images (500x500), shared across dataset formats
IMAGE_SRC_DIR = os.environ.get('UBP_IMAGE_SRC_DIR', '/work3/s193209/data/images')
# Resized images stay in the shared images dir, not in the per-format EEG dirs
IMAGE_RESIZE_DIR = os.environ.get('UBP_IMAGE_RESIZE_DIR',
    os.path.join(IMAGE_SRC_DIR, 'Image_set_Resize'))

# UBP expects the train split dir to be called 'train_images'
SPLIT_DIRS = {'training_images': 'train_images', 'test_images': 'test_images'}


def get_args_parser():
    parser = argparse.ArgumentParser('resize THINGS images for UBP', add_help=False)
    parser.add_argument('--type', type=str, default='eeg', choices=['eeg', 'meg'])
    parser.add_argument('--src', type=str, default=None,
        help=f'source image root (default for --type eeg: {IMAGE_SRC_DIR})')
    parser.add_argument('--dst', type=str, default=None,
        help=f'output dir (default for --type eeg: {IMAGE_RESIZE_DIR})')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--overwrite', action='store_true',
        help='re-resize images that already exist in --dst')
    return parser.parse_args()


args = get_args_parser()

if args.type == 'eeg':
    data_dir = args.src or IMAGE_SRC_DIR
    save_dir = args.dst or IMAGE_RESIZE_DIR
elif args.type == 'meg':
    data_dir = args.src or 'data/things-meg/Image_set'
    save_dir = args.dst or 'data/things-meg/Image_set_Resize'

os.makedirs(save_dir, exist_ok=True)
print(f'{data_dir} -> {save_dir} ({args.size}x{args.size})')

# Only walk the split dirs: the source tree may hold unrelated images (plots, etc.)
splits = {src: dst for src, dst in SPLIT_DIRS.items()
          if os.path.isdir(os.path.join(data_dir, src))}
if not splits:
    raise FileNotFoundError(f'none of {list(SPLIT_DIRS)} found in {data_dir}')

t1 = transforms.Resize((args.size, args.size))

for src_split, dst_split in splits.items():
    n_done = n_skipped = 0
    for root, dirs, files in os.walk(os.path.join(data_dir, src_split)):
        for file in sorted(files):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            path = os.path.join(root, file)
            rel = os.path.relpath(path, os.path.join(data_dir, src_split))
            save_path = os.path.join(save_dir, dst_split, rel)
            if os.path.exists(save_path) and not args.overwrite:
                n_skipped += 1
                continue
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            t1(Image.open(path).convert('RGB')).save(save_path)
            n_done += 1
    print(f'{src_split} -> {dst_split}: {n_done} written, {n_skipped} already present')
