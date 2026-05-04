"""
Test script: run the pretrained INSTR model on the STIOS dataset (left RGB only).

Usage:
    python test_stios.py \
        --state-dict ./pretrained_instr/models/model_39.pth \
        --root ./STIOS
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from predictor import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict', type=str, required=True,
                        help="Path to the pretrained INSTR checkpoint (.pth)")
    parser.add_argument('--root', type=str, required=True,
                        help="STIOS dataset root directory")
    parser.add_argument('--backbone', type=str, default='INSTR',
                        choices=['INSTR', 'dinov2b', 'dinov2l'],
                        help="Backbone to use (default: INSTR). dinov2b/dinov2l download from torch hub.")
    args = parser.parse_args()

    assert os.path.isfile(args.state_dict), f"State dict not found: {args.state_dict}"
    assert os.path.isdir(args.root), f"STIOS root not found: {args.root}"

    predictor = Predictor(state_dict_path=args.state_dict, return_depth=False,
                          backbone=args.backbone)

    sensors = [s for s in ('rc_visard', 'zed')
               if os.path.isdir(os.path.join(args.root, s))]
    assert sensors, f"No 'rc_visard' or 'zed' subfolder found under {args.root}"

    total_processed = 0

    for sensor in sensors:
        sensor_path = os.path.join(args.root, sensor)
        folders = sorted(d for d in os.listdir(sensor_path)
                         if os.path.isdir(os.path.join(sensor_path, d)))

        for folder in tqdm(folders, desc=sensor):
            left_dir = os.path.join(sensor_path, folder, 'left_rgb')
            if not os.path.isdir(left_dir):
                continue

            fnames = sorted(f for f in os.listdir(left_dir)
                            if not f.startswith('.'))
            if not fnames:
                continue

            for fname in fnames:
                left_path = os.path.join(left_dir, fname)
                left_img = np.array(Image.open(left_path).convert('RGB'))

                predictor.predict(left_img)
                total_processed += 1

    print(f"Done. {total_processed} images processed.")


if __name__ == '__main__':
    main()
