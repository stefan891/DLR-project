"""
Demo script: loads stereo pairs from a STIOS-style folder, runs the INSTR
predictor, and saves the colorized overlay for each left image into --save-dir
using the source image's filename.
"""

import os
import argparse
from turtle import right
import cv2
import torch
import numpy as np
from PIL import Image
from predictor import Predictor
from utils.pred_utils import load_data


def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict', type=str, default='./pretrained_instr/models/pretrained_model.pth', help="Path to INSTR checkpoint")
    parser.add_argument('--root', type=str, required=True, help="STIOS root")
    parser.add_argument('--rcvisard', default=False, action='store_true', help="Run on rc_visard images")
    parser.add_argument('--zed', default=False, action='store_true', help="Run on ZED images")
    parser.add_argument('--save-dir', type=str, default=None, help="Directory to save prediction overlays")
    parser.add_argument('--focal-length', type=float, default=None, help="Camera focal length (defaults per sensor)")
    parser.add_argument('--baseline', type=float, default=None, help="Camera baseline in meters (defaults per sensor)")
    parser.add_argument('--alpha', type=float, default=0.4, help="Blend alpha for overlay")
    parser.add_argument('--aux-modality', type=str, default='depth', choices=['depth', 'disp'])
    args = parser.parse_args()

    assert args.rcvisard or args.zed
    assert not (args.rcvisard and args.zed)
    assert os.path.isfile(args.state_dict)

    if args.save_dir is not None:
        print(f"Saving images to {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)

    if args.focal_length is None or args.baseline is None:
        if args.zed:
            args.focal_length = 1390.0277099609375 / (2208 / 640)
            args.baseline = 0.12
        else:
            args.focal_length = 1082.28 / (1280 / 640)
            args.baseline = 0.0650206

    net = Predictor(
        state_dict_path=args.state_dict,
        focal_length=args.focal_length,
        baseline=args.baseline,
        return_depth=True if args.aux_modality == 'depth' else False
        )

    paths = load_data(root=args.root, sensor='rc_visard' if args.rcvisard else 'zed')

    for sensor in paths.keys():
        if sensor == 'zed' and not args.zed:
            continue
        if sensor == 'rc_visard' and not args.rcvisard:
            continue

        for folder in paths[sensor].keys():
            for left in paths[sensor][folder]:
                
                left = Image.open(left)
                left_t = net.process_im(left).to(net.device)
                

                with torch.no_grad():
                    pred_segmap, pred_depth = net.predict(left_t)

                left = cv2.resize(left, (640, 480), interpolation=cv2.INTER_LINEAR)
                left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
                cv2.imshow('left', cv2.resize(left.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
                cv2.imshow('pred', left_overlay)
                cv2.imshow(args.aux_modality, pred_depth / pred_depth.max())
                cv2.waitKey(1)

                if args.save_dir is not None:
                    filename = os.path.basename(left_path)
                    cv2.imwrite(
                        os.path.join(args.save_dir, filename),
                        cv2.cvtColor(left_overlay, cv2.COLOR_RGB2BGR),
                    )


if __name__ == '__main__':
    demo()
