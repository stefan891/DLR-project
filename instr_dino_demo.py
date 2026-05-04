import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import time
import cv2

from predictor import Predictor
from utils.pred_utils import load_data, process_im, stuff_from_state_dict_path, overlay_im_with_masks
from utils.confmat import ConfusionMatrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict', type=str, default='./pretrained_instr/models/pretrained_model.pth', help="Path to INSTR checkpoint")
    parser.add_argument('--root', type=str, required=True, help="STIOS root")
    parser.add_argument('--rcvisard', default=False, action='store_true', help="Run on rc_visard images")
    parser.add_argument('--zed', default=False, action='store_true', help="Run on ZED images")
    parser.add_argument('--save-dir', type=str, default=None, help="Directory to save prediction overlays")
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--aux-modality', type=str, default='depth', choices=['depth', 'disp'])
    parser.add_argument('--viz', default=False, action='store_true')
    parser.add_argument('--backbone', type=str, default='dinov2b', choices=['dinov2l', 'dinov2b', 'INSTR'], help="Backbone to use (overrides config.yaml setting)")
    parser.add_argument('--dino-weights', type=str, default=None, help="Path to local DINO pretrained .pth (skips HF Hub download)")
    args = parser.parse_args()
    
    assert args.rcvisard or args.zed
    assert not (args.rcvisard and args.zed)
    assert os.path.isfile(args.state_dict)
    
    paths = load_data(root=args.root, sensor='rc_visard' if args.rcvisard else 'zed')
    net = Predictor(state_dict_path=args.state_dict, return_depth=True if args.aux_modality == 'depth' else False, backbone=args.backbone, dino_weights=args.dino_weights)
    
    for sensor in paths.keys():
        if sensor == 'zed' and not args.zed:
            continue
        if sensor == 'rc_visard' and not args.rcvisard:
            continue
        
        for folder in tqdm(paths[sensor].keys()):
            
            for left_path, _ in tqdm(paths[sensor][folder]):
                
                
                # print(f"Predicting {left_path}...")
                left_im = Image.open(left_path)
                with torch.no_grad():
                    pred_segmap, pred_depth = net.predict(left_im)
                
                if args.viz:
                    # left = Image.open(left_path)
                    left = cv2.resize(np.array(left), (640, 480), interpolation=cv2.INTER_LINEAR)
                    left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
                    cv2.imshow('left', cv2.resize(left.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
                    cv2.imshow('pred', left_overlay)
                    cv2.imshow(args.aux_modality, pred_depth / pred_depth.max())
                    cv2.waitKey(1)
                    
                if args.save_dir:
                    save_folder = os.path.join(args.save_dir, sensor, folder)
                    os.makedirs(save_folder, exist_ok=True)
                    left = cv2.resize(np.array(left_im), (640, 480), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(save_folder, os.path.basename(left_path) + '.png'), left)
                    cv2.imwrite(os.path.join(save_folder, os.path.basename(left_path) + '_mask.png'), pred_segmap)
                    # left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
                    left_overlay = overlay_im_with_masks(left, pred_segmap, alpha=args.alpha)
                    cv2.imwrite(os.path.join(save_folder, os.path.basename(left_path) + '_overlay.png'), left_overlay)
                    

if __name__ == '__main__':
    main()