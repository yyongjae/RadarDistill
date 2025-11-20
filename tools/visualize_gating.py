"""
Generate gating visualizations from trained checkpoint

Usage:
    # Visualize validation set
    python visualize_gating.py \
        --cfg_file cfgs/radar_distill/radar_distill_train_multi_sweep.yaml \
        --ckpt ../output/radar_distill_train_multi_sweep/default/ckpt/checkpoint_epoch_40.pth \
        --num_samples 50 \
        --output_dir paper_figures/
        
This generates:
    - Gating choice maps (PNG)
    - BEV overlay visualizations
    - Per-scene statistics
"""

import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('..')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.models.backbones_2d.gating_analysis import GatingAnalyzer


def colorize_gating_choice(choice_map, teacher_sweeps=[1, 5, 10]):
    """
    Colorize gating choice for visualization
    
    Args:
        choice_map: [H, W] array with values 0, 1, 2
        teacher_sweeps: list of teacher sweep counts
    
    Returns:
        [H, W, 3] RGB image
    """
    # Color scheme: Blue(s1), Green(s5), Red(s10)
    colors = [
        (66, 135, 245),   # Blue for recent (s1)
        (76, 209, 55),    # Green for mid (s5)
        (245, 66, 66)     # Red for old (s10)
    ]
    
    h, w = choice_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        mask = (choice_map == i)
        rgb[mask] = color
    
    return rgb


def visualize_sample(model, batch_dict, output_dir, sample_idx, teacher_sweeps):
    """Generate visualization for one sample"""
    model.eval()
    with torch.no_grad():
        # Forward pass
        batch_dict = model(batch_dict)
        
        # Get gating weights from backbone
        alpha_low = batch_dict.get('alpha_low')
        alpha_high = batch_dict.get('alpha_high')
        
        if alpha_low is None:
            print("Warning: alpha_low not found in batch_dict")
            return
        
        # Get choice maps (argmax)
        choice_low = alpha_low.argmax(dim=1)[0, 0].cpu().numpy()   # [H, W]
        choice_high = alpha_high.argmax(dim=1)[0, 0].cpu().numpy()
        
        # Colorize
        img_low = colorize_gating_choice(choice_low, teacher_sweeps)
        img_high = colorize_gating_choice(choice_high, teacher_sweeps)
        
        # Save
        sample_dir = output_dir / f'sample_{sample_idx:04d}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(sample_dir / 'gating_low.png'), cv2.cvtColor(img_low, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sample_dir / 'gating_high.png'), cv2.cvtColor(img_high, cv2.COLOR_RGB2BGR))
        
        # Save raw choice for analysis
        np.save(sample_dir / 'choice_low.npy', choice_low)
        np.save(sample_dir / 'choice_high.npy', choice_high)
        
        # Save metadata
        import json
        metadata = {
            'sample_idx': sample_idx,
            'frame_id': batch_dict.get('frame_id', ['unknown'])[0],
            'teacher_sweeps': teacher_sweeps,
            'low_ratios': [(choice_low == i).sum() / choice_low.size for i in range(len(teacher_sweeps))],
            'high_ratios': [(choice_high == i).sum() / choice_high.size for i in range(len(teacher_sweeps))],
        }
        with open(sample_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./paper_figures')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    
    # Load config
    cfg_from_yaml_file(args.cfg_file, cfg)
    
    # Build dataloader (validation set)
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=4,
        training=False
    )
    
    # Build model
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.cuda()
    model.eval()
    
    print(f"Loaded checkpoint from {args.ckpt}")
    print(f"Generating visualizations for {args.num_samples} samples...")
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Teacher sweeps
    teacher_sweeps = cfg.DATA_CONFIG.get('TEACHER_SWEEPS', [1, 5, 10])
    
    # Gating analyzer for scene statistics
    analyzer = GatingAnalyzer(teacher_sweeps)
    
    # Process samples
    sample_count = 0
    for batch_dict in tqdm(test_loader, desc="Visualizing"):
        # Move to GPU
        for key, val in batch_dict.items():
            if isinstance(val, torch.Tensor):
                batch_dict[key] = val.cuda()
        
        # Visualize
        visualize_sample(model, batch_dict, output_dir, sample_count, teacher_sweeps)
        
        # Update analyzer
        if 'alpha_low' in batch_dict:
            analyzer.update(batch_dict['alpha_low'])
        
        sample_count += 1
        if sample_count >= args.num_samples:
            break
    
    # Save scene analysis
    analyzer.save(output_dir / 'scene_statistics.json')
    analyzer.print_summary()
    
    print(f"\nDone! Visualizations saved to {output_dir}")
    print(f"Generated {sample_count} samples")
    print("\nTo create paper figures, run:")
    print(f"  python paper_visualize.py --vis_dir {output_dir}")


if __name__ == '__main__':
    main()
