"""
Generate paper figures from saved data

Usage:
    python paper_visualize.py --vis_dir ./output/visualizations
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_gating_evolution(json_path, save_path):
    """Plot teacher selection ratio evolution over epochs"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    epochs = [d['epoch'] for d in data]
    teacher_sweeps = data[0]['teacher_sweeps']
    
    # Extract ratios for each teacher (low resolution)
    ratios = {f's{sw}': [] for sw in teacher_sweeps}
    for d in data:
        for i, sw in enumerate(teacher_sweeps):
            ratios[f's{sw}'].append(d['low_ratios'][i])
    
    # Plot
    plt.figure(figsize=(10, 6))
    for sw_key in ratios:
        plt.plot(epochs, ratios[sw_key], marker='o', label=f'Teacher {sw_key}', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Selection Ratio', fontsize=12)
    plt.title('Teacher Selection Evolution During Training', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_scene_analysis(json_path, save_path):
    """Plot per-scene teacher preference"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    scenes = list(data.keys())
    teacher_sweeps = data[scenes[0]]['teacher_sweeps']
    
    # Prepare data for grouped bar chart
    x = np.arange(len(scenes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, sw in enumerate(teacher_sweeps):
        ratios = [data[scene]['ratios'][i] * 100 for scene in scenes]
        ax.bar(x + i * width, ratios, width, label=f's{sw}')
    
    ax.set_xlabel('Scene', fontsize=12)
    ax.set_ylabel('Selection Ratio (%)', fontsize=12)
    ax.set_title('Teacher Preference by Scene Type', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenes, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_dir', type=str, default='./output/visualizations')
    args = parser.parse_args()
    
    vis_dir = Path(args.vis_dir)
    
    # 1. Gating Evolution
    evolution_file = vis_dir / 'gating_evolution.json'
    if evolution_file.exists():
        plot_gating_evolution(
            evolution_file,
            vis_dir / 'paper_gating_evolution.png'
        )
    else:
        print(f"Warning: {evolution_file} not found")
    
    # 2. Scene Analysis
    scene_file = vis_dir / 'scene_analysis.json'
    if scene_file.exists():
        plot_scene_analysis(
            scene_file,
            vis_dir / 'paper_scene_analysis.png'
        )
    else:
        print(f"Warning: {scene_file} not found (run validation first)")
    
    print("\nDone! Check ./output/visualizations/ for paper figures")


if __name__ == '__main__':
    main()
