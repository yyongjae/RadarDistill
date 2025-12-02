import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Model configurations
MODELS = {
    'Teacher (S10)': '/home/yongjae/4drkd/RadarDistill/output/teacher_s10/eval/final_result/data/teacher_s10_custom_eval_metrics.json',
    'Teacher (S1)':  '/home/yongjae/4drkd/RadarDistill/output/teacher_s1/eval/final_result/data/teacher_s1_custom_eval_metrics.json',
    'Baseline':      '/home/yongjae/4drkd/RadarDistill/output/baseline/eval/final_result/data/baseline_custom_eval_metrics.json',
    'Student (S1)': '/home/yongjae/4drkd/RadarDistill/output/sweep1_base/eval/final_result/data/s1_custom_eval_metrics.json',
    'Student (S1_80)': '/home/yongjae/4drkd/RadarDistill/output/sweep1_80_base/eval/final_result/data/s1_80_custom_eval_metrics.json',
    'Student (S1_50)': '/home/yongjae/4drkd/RadarDistill/output/sweep1_50/eval/final_result/data/s1_50_custom_eval_metrics.json',
}

CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
           'pedestrian', 'motorcycle', 'bicycle', 'barrier', 'traffic_cone']

ATTRIBUTES = {
    'distance': ['0-20m', '20-40m', '40-60m', '60m+'],
    'speed': ['Stationary', 'Slow', 'Fast'],
    'radar_pts': ['0 pts', '1-4 pts', '5+ pts']
}

def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_ap_values(metrics, cls, attr_type):
    """Extract AP values for a specific class and attribute type"""
    bins = ATTRIBUTES[attr_type]
    ap_values = []
    
    for bin_label in bins:
        key = f"{attr_type}_{bin_label}"
        if cls in metrics and key in metrics[cls]:
            ap_value = metrics[cls][key][0] * 100  # Convert to percentage
            ap_values.append(ap_value)
        else:
            ap_values.append(0.0)
    
    return ap_values

def plot_attribute_comparison(models_data, attr_type, save_dir):
    """Plot comparison for a specific attribute across all classes"""
    bins = ATTRIBUTES[attr_type]
    n_bins = len(bins)
    n_models = len(models_data)
    
    # Create subplots for each class
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    fig.suptitle(f'AP Comparison by {attr_type.upper()}', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, cls in enumerate(CLASSES):
        ax = axes[idx]
        
        # Prepare data
        x = np.arange(n_bins)
        width = 0.15
        
        # Plot bars for each model
        for i, (model_name, metrics) in enumerate(models_data.items()):
            ap_values = extract_ap_values(metrics, cls, attr_type)
            offset = (i - n_models/2 + 0.5) * width
            ax.bar(x + offset, ap_values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Bins', fontsize=10)
        ax.set_ylabel('AP (%)', fontsize=10)
        ax.set_title(cls.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f'comparison_{attr_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_overall_comparison(models_data, save_dir):
    """Plot overall AP comparison across all classes"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Overall AP Comparison Across Attributes', fontsize=16, fontweight='bold')
    
    for attr_idx, (attr_type, bins) in enumerate(ATTRIBUTES.items()):
        ax = axes[attr_idx]
        
        # Calculate mean AP for each model across all classes
        model_means = {}
        for model_name, metrics in models_data.items():
            all_aps = []
            for cls in CLASSES:
                ap_values = extract_ap_values(metrics, cls, attr_type)
                all_aps.extend(ap_values)
            model_means[model_name] = np.mean(all_aps)
        
        # Plot
        models = list(model_means.keys())
        means = list(model_means.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars = ax.bar(models, means, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Mean AP (%)', fontsize=12)
        ax.set_title(f'{attr_type.upper()}', fontsize=14, fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = save_dir / 'overall_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_class_wise_summary(models_data, save_dir):
    """Plot class-wise summary showing best performance per attribute"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Prepare data matrix: classes x (attributes * bins)
    all_bins = []
    bin_labels = []
    for attr_type, bins in ATTRIBUTES.items():
        for bin_label in bins:
            all_bins.append((attr_type, bin_label))
            bin_labels.append(f"{attr_type}\n{bin_label}")
    
    n_bins = len(all_bins)
    n_classes = len(CLASSES)
    
    # Create heatmap data for best model per class-bin combination
    heatmap_data = np.zeros((n_classes, n_bins))
    
    for cls_idx, cls in enumerate(CLASSES):
        for bin_idx, (attr_type, bin_label) in enumerate(all_bins):
            # Find best AP among all models
            best_ap = 0
            for model_name, metrics in models_data.items():
                key = f"{attr_type}_{bin_label}"
                if cls in metrics and key in metrics[cls]:
                    ap_value = metrics[cls][key][0] * 100
                    best_ap = max(best_ap, ap_value)
            heatmap_data[cls_idx, bin_idx] = best_ap
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(bin_labels, rotation=90, ha='center', fontsize=9)
    ax.set_yticklabels([c.replace('_', ' ').title() for c in CLASSES], fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Best AP (%)', rotation=270, labelpad=20, fontsize=12)
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_bins):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title('Best AP Performance Heatmap (Across All Models)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_path = save_dir / 'best_performance_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def generate_summary_table(models_data, save_dir):
    """Generate summary statistics table"""
    summary = []
    
    for model_name, metrics in models_data.items():
        row = {'Model': model_name}
        
        for attr_type in ATTRIBUTES.keys():
            all_aps = []
            for cls in CLASSES:
                ap_values = extract_ap_values(metrics, cls, attr_type)
                all_aps.extend(ap_values)
            
            row[f'{attr_type}_mean'] = np.mean(all_aps)
            row[f'{attr_type}_std'] = np.std(all_aps)
        
        summary.append(row)
    
    # Save to text file
    save_path = save_dir / 'summary_statistics.txt'
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        for row in summary:
            f.write(f"Model: {row['Model']}\n")
            f.write("-" * 60 + "\n")
            for attr_type in ATTRIBUTES.keys():
                mean = row[f'{attr_type}_mean']
                std = row[f'{attr_type}_std']
                f.write(f"  {attr_type.upper():<15} Mean: {mean:6.2f}%  Std: {std:6.2f}%\n")
            f.write("\n")
    
    print(f"Saved: {save_path}")

def main():
    # Create output directory
    output_dir = Path('/home/yongjae/4drkd/RadarDistill/output/visualization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading metrics...")
    models_data = {}
    for model_name, filepath in MODELS.items():
        models_data[model_name] = load_metrics(filepath)
        print(f"  Loaded: {model_name}")
    
    print("\nGenerating visualizations...")
    
    # Plot comparisons for each attribute
    for attr_type in ATTRIBUTES.keys():
        print(f"  Creating {attr_type} comparison...")
        plot_attribute_comparison(models_data, attr_type, output_dir)
    
    # Plot overall comparison
    print("  Creating overall comparison...")
    plot_overall_comparison(models_data, output_dir)
    
    # Plot class-wise summary
    print("  Creating class-wise heatmap...")
    plot_class_wise_summary(models_data, output_dir)
    
    # Generate summary table
    print("  Generating summary statistics...")
    generate_summary_table(models_data, output_dir)
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - comparison_distance.png")
    print("  - comparison_speed.png")
    print("  - comparison_radar_pts.png")
    print("  - overall_comparison.png")
    print("  - best_performance_heatmap.png")
    print("  - summary_statistics.txt")

if __name__ == '__main__':
    main()
