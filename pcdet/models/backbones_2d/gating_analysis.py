"""
Per-Scene Gating Analysis for Validation

Usage in validation loop:
    analyzer = GatingAnalyzer()
    for batch in val_loader:
        analyzer.update(alpha, scene_tokens)
    analyzer.save('output/scene_analysis.json')
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path


class GatingAnalyzer:
    """Lightweight scene-level gating statistics collector"""
    
    def __init__(self, teacher_sweeps=[1, 5, 10]):
        self.teacher_sweeps = teacher_sweeps
        self.scene_stats = defaultdict(lambda: {'counts': [0, 0, 0], 'total': 0})
        
    def update(self, alpha, scene_tokens=None):
        """
        Update statistics from a batch
        
        Args:
            alpha: [B, N, 1, H, W] gating weights
            scene_tokens: list of scene identifiers (optional)
        """
        # Get teacher selection (argmax)
        choice = alpha.argmax(dim=1).cpu().numpy()  # [B, 1, H, W]
        
        B = choice.shape[0]
        for b in range(B):
            # Count each teacher selection
            choice_b = choice[b, 0]  # [H, W]
            counts = [(choice_b == i).sum() for i in range(len(self.teacher_sweeps))]
            total = choice_b.size
            
            # Aggregate by scene if available
            scene_key = scene_tokens[b] if scene_tokens else 'all'
            self.scene_stats[scene_key]['counts'] = [
                self.scene_stats[scene_key]['counts'][i] + int(counts[i])
                for i in range(len(self.teacher_sweeps))
            ]
            self.scene_stats[scene_key]['total'] += int(total)
    
    def get_summary(self):
        """Get summary statistics"""
        summary = {}
        for scene, stats in self.scene_stats.items():
            total = stats['total']
            if total > 0:
                ratios = [c / total for c in stats['counts']]
                summary[scene] = {
                    'teacher_sweeps': self.teacher_sweeps,
                    'counts': stats['counts'],
                    'ratios': ratios,
                    'total_pixels': total
                }
        return summary
    
    def save(self, filepath):
        """Save to JSON"""
        summary = self.get_summary()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[GatingAnalyzer] Saved scene analysis to {filepath}")
        return summary
    
    def print_summary(self):
        """Print human-readable summary"""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("Per-Scene Gating Analysis")
        print("="*60)
        for scene, stats in summary.items():
            print(f"\nScene: {scene}")
            print(f"  Total pixels: {stats['total_pixels']:,}")
            for i, sweep in enumerate(stats['teacher_sweeps']):
                ratio = stats['ratios'][i]
                count = stats['counts'][i]
                print(f"  Teacher s{sweep}: {ratio*100:.1f}% ({count:,} pixels)")
        print("="*60 + "\n")
