import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.utils import center_distance

def get_attr_value(box, attr_name, nusc=None, token=None):
    """
    Extract attribute value from a NuScenes annotation token or box.
    """
    if attr_name == 'distance':
        return np.sqrt(box.translation[0]**2 + box.translation[1]**2)
    elif attr_name == 'speed':
        return np.sqrt(box.velocity[0]**2 + box.velocity[1]**2)
    elif attr_name == 'size':
        return max(box.size) # Max dimension
    return 0

def compute_ap_for_bin(matches, gt_subset_mask, gt_ignore_mask):
    """
    matches: list of (gt_idx, match_dist) for each prediction (sorted by score). 
             gt_idx is -1 if no match.
    gt_subset_mask: boolean array of shape (num_gts,), True if GT is in the target bin.
    gt_ignore_mask: boolean array of shape (num_gts,), True if GT should be ignored (e.g. in other bins).
    """
    tp = []
    fp = []
    npos = np.sum(gt_subset_mask)
    
    # Track which GTs have been assigned
    gt_assigned = np.zeros(len(gt_subset_mask), dtype=bool)
    
    for gt_idx, dist in matches:
        if gt_idx == -1:
            # No match -> FP
            tp.append(0)
            fp.append(1)
        else:
            if gt_subset_mask[gt_idx]:
                if not gt_assigned[gt_idx]:
                    # Match to target GT -> TP
                    tp.append(1)
                    fp.append(0)
                    gt_assigned[gt_idx] = True
                else:
                    # Duplicate match to already assigned target GT -> FP
                    tp.append(0)
                    fp.append(1)
            elif gt_ignore_mask[gt_idx]:
                # Match to ignore GT -> Ignore (neither TP nor FP)
                pass
            else:
                # Match to GT that is neither target nor ignore? Should not happen if masks cover all.
                # Assume FP if not target and not ignore.
                tp.append(0)
                fp.append(1)

    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    rec = tp / max(npos, 1e-6)
    prec = tp / np.maximum(tp + fp, 1e-6)

    # 11-point AP or Area under curve? NuScenes uses 101-point interpolation, but standard is area.
    # Let's use simple VOC-style 11-point or smoothed area for simplicity/robustness.
    # Using continuous integration (VOC 2010+)
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, npos

def custom_evaluation(nusc, results, output_dir, verbose=True):
    """
    nusc: NuScenes object
    results: dict of {token: [dict(translation, size, velocity, detection_name, detection_score), ...]}
    """
    print("Starting Custom Attribute-based Evaluation...")
    
    # 1. Setup Bins
    BINS = {
        'distance': [
            (0, 20, '0-20m'),
            (20, 40, '20-40m'),
            (40, 60, '40-60m'),
            (60, 1000, '60m+')
        ],
        'speed': [
            (0, 0.5, 'Stationary'),
            (0.5, 5.0, 'Slow'),
            (5.0, 1000, 'Fast')
        ],
        'radar_pts': [
            (0, 1, '0 pts'),
            (1, 5, '1-4 pts'),
            (5, 1000, '5+ pts')
        ]
    }
    
    CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
               'pedestrian', 'motorcycle', 'bicycle', 'barrier', 'traffic_cone']
    
    DIST_TH = 2.0 # Standard NuScenes matching threshold
    
    # 2. Pre-process Data
    # Group by class
    gt_by_class = {c: [] for c in CLASSES}
    pred_by_class = {c: [] for c in CLASSES}
    
    # Iterate over all samples in validation set
    # We assume 'results' keys cover the validation set
    val_tokens = list(results.keys())
    
    for token in tqdm(val_tokens, desc="Loading Data"):
        # Load GT
        try:
            sample = nusc.get('sample', token)
        except:
            continue
            
        # Get Ego Pose for distance calculation
        sd_token = sample['data']['LIDAR_TOP']
        sd_record = nusc.get('sample_data', sd_token)
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
        ego_pos = np.array(pose_record['translation'])
        
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            cat = ann['category_name']
            # Map category to detection class
            # NuScenes category to detection name mapping
            # This is usually handled by the eval script, but we need it here.
            # Simplified mapping based on standard NuScenes classes
            cat_to_det = {
                'vehicle.car': 'car',
                'vehicle.truck': 'truck',
                'vehicle.bus.bendy': 'bus',
                'vehicle.bus.rigid': 'bus',
                'vehicle.trailer': 'trailer',
                'vehicle.construction': 'construction_vehicle',
                'human.pedestrian.adult': 'pedestrian',
                'human.pedestrian.child': 'pedestrian',
                'human.pedestrian.construction_worker': 'pedestrian',
                'human.pedestrian.police_officer': 'pedestrian',
                'vehicle.motorcycle': 'motorcycle',
                'vehicle.bicycle': 'bicycle',
                'movable_object.barrier': 'barrier',
                'movable_object.trafficcone': 'traffic_cone',
            }
            
            det_name = None
            # Try exact match first
            if cat in cat_to_det:
                det_name = cat_to_det[cat]
            else:
                # Try prefix matching if needed, or skip
                pass
                
            if det_name is None:
                continue
            if det_name not in CLASSES:
                continue
                
            # Create GT object
            gt_box = DetectionBox(
                sample_token=token,
                translation=ann['translation'],
                size=ann['size'],
                rotation=ann['rotation'],
                velocity=nusc.box_velocity(ann_token)[:2],
                ego_translation=(0,0,0) # Not needed for global matching
            )
            
            # Calculate Ego-relative distance
            gt_pos = np.array(ann['translation'])
            ego_distance = np.linalg.norm(gt_pos[:2] - ego_pos[:2])
            
            # Extract attributes
            attrs = {
                'distance': ego_distance,  # Use Ego-relative distance
                'speed': get_attr_value(gt_box, 'speed'),
                'size': get_attr_value(gt_box, 'size'),
                'radar_pts': ann['num_radar_pts']
            }
            
            gt_by_class[det_name].append({
                'box': gt_box,
                'attrs': attrs,
                'token': ann_token
            })
            
        # Load Preds
        if token in results:
            for pred in results[token]:
                if pred['detection_name'] not in CLASSES:
                    continue
                pred_box = DetectionBox(
                    sample_token=token,
                    translation=pred['translation'],
                    size=pred['size'],
                    rotation=pred['rotation'],
                    velocity=pred['velocity'],
                    ego_translation=(0,0,0),
                    detection_name=pred['detection_name'],
                    detection_score=pred['detection_score']
                )
                pred_by_class[pred['detection_name']].append(pred_box)

    # DEBUG: Print loaded counts
    print(f"\n[DEBUG] Loaded Data Summary:")
    for cls in CLASSES:
        n_gt = len(gt_by_class[cls])
        n_pred = len(pred_by_class[cls])
        if n_gt > 0 or n_pred > 0:
            print(f"  - {cls}: {n_gt} GTs, {n_pred} Preds")

    # 3. Compute AP per Bin
    summary = {}
    
    for cls in CLASSES:
        gts = gt_by_class[cls]
        preds = pred_by_class[cls]
        
        if len(gts) == 0:
            continue
            
        # Sort preds by score
        preds.sort(key=lambda x: x.detection_score, reverse=True)
        
        # Pre-compute matches (Greedy matching)
        # For each pred, find best matching GT (dist < 2.0)
        # We need to be careful: matching depends on the subset?
        # No, standard AP calculation usually matches to *any* valid GT first.
        # But here we want to ignore GTs outside the bin.
        # Strategy: Match Preds to ALL GTs first. 
        # Then, for a specific bin, check if the matched GT is in the bin.
        
        # Group by sample_token for faster matching
        gts_by_sample = {}
        for i, gt in enumerate(gts):
            t = gt['box'].sample_token
            if t not in gts_by_sample: gts_by_sample[t] = []
            gts_by_sample[t].append((i, gt['box']))
            
        matches = [] # (gt_idx, dist)
        gt_matched_global = np.zeros(len(gts), dtype=bool)
        
        # This global matching is slightly incorrect for "Ignore" logic if we want to be strict.
        # Strict way: For each bin, re-run matching where out-of-bin GTs are 'dontcare'.
        # But that's slow.
        # Approximation: Match globally. If Pred matches GT_A (dist=0.5), and GT_A is in Bin 1.
        # Then for Bin 2, GT_A is 'ignore'. Pred is matched to 'ignore' -> Pred is ignored.
        # This is the standard way "Ignore" works in benchmarks (e.g. COCO crowd).
        # So Global Matching is correct.
        
        for p in preds:
            t = p.sample_token
            best_dist = DIST_TH
            best_gt_idx = -1
            
            if t in gts_by_sample:
                for gt_idx, gt_box in gts_by_sample[t]:
                    # We allow multiple preds to match same GT? No, greedy.
                    # But for "Pre-computation", we just find nearest GT.
                    # The "Used" check happens during AP calc.
                    # WAIT. If we match globally, we might match a Pred to a GT that is "Ignored" in the current bin,
                    # preventing it from matching a "Target" GT that is slightly further away?
                    # Usually, GTs are distinct. A Pred can only match one GT.
                    # If it matches an "Ignore" GT, it consumes the Pred.
                    # So yes, global matching is fine.
                    
                    dist = center_distance(p, gt_box)
                    if dist < best_dist:
                        best_dist = dist
                        best_gt_idx = gt_idx
            
            matches.append((best_gt_idx, best_dist))
            
        # Now iterate bins
        cls_res = {}
        for attr, bins in BINS.items():
            for min_v, max_v, label in bins:
                # Create masks
                gt_subset_mask = np.zeros(len(gts), dtype=bool)
                gt_ignore_mask = np.zeros(len(gts), dtype=bool)
                
                for i, gt in enumerate(gts):
                    val = gt['attrs'][attr]
                    if min_v <= val < max_v:
                        gt_subset_mask[i] = True
                    else:
                        gt_ignore_mask[i] = True
                        
                ap, npos = compute_ap_for_bin(matches, gt_subset_mask, gt_ignore_mask)
                cls_res[f"{attr}_{label}"] = (ap, npos)
                
        summary[cls] = cls_res

    # 4. Print & Save
    print("\n" + "="*80)
    print("CUSTOM ATTRIBUTE ANALYSIS (AP @ 2.0m)")
    print("="*80)
    
    # Print Table Header
    # We will print one table per Attribute Type
    
    for attr in BINS.keys():
        print(f"\n[ Attribute: {attr.upper()} ]")
        headers = [b[2] for b in BINS[attr]]
        print(f"{'Class':<15} | " + " | ".join([f"{h:<10}" for h in headers]))
        print("-" * (15 + 13 * len(headers)))
        
        for cls in CLASSES:
            if cls not in summary: continue
            row = f"{cls:<15} | "
            for _, _, label in BINS[attr]:
                key = f"{attr}_{label}"
                if key in summary[cls]:
                    ap, npos = summary[cls][key]
                    if npos > 0:
                        row += f"{ap*100:5.1f} ({npos}) | "
                    else:
                        row += f"{'-':^10} | "
                else:
                    row += f"{'-':^10} | "
            print(row)
            
    # Save to JSON
    out_file = Path(output_dir) / 'custom_eval_metrics.json'
    with open(out_file, 'w') as f:
        # Convert numpy types
        json_summary = {}
        for cls, data in summary.items():
            json_summary[cls] = {k: (float(v[0]), int(v[1])) for k, v in data.items()}
        json.dump(json_summary, f, indent=2)
    print(f"\nSaved custom analysis to {out_file}")
