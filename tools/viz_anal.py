import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# 1. 설정
# ============================================================

MODELS = {
    'Teacher (S10)': '/home/yongjae/4drkd/RadarDistill/output/teacher_s10/eval/final_result/data/teacher_s10_custom_eval_metrics.json',
    'Teacher (S1)':  '/home/yongjae/4drkd/RadarDistill/output/teacher_s1/eval/final_result/data/teacher_s1_custom_eval_metrics.json',
    'Baseline':      '/home/yongjae/4drkd/RadarDistill/output/baseline/eval/final_result/data/baseline_custom_eval_metrics.json',
    'Student (S1)': '/home/yongjae/4drkd/RadarDistill/output/sweep1_base/eval/final_result/data/s1_custom_eval_metrics.json',
    'Student (S1_80)': '/home/yongjae/4drkd/RadarDistill/output/sweep1_80_base/eval/final_result/data/s1_80_custom_eval_metrics.json',
    'Student (S1_50)': '/home/yongjae/4drkd/RadarDistill/output/sweep1_50/eval/final_result/data/s1_50_custom_eval_metrics.json',
}

# nuScenes 클래스 (radar 관련 10개)
CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'barrier', 'traffic_cone'
]

# Attribute 정의 (키 이름은 json의 prefix와 동일하게)
ATTRIBUTES = {
    'distance': ['0-20m', '20-40m', '40-60m', '60m+'],
    'speed':    ['Stationary', 'Slow', 'Fast'],
    'radar_pts': ['0 pts', '1-4 pts', '5+ pts'],
}

# “대표 클래스” (per-class 분석용)
FOCUS_CLASSES = ['car', 'pedestrian', 'motorcycle', 'bicycle']


# ============================================================
# 2. 유틸 함수
# ============================================================

def load_metrics(filepath):
    """JSON metric 파일 로드."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_ap_for_bin(metrics, cls, attr_type, bin_label):
    """
    한 모델의 metrics에서
    특정 class + attr_type + bin에 대한 AP(%) 를 가져옴.
    없으면 None 리턴.
    """
    key = f"{attr_type}_{bin_label}"
    if cls in metrics and key in metrics[cls]:
        ap = metrics[cls][key][0] * 100.0  # 0~1 -> %
        return ap
    return None


def compute_mean_ap_over_classes(metrics, classes, attr_type):
    """
    한 모델에 대해,
    attr_type의 각 bin마다 클래스 평균 AP(%) 계산.
    - classes: 평균을 낼 클래스 리스트
    - 반환: [bin1_mean, bin2_mean, ...]
    """
    bins = ATTRIBUTES[attr_type]
    mean_aps = []

    for bin_label in bins:
        vals = []
        for cls in classes:
            ap = extract_ap_for_bin(metrics, cls, attr_type, bin_label)
            if ap is not None:
                vals.append(ap)
        if len(vals) == 0:
            mean_aps.append(0.0)
        else:
            mean_aps.append(float(np.mean(vals)))
    return mean_aps


def compute_ap_per_class(metrics, cls, attr_type):
    """
    한 모델에 대해,
    특정 class + attr_type의 각 bin AP(%)를 리스트로 반환.
    """
    bins = ATTRIBUTES[attr_type]
    vals = []
    for bin_label in bins:
        ap = extract_ap_for_bin(metrics, cls, attr_type, bin_label)
        vals.append(ap if ap is not None else 0.0)
    return vals


# ============================================================
# 3. 시각화 함수
# ============================================================

def plot_global_attr_curves(models_data, attr_type, save_dir):
    """
    [그림 1 스타일] 각 attribute에 대해
    - x축: bin
    - y축: 클래스 평균 AP(%)
    - 여러 모델의 곡선 한꺼번에
    """
    bins = ATTRIBUTES[attr_type]
    x = np.arange(len(bins))

    plt.figure(figsize=(8, 5))
    for model_name, metrics in models_data.items():
        mean_aps = compute_mean_ap_over_classes(metrics, CLASSES, attr_type)
        plt.plot(x, mean_aps, marker='o', linewidth=2, label=model_name)

    plt.xticks(x, bins, rotation=0)
    plt.ylabel('Mean AP (%)')
    plt.xlabel(attr_type.capitalize())
    plt.title(f'Mean AP vs {attr_type.capitalize()} (All classes)')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = save_dir / f'global_{attr_type}_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_class_attr_curves(models_data, attr_type, save_dir, focus_classes=None):
    """
    [그림 2 스타일] attr_type 하나에 대해
    여러 클래스를 2x2 혹은 2xN subplot으로 배치.
    각 subplot에는 (1) x축: bin (2) y축: AP (%)
    여러 모델의 곡선이 겹쳐서 나옴.
    """
    if focus_classes is None:
        focus_classes = FOCUS_CLASSES

    bins = ATTRIBUTES[attr_type]
    x = np.arange(len(bins))

    n_cls = len(focus_classes)
    n_cols = 2
    n_rows = int(np.ceil(n_cls / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes = np.array(axes).reshape(-1)  # flatten

    fig.suptitle(f'{attr_type.capitalize()}-wise AP for Selected Classes', fontsize=16)

    for idx, cls in enumerate(focus_classes):
        ax = axes[idx]

        for model_name, metrics in models_data.items():
            vals = compute_ap_per_class(metrics, cls, attr_type)
            ax.plot(x, vals, marker='o', linewidth=2, label=model_name)

        ax.set_title(cls.replace('_', ' ').title())
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=0)
        ax.set_ylabel('AP (%)')
        ax.set_xlabel(attr_type.capitalize())
        ax.grid(axis='y', alpha=0.3)

        # 첫 번째 subplot에만 legend
        if idx == 0:
            ax.legend(fontsize=9)

    # 남는 subplot은 숨기기
    for j in range(n_cls, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = save_dir / f'classwise_{attr_type}_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_teacher_student_gap(models_data, attr_type, save_dir):
    """
    [그림 3 스타일]
    Teacher(S10), Teacher(S1), Baseline, Student(E80/E50)를
    각 bin마다 하나의 막대 묶음으로 그려서
    teacher gap과 student gap의 스케일 차이를 직관적으로 보여주는 그림.
    (전체 클래스 평균 기준)
    """
    bins = ATTRIBUTES[attr_type]
    x = np.arange(len(bins))

    # 관심 모델 순서 고정
    order = ['Teacher (S10)', 'Teacher (S1)', 'Baseline', 'Student (E80)', 'Student (E50)']
    order = [m for m in order if m in models_data]

    width = 0.12
    plt.figure(figsize=(10, 5))

    for i, model_name in enumerate(order):
        metrics = models_data[model_name]
        mean_aps = compute_mean_ap_over_classes(metrics, CLASSES, attr_type)
        offset = (i - (len(order) - 1) / 2.0) * width
        plt.bar(x + offset, mean_aps, width=width, label=model_name)

    plt.xticks(x, bins)
    plt.ylabel('Mean AP (%)')
    plt.xlabel(attr_type.capitalize())
    plt.title(f'Teacher vs Student Gap by {attr_type.capitalize()}')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = save_dir / f'gap_{attr_type}_bars.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 4. main
# ============================================================

def main():
    output_dir = Path('/home/yongjae/4drkd/RadarDistill/output/visualization_v2')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics...")
    models_data = {}
    for model_name, filepath in MODELS.items():
        p = Path(filepath)
        if not p.exists():
            print(f"[WARN] File not found: {filepath}")
            continue
        models_data[model_name] = load_metrics(filepath)
        print(f"  Loaded: {model_name}")

    print("\nGenerating intuitive visualizations...\n")

    # 1) Global curves: 전체 클래스 평균 AP vs (distance/speed/radar_pts)
    for attr_type in ATTRIBUTES.keys():
        print(f"  - Global curves for {attr_type}")
        plot_global_attr_curves(models_data, attr_type, output_dir)

    # 2) Class-wise curves: 대표 클래스 4개(car, ped, mc, bicycle) x attribute
    for attr_type in ATTRIBUTES.keys():
        print(f"  - Class-wise curves for {attr_type}")
        plot_class_attr_curves(models_data, attr_type, output_dir, focus_classes=FOCUS_CLASSES)

    # 3) Teacher vs Student gap bar chart: teacher gap vs student gap의 스케일 비교
    for attr_type in ATTRIBUTES.keys():
        print(f"  - Teacher-Student gap bars for {attr_type}")
        plot_teacher_student_gap(models_data, attr_type, output_dir)

    print("\n✓ All visualizations saved to:", output_dir)


if __name__ == '__main__':
    main()
