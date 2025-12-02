import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# 1. 경로 및 설정
# ---------------------------------------------------------

MODELS = {
    'Teacher_S10':   '/home/yongjae/4drkd/RadarDistill/output/teacher_s10/eval/final_result/data/teacher_s10_custom_eval_metrics.json',
    'Teacher_S1':    '/home/yongjae/4drkd/RadarDistill/output/teacher_s1/eval/final_result/data/teacher_s1_custom_eval_metrics.json',
    'Student_naive': '/home/yongjae/4drkd/RadarDistill/output/student_naive/eval/final_result/data/s_naive_custom_eval_metrics.json',
    'Student_S10':   '/home/yongjae/4drkd/RadarDistill/output/baseline/eval/final_result/data/baseline_custom_eval_metrics.json',
    'Student_S1':    '/home/yongjae/4drkd/RadarDistill/output/sweep1_base/eval/final_result/data/s1_custom_eval_metrics.json',
    'Student_S1_80': '/home/yongjae/4drkd/RadarDistill/output/sweep1_80_base/eval/final_result/data/s1_80_custom_eval_metrics.json',
    'Student_S1_50': '/home/yongjae/4drkd/RadarDistill/output/sweep1_50/eval/final_result/data/s1_50_custom_eval_metrics.json',
}

CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'barrier', 'traffic_cone']

ATTR_BINS = {
    'radar_pts': ['0 pts', '1-4 pts', '5+ pts'],
}

def load_metrics(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_ap(metrics, cls, attr_type, bin_label):
    """metrics[cls][f'{attr_type}_{bin_label}'][0] * 100, 없으면 np.nan"""
    key = f'{attr_type}_{bin_label}'
    if cls in metrics and key in metrics[cls]:
        return metrics[cls][key][0] * 100.0
    return np.nan

def plot_gap_gain_gpg_radar_pts(
    teacher_name='Teacher_S1',
    naive_name='Student_naive',
    distilled_name='Student_S1',
    save_dir='/home/yongjae/4drkd/RadarDistill/output/visualization'
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) 메트릭 로드
    metrics_teacher   = load_metrics(MODELS[teacher_name])
    metrics_naive     = load_metrics(MODELS[naive_name])
    metrics_distilled = load_metrics(MODELS[distilled_name])

    attr_type = 'radar_pts'
    bins = ATTR_BINS[attr_type]

    # 2) 시각화 설정
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(
        f'Gap–Gain–GpG by {attr_type}\n'
        f'Teacher={teacher_name}, Naive={naive_name}, Distilled={distilled_name}',
        fontsize=14, fontweight='bold'
    )

    # 색: 클래스
    cmap = plt.cm.get_cmap('tab10', len(CLASSES))
    class2color = {cls: cmap(i) for i, cls in enumerate(CLASSES)}

    # 마커: bin
    bin2marker = {
        '0 pts': 'o',
        '1-4 pts': 's',
        '5+ pts': '^',
    }

    # 범례용 핸들 저장
    class_handles = {}
    bin_handles = {}

    # 3) 점 찍기
    for cls_idx, cls in enumerate(CLASSES):
        for bin_label in bins:
            ap_t = get_ap(metrics_teacher, cls, attr_type, bin_label)
            ap_n = get_ap(metrics_naive, cls, attr_type, bin_label)
            ap_d = get_ap(metrics_distilled, cls, attr_type, bin_label)

            # 유효하지 않은 경우 skip
            if np.isnan(ap_t) or np.isnan(ap_n) or np.isnan(ap_d):
                continue

            gap = ap_t - ap_n
            gain = ap_d - ap_n

            # gap=0이면 GpG 정의 안 되므로 스킵 (또는 gpg=0 처리)
            if abs(gap) < 1e-6:
                continue
            gpg = 100.0 * gain / gap  # Gain per Gap (%)

            # 마커 크기에 GpG 반영 (절댓값 기준)
            base_size = 40
            size = base_size + 1.0 * abs(gpg)  # 적당히 조절

            color = class2color[cls]
            marker = bin2marker[bin_label]

            sc = ax.scatter(
                gap, gain,
                s=size,
                color=color,
                marker=marker,
                alpha=0.8,
                edgecolor='k',
                linewidth=0.5
            )

            # 클래스 범례용 핸들 하나만 저장
            if cls not in class_handles:
                class_handles[cls] = sc

            # bin 범례용 핸들 하나만 저장
            if bin_label not in bin_handles:
                bin_handles[bin_label] = plt.Line2D(
                    [], [], color='black', marker=marker,
                    linestyle='None', markersize=8, label=bin_label
                )

            # 아주 눈에 띄는 점들만 라벨링 (예: GpG>40% or Gain>|5|)
            if abs(gain) > 5 or gpg > 40:
                short_cls = cls[:3]  # car, tru, bus ...
                ax.text(
                    gap, gain,
                    f'{short_cls}|{bin_label}',
                    fontsize=7,
                    ha='center', va='bottom'
                )

    # 4) 축 / 기준선 / 라벨
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    # y=x 보조선 (100% GpG)
    lim = max(abs(ax.get_xlim()[0]), ax.get_xlim()[1],
              abs(ax.get_ylim()[0]), ax.get_ylim()[1])
    ax.plot([-lim, lim], [-lim, lim], 'k:', linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_xlabel('Gap = AP(Teacher) − AP(Student_naive)  [%]', fontsize=11)
    ax.set_ylabel('Gain = AP(Distilled) − AP(Student_naive)  [%]', fontsize=11)

    # 5) 범례: 클래스 / radar_pts bin 분리
    # 클래스 범례 (색)
    class_legend = ax.legend(
        class_handles.values(), [c for c in class_handles.keys()],
        title='Class (color)', loc='upper left', fontsize=8
    )
    ax.add_artist(class_legend)

    # bin 범례 (마커)
    bin_legend = ax.legend(
        bin_handles.values(), [b for b in bin_handles.keys()],
        title='Radar pts (marker)', loc='lower right', fontsize=8
    )

    ax.grid(True, alpha=0.3)

    # 6) 저장
    save_path = save_dir / f'gap_gain_gpg_{attr_type}_{teacher_name}_{distilled_name}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

if __name__ == '__main__':
    plot_gap_gain_gpg_radar_pts(
        teacher_name='Teacher_S1',
        naive_name='Student_naive',
        distilled_name='Student_S1',
        save_dir='/home/yongjae/4drkd/RadarDistill/output/visualization'
    )