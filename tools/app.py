# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from io import BytesIO
from pathlib import Path
import pickle
from typing import Optional, Dict, Tuple, List, Union

st.set_page_config(page_title="BEV Feature & BBoxes Viewer (3×3)", layout="wide")

# ======================
# 기본 경로/구성
# ======================
DEFAULTS = {
    "student_result": "/home/byounghun/RadarDistill/output/radar_distill/radar_distill_val_student/default/eval/epoch_no_number/val/default/result.pkl",
    "baseline_result": "/home/byounghun/RadarDistill/output/radar_distill/radar_distill_val/default/eval/epoch_no_number/val/default/result.pkl",
    "teacher_result": "/home/byounghun/RadarDistill/output/nuscenes_models/pillarnet/default/eval/epoch_no_number/val/default/result.pkl",
    "gt_annos": "/home/byounghun/RadarDistill/output/radar_distill/radar_distill_val_student/default/eval/epoch_no_number/val/default/gt_annos_list.pkl",
    "base_root": "vis_max_feat"  # 또는 "vis_mean_feat"
}

# 열(모델): sskd 제외 → 3개
MODEL_ORDER = ["student", "baseline", "teacher"]

# 행(피처): 3개
ROW_SUBFOLDERS = [
    "low_radar_bev",
    "low_radar_de_8x",
    "high_radar_bev_8x",
    "high_radar_bev",
]
# teacher 매핑
TEACHER_MAP = {
    "low_radar_bev":      "low_lidar_bev",
    "low_radar_de_8x":    "low_lidar_bev",
    "high_radar_bev_8x":  "high_lidar_bev_8x",
    "high_radar_bev":     "high_lidar_bev",
}

# ======================
# 캐시/유틸
# ======================
@st.cache_data(show_spinner=False)
def load_pickle(path: str):
    with open(Path(path), "rb") as f:
        return pickle.load(f)

def index_to_name(idx: int) -> str:
    return f"{idx:06d}.png"

def find_image_in(root: Path, subfolder: str, fname: str) -> Optional[Path]:
    base = root / subfolder
    if not base.exists():
        return None
    # 첫 매치 반환
    for p in base.rglob(fname):
        return p
    return None
@st.cache_data(show_spinner=False)
def load_index_list(path: str):
    idxs = []
    try:
        with open(Path(path), "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    idxs.append(int(line))
    except Exception as e:
        st.warning(f"Index file load failed ({path}): {e}")
    return idxs
# ---- 박스/좌표 유틸 ----
def bev_corners_xy(box):
    # [x,y,z,dx,dy,dz,yaw] → (4,2)
    x, y, dx, dy, yaw = float(box[0]), float(box[1]), float(box[3]), float(box[4]), float(box[6])
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[ dx/2,  dy/2],
                      [ dx/2, -dy/2],
                      [-dx/2, -dy/2],
                      [-dx/2,  dy/2]], dtype=np.float32)
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    return (local @ R.T) + np.array([x, y], dtype=np.float32)

def world_to_pixels_flexible(
    xy, x_lim, y_lim, W, H,
    swap_xy: bool = False,
    flip_u: bool = False,
    flip_v: bool = True,
    rotate90: int = 0
):
    xmin, xmax = x_lim
    ymin, ymax = y_lim
    u = (xy[..., 0] - xmin) / max(xmax - xmin, 1e-8)
    v = (xy[..., 1] - ymin) / max(ymax - ymin, 1e-8)

    if swap_xy:
        u, v = v, u
    if flip_u:
        u = 1.0 - u
    if flip_v:
        v = 1.0 - v

    # 회전(정규화 좌표계)
    if rotate90 % 360 == 90:
        u, v = v, 1.0 - u
    elif rotate90 % 360 == 180:
        u, v = 1.0 - u, 1.0 - v
    elif rotate90 % 360 == 270:
        u, v = 1.0 - v, u

    U = u * (W - 1)
    V = v * (H - 1)
    return np.stack([U, V], axis=-1)

def draw_boxes_on_ax(ax, boxes, x_lim, y_lim, img_shape, color,
                     linewidth=1.2, max_boxes=None,
                     swap_xy=False, flip_u=False, flip_v=True, rotate90=0, zorder=None, alpha=1.0):
    if boxes is None or len(boxes) == 0:
        return 0
    H, W = img_shape
    n = int(min(len(boxes), max_boxes)) if max_boxes is not None else len(boxes)
    for i in range(n):
        poly_xy = bev_corners_xy(boxes[i])
        poly_uv = world_to_pixels_flexible(poly_xy, x_lim, y_lim, W, H,
                                           swap_xy=swap_xy, flip_u=flip_u, flip_v=flip_v, rotate90=rotate90)
        patch = patches.Polygon(poly_uv, closed=True, fill=False,
                                edgecolor=color, linewidth=linewidth, zorder=zorder, alpha=alpha)
        ax.add_patch(patch)
    return n

def pred_for_idx(det_annos: list, idx: int, score_th: float):
    if not (0 <= idx < len(det_annos)):
        return None, None, None, None
    rec = det_annos[idx]
    fid   = rec.get("frame_id")
    names = list(rec.get("name", []))
    scores= np.asarray(rec.get("score", []), dtype=float)
    boxes = np.asarray(rec.get("boxes_lidar", rec.get("boxes", [])), dtype=float)
    n = min(len(names), len(scores), boxes.shape[0])
    names = names[:n]; scores = scores[:n]; boxes = boxes[:n]
    keep = scores >= score_th
    return fid, boxes[keep], scores[keep], [n for n,k in zip(names, keep) if k]

def gt_for_frame(gt_annos: Union[list, dict], frame_id: str):
    if frame_id is None:
        return None
    if isinstance(gt_annos, dict):
        grec = gt_annos.get(str(frame_id))
    else:
        grec = next((r for r in gt_annos if r.get("frame_id") == frame_id), None)
    if grec is None:
        return None
    return np.asarray(grec.get("gt_boxes_lidar", []), dtype=float)

# ======================
# 사이드바 컨트롤
# ======================
st.sidebar.header("Controls")
st.sidebar.header("Scene filter")
split = st.sidebar.radio("Use subset", ["All", "Day", "Rain", "Night"], index=0, horizontal=True)

# 인덱스 파일 경로 입력(기본값은 /mnt/data 경로로 세팅)
day_idx_path   = st.sidebar.text_input("day indices .txt", "day_samples_idx.txt")
rain_idx_path  = st.sidebar.text_input("rain indices .txt", "rain_samples_idx.txt")
night_idx_path = st.sidebar.text_input("night indices .txt", "night_samples_idx.txt")

# split 로드
split_map = {
    "Day":   load_index_list(day_idx_path),
    "Rain":  load_index_list(rain_idx_path),
    "Night": load_index_list(night_idx_path),
}

# --- 랜덤 콜백: 위젯 값을 안전하게 갱신 ---
def _set_random_k(key: str, n_sub: int):
    import random
    st.session_state[key] = random.randint(1, n_sub)  # 1-based

# 전역 idx 결정
if split == "All":
    idx = st.sidebar.number_input("Global idx (1-based)", min_value=1, value=1, step=1)
    st.sidebar.caption("Using full set")
else:
    subset = split_map.get(split, [])
    n_sub  = len(subset)
    if n_sub == 0:
        st.sidebar.error(f"No indices found for {split}. Falling back to global idx.")
        idx = st.sidebar.number_input("Global idx (1-based)", min_value=1, value=1, step=1)
    else:
        # --- 상태키 (split별로 독립) ---
        k_key = f"{split.lower()}_k_1based"

        # 위젯 생성 전에 기본값만 세팅 (한 번만)
        if k_key not in st.session_state:
            st.session_state[k_key] = 1  # 1-based default

        c1, c2 = st.sidebar.columns([2,1])

        # number_input: 항상 1-based만 노출
        with c1:
            st.number_input(
                f"{split} sample (1 ~ {n_sub})",
                min_value=1, max_value=n_sub, step=1,
                key=k_key
            )

        # 랜덤 버튼: 콜백으로 세션 값 갱신 (위젯과 충돌 없음)
        with c2:
            st.button(
                f"🎲 Random {split}",
                use_container_width=True,
                on_click=_set_random_k,
                args=(k_key, n_sub)
            )

        # 최종 선택(1-based) -> txt(0-based) -> global(1-based)
        k_1   = int(st.session_state[k_key])  # 1..n_sub
        raw_0 = subset[k_1 - 1]               # txt의 0-based 값
        idx   = raw_0 + 1                     # feature/gt의 1-based 전역 인덱스
        fname = index_to_name(idx)
        
        st.sidebar.caption(
            f"{split} set size: {n_sub}\n"
            f"selected: {k_1} / {n_sub}\n"
            f"global idx: {idx} ({fname})"
        )

score_th = st.sidebar.slider("Score threshold", 0.0, 1.0, 0.3, 0.01)
show_pred = st.sidebar.checkbox("Show Pred boxes (red)", value=True)
show_gt   = st.sidebar.checkbox("Show GT boxes (green)", value=True)

pred_lw = st.sidebar.slider("Pred linewidth", 0.5, 5.0, 1.0, 0.1)
gt_lw   = st.sidebar.slider("GT linewidth", 0.5, 5.0, 1.2, 0.1)

pred_alpha = st.sidebar.slider("Pred alpha", 0.1, 1.0, 1.0, 0.05)
gt_alpha   = st.sidebar.slider("GT alpha", 0.1, 1.0, 1.0, 0.05)

base_root = st.sidebar.selectbox("Feature root", options=["vis_max_feat", "vis_mean_feat"], index=0)
x_min = st.sidebar.number_input("x_min (m)", value=-50.0)
x_max = st.sidebar.number_input("x_max (m)", value=50.0)
y_min = st.sidebar.number_input("y_min (m)", value=-50.0)
y_max = st.sidebar.number_input("y_max (m)", value=50.0)

with st.sidebar.expander("Advanced (axis alignment)"):
    swap_xy  = st.checkbox("swap_xy (swap x/y)", value=False)
    flip_u   = st.checkbox("flip_u (mirror LR)", value=False)
    flip_v   = st.checkbox("flip_v (mirror UD)", value=False)
    rotate90 = st.selectbox("rotate90", options=[0, 90, 180, 270], index=0)
    norm_mode = st.selectbox("Normalization", options=["per_row", "global"], index=0)

st.sidebar.header("Paths")
student_result = st.sidebar.text_input("student result.pkl", DEFAULTS["student_result"])
baseline_result = st.sidebar.text_input("baseline result.pkl", DEFAULTS["baseline_result"])
teacher_result = st.sidebar.text_input("teacher result.pkl", DEFAULTS["teacher_result"])
gt_path = st.sidebar.text_input("gt_annos_list.pkl or gt_map.pkl", DEFAULTS["gt_annos"])

# ======================
# 데이터 로드
# ======================
try:
    student_det = load_pickle(student_result)
    baseline_det = load_pickle(baseline_result)
    teacher_det = load_pickle(teacher_result)
    gt_annos = load_pickle(gt_path)
except Exception as e:
    st.error(f"Pickle load error: {e}")
    st.stop()

DET_BY_MODEL = {
    "student":  student_det,
    "baseline": baseline_det,
    "teacher":  teacher_det,
}

ROOTS = [Path(base_root) / m for m in MODEL_ORDER]
ROOT_NAMES_ORDERED = MODEL_ORDER

# ======================
# 그리기
# ======================
st.title("BEV Feature + Boxes")
st.caption("Rows: low_radar_bev / low_radar_de_8x / high_radar_bev_8x / high_radar_bev | Cols: student / baseline / teacher")

target = index_to_name(idx)

# 1) 이미지 로드
images: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
for sub in ROW_SUBFOLDERS:
    for root_name, root in zip(ROOT_NAMES_ORDERED, ROOTS):
        sf = TEACHER_MAP[sub] if root_name == "teacher" else sub
        path = find_image_in(root, sf, target)
        if path is not None:
            arr = mpimg.imread(path)
            if arr.ndim == 3:
                arr = arr[..., 0]
            images[(sub, root_name)] = arr
        else:
            images[(sub, root_name)] = None

# 2) 정규화 범위
def get_minmax(arr_list: List[np.ndarray]):
    vmin = min(a.min() for a in arr_list)
    vmax = max(a.max() for a in arr_list)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-8
    return float(vmin), float(vmax)

if norm_mode == "global":
    all_imgs = [img for img in images.values() if img is not None]
    if len(all_imgs) == 0:
        st.warning("No images found.")
        st.stop()
    gmin, gmax = get_minmax(all_imgs)

# 3) 그리드 렌더
cols = len(ROOT_NAMES_ORDERED)
rows = len(ROW_SUBFOLDERS)
fig = plt.figure(figsize=(cols * 4, rows * 3.5))

for r_idx, sub in enumerate(ROW_SUBFOLDERS, start=1):
    if norm_mode == "per_row":
        row_imgs = [images[(sub, rn)] for rn in ROOT_NAMES_ORDERED if images[(sub, rn)] is not None]
        rmin, rmax = get_minmax(row_imgs) if len(row_imgs) > 0 else (0.0, 1.0)

    for c_idx, root_name in enumerate(ROOT_NAMES_ORDERED, start=1):
        ax = plt.subplot(rows, cols, (r_idx - 1) * cols + c_idx)
        img = images[(sub, root_name)]
        title_name = TEACHER_MAP[sub] if root_name == "teacher" else sub

        if img is not None:
            # normalize
            if norm_mode == "global":
                norm_img = (img - gmin) / (gmax - gmin + 1e-8)
            else:
                norm_img = (img - rmin) / (rmax - rmin + 1e-8)
            ax.imshow(norm_img, cmap="viridis")
            ax.set_title(f"{title_name} | {root_name}", fontsize=10)
            ax.axis("off")

            # 박스 토글: Pred / GT 개별 제어
            if show_pred or show_gt:
                det_list = DET_BY_MODEL.get(root_name)
                if det_list is not None:
                    fid, p_boxes, _, _ = pred_for_idx(det_list, idx, score_th)
                    g_boxes = gt_for_frame(gt_annos, fid)

                    if show_gt and g_boxes is not None and len(g_boxes) > 0:
                        draw_boxes_on_ax(
                            ax, g_boxes, (x_min, x_max), (y_min, y_max), img.shape,
                            color='lime', linewidth=gt_lw, zorder=2, alpha=gt_alpha,
                            swap_xy=swap_xy, flip_u=flip_u, flip_v=flip_v, rotate90=rotate90
                        )

                    if show_pred and p_boxes is not None and len(p_boxes) > 0:
                        draw_boxes_on_ax(
                            ax, p_boxes, (x_min, x_max), (y_min, y_max), img.shape,
                            color='red', linewidth=pred_lw, zorder=3, alpha=pred_alpha,
                            swap_xy=swap_xy, flip_u=flip_u, flip_v=flip_v, rotate90=rotate90
                        )


        else:
            ax.text(0.5, 0.5, f"{title_name} | {root_name}\n(missing)",
                    ha="center", va="center", fontsize=10)
            ax.axis("off")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# 다운로드(선택)
buf = BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0)
st.download_button("Download PNG", data=buf.getvalue(), file_name=f"feat_boxes_{idx:06d}.png", mime="image/png")
