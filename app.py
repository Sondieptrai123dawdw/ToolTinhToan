import io
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

st.set_page_config(
    page_title="Bù Trừ",
    page_icon="⚙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ======================================================================
# CẤU HÌNH CỐ ĐỊNH (giao diện không còn hiện các mục này – muốn đổi thì sửa tại đây)
# ======================================================================
MODE = "2/2"                      # Đổi thành "1/1", "2/2" hoặc "4/4" tùy nhu cầu
ANGLES = {                        # góc xoay hình của từng vị trí
    "1/1": [0],
    "2/2": [180, -180],
    "4/4": [90, 180, -90, -180],
}
CCW = True                        # True: hình xoay ngược chiều kim đồng hồ
UNIT = "mm"                       # "mm" hoặc "inch"
DECIMALS_POS, DECIMALS_ANG = 3, 2
TOL_POS, TOL_ANG = 0.020, 0.050   # dung sai để báo OK/NG
INVERT = (False, False, False)    # đảo dấu (X, Y, Góc) nếu máy nhập ngược chiều
L_DIST = 6.165                    # Khoảng cách L cố định
PDF_FILE = "ban_ve.pdf"           # file PDF đặt cùng thư mục với file .py trên GitHub
PDF_PAGE = 0                      # trang hiển thị (0 là trang đầu)
PDF_ZOOM = 1.5
SHOW_AXES, AXES_CORNER = True, "Trên phải"   # "Dưới trái", "Dưới phải", "Trên trái", "Trên phải"
AXES_SCALE = 1.5                  # độ lớn hệ trục XYA (1.0 = nhỏ, 2.0 = rất to)

# Màu sắc hệ trục chuẩn (Xanh đậm chuyên nghiệp)
COLOR_X, COLOR_Y, COLOR_A = (30, 90, 120), (30, 90, 120), (30, 90, 120)

# ======================================================================
# CSS TỐI ƯU CHO ĐIỆN THOẠI & ẨN DÒNG CHỮ PRESS ENTER
# ======================================================================
st.markdown(
    """
<style>
.block-container {padding: 0.8rem 0.7rem 4rem !important; max-width: 640px;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none !important;}
h1 {font-size: 1.4rem !important; padding: 0 0 .3rem 0 !important;}
h2, h3 {font-size: 1.1rem !important; padding: .4rem 0 .2rem 0 !important;}
/* iOS tự phóng to trang nếu chữ ô nhập < 16px */
input, textarea, [data-baseweb="select"] * {font-size: 16px !important;}
/* Ẩn nút +/- trong ô số để dành chỗ cho số */
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {display: none !important;}
/* Giữ các cột nằm ngang trên màn hình nhỏ */
[data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: .5rem !important;}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
[data-testid="stHorizontalBlock"] > [data-testid="column"] {min-width: 0 !important; flex: 1 1 0 !important; width: auto !important;}
.stButton button, [data-testid="stFormSubmitButton"] button, .stDownloadButton button {min-height: 3rem; font-weight: 600;}
[data-testid="stMetricValue"] {font-size: 1.35rem !important;}
[data-testid="stMetricLabel"] p {font-size: .8rem !important;}

/* Ẩn hoàn toàn dòng chữ hướng dẫn "Press Enter to submit form" */
[data-testid="InputInstructions"] {
    display: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def stretch(fn, *args, **kwargs):
    """Gọi hàm hiển thị full chiều rộng, tương thích cả bản Streamlit cũ và mới."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


# ======================================================================
# HÀM TÍNH TOÁN VÀ VẼ (không phụ thuộc giao diện)
# ======================================================================
# >>> PURE_START
def clean(v: float, decimals: int) -> float:
    return round(v, decimals) + 0.0


def fmt(v: float, decimals: int) -> str:
    return f"{clean(v, decimals):+.{decimals}f}"


def rotate_to_machine(cx: float, cy: float, theta: int, ccw: bool):
    """Đổi bù trừ từ hệ bản vẽ sang hệ máy cho vị trí xoay theta độ (bội số của 90)."""
    k = int(round(theta / 90)) % 4
    if not ccw:
        k = (-k) % 4
    if k == 0:
        return cx, cy
    if k == 1:
        return -cy, cx
    if k == 2:
        return -cx, -cy
    return cy, -cx


def compute_rows(measured, angles, L, ccw, inv, tol_pos, tol_ang):
    inv_x, inv_y, inv_a = inv
    rows = []
    for i, (ax, ay, k1, k2, draw_x, draw_y, d_target) in enumerate(measured):
        err_x, err_y = draw_x - ax, draw_y - ay
        err_a = float(np.degrees(np.arctan((k1 - k2) / L)))
        mx, my = rotate_to_machine(-err_x, -err_y, angles[i], ccw)
        ma = -err_a
        mx = -mx if inv_x else mx
        my = -my if inv_y else my
        ma = -ma if inv_a else ma
        mean_k = (k1 + k2) / 2
        rows.append({
            "i": i + 1, "angle": angles[i],
            "draw_x": draw_x, "draw_y": draw_y, "d_target": d_target,
            "err_x": err_x, "err_y": err_y, "err_a": err_a,
            "k1": k1, "k2": k2, "mean_k": mean_k, "remain": mean_k - d_target,
            "mx": mx, "my": my, "ma": ma,
            "ok": abs(err_x) <= tol_pos and abs(err_y) <= tol_pos and abs(err_a) <= tol_ang,
        })
    return rows


def get_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_arrow(draw, p0, p1, color, width, head):
    draw.line([p0, p1], fill=color, width=width)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    bx, by = p1[0] - ux * head, p1[1] - uy * head
    draw.polygon(
        [p1, (bx + px * head * 0.5, by + py * head * 0.5),
         (bx - px * head * 0.5, by - py * head * 0.5)],
        fill=color,
    )


def rotate_drawing(img, angle: int, ccw: bool):
    a = (angle if ccw else -angle) % 360
    if a == 0:
        return img
    return img.rotate(a, expand=True, fillcolor=(255, 255, 255))


def draw_axes(img, corner: str, scale: float = 1.0):
    base = img.convert("RGBA")
    w, h = base.size

    L = int(max(85, min(w, h) * 0.15 * scale))
    fs = max(18, L // 3)
    lw = max(5, L // 15)
    head = max(14, L // 5)

    font = get_font(fs)

    bw = int(L * 1.7 + fs * 2.5)
    bh = int(L * 1.7 + fs * 2.0)
    margin = int(L * 0.18)

    left = "trái" in corner
    bottom = "Dưới" in corner
    bx0 = margin if left else w - bw - margin
    by0 = h - bh - margin if bottom else margin

    ox = bx0 + int(L * 0.35)
    oy = by0 + bh - int(L * 0.35)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    d.rounded_rectangle(
        [bx0, by0, bx0 + bw, by0 + bh],
        radius=max(12, L // 8),
        fill=(255, 255, 255, 225),
        outline=(90, 90, 90, 255),
        width=3,
    )

    draw_arrow(d, (ox, oy), (ox + L, oy), COLOR_X, lw, head)
    draw_arrow(d, (ox, oy), (ox, oy - L), COLOR_Y, lw, head)

    r = int(L * 0.6)
    arc_box = [ox - r, oy - r, ox + r, oy + r]
    d.arc(arc_box, start=270, end=360, fill=COLOR_A, width=lw)

    d.text((ox + L + 8, oy + 4), "X", fill=COLOR_X, font=font)
    d.text((ox - fs - 4, oy - L - fs - 2), "Y", fill=COLOR_Y, font=font)
    d.text((ox + int(r * 0.7), oy - int(r * 0.7) - fs // 2), "A", fill=COLOR_A, font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")

# <<< PURE_END


@st.cache_data(show_spinner=False)
def pdf_page_count(_pdf_bytes: bytes, pdf_key: str) -> int:
    with fitz.open(stream=_pdf_bytes, filetype="pdf") as doc:
        return len(doc)


@st.cache_data(show_spinner=False, max_entries=24)
def render_view(_pdf_bytes: bytes, pdf_key: str, page_no: int, zoom: float,
                angle: int, ccw: bool, show_axes: bool, corner: str, axes_scale: float) -> bytes:
    with fitz.open(stream=_pdf_bytes, filetype="pdf") as doc:
        pix = doc[page_no].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = rotate_drawing(img, angle, ccw)
    max_w = 1200
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)
    if show_axes:
        img = draw_axes(img, corner, axes_scale)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85, optimize=True)
    return buf.getvalue()


# ======================================================================
# GIAO DIỆN
# ======================================================================
if "history" not in st.session_state:
    st.session_state.history = []

st.title("⚙️ Bù Trừ Nhanh")

ccw, unit = CCW, UNIT
decimals_pos, decimals_ang = DECIMALS_POS, DECIMALS_ANG
tol_pos, tol_ang = TOL_POS, TOL_ANG
inv_x, inv_y, inv_a = INVERT
mode = MODE
n_units = int(mode.split("/")[0])
angles = ANGLES[mode]
labels = [f"{i + 1}/{n_units} ({angles[i]}°)" for i in range(n_units)]

pdf_bytes, pdf_key, page_no = None, "", PDF_PAGE
show_axes, corner, zoom = SHOW_AXES, AXES_CORNER, PDF_ZOOM
_pdf_path = Path(__file__).with_name(PDF_FILE)
if HAS_FITZ and _pdf_path.exists():
    try:
        pdf_bytes = _pdf_path.read_bytes()
        pdf_key = f"{_pdf_path.name}-{_pdf_path.stat().st_mtime_ns}"
        page_no = min(PDF_PAGE, pdf_page_count(pdf_bytes, pdf_key) - 1)
    except Exception as e:
        pdf_bytes = None
        st.error(f"Không đọc được {PDF_FILE}: {e}")

if pdf_bytes is not None:
    with st.expander("🖼️ Xem bản vẽ", expanded=True):
        view_idx = 0
        if n_units > 1:
            view_idx = st.radio("Vị trí", list(range(n_units)), format_func=lambda k: labels[k],
                                horizontal=True, key=f"view_{mode}")
        try:
            jpg = render_view(pdf_bytes, pdf_key, page_no, float(zoom),
                            angles[view_idx], ccw, show_axes, corner, AXES_SCALE)
            stretch(st.image, jpg)
        except Exception as e:
            st.error(f"Không hiển thị được bản vẽ: {e}")

# ---------------- FORM NHẬP LIỆU ----------------
with st.form("form_bu_tru"):
    h_title, h_axis = st.columns([2.15, 1.0], vertical_alignment="center")
    with h_title:
        st.markdown("**Thông số mục tiêu & thực tế theo vị trí**")
    with h_axis:
        st.markdown(
'<div style="display: flex; justify-content: center; align-items: center;">'
'<svg width="150" height="110" viewBox="0 0 160 110" style="background: transparent;">'
'<line x1="30" y1="90" x2="135" y2="90" stroke="#1e5a78" stroke-width="4" stroke-linecap="round"/>'
'<polygon points="135,90 123,84 123,96" fill="#1e5a78"/>'
'<text x="142" y="96" fill="#1e5a78" font-family="sans-serif" font-weight="bold" font-size="16">X</text>'
'<line x1="30" y1="90" x2="30" y2="18" stroke="#1e5a78" stroke-width="4" stroke-linecap="round"/>'
'<polygon points="30,18 24,28 36,28" fill="#1e5a78"/>'
'<text x="22" y="13" fill="#1e5a78" font-family="sans-serif" font-weight="bold" font-size="16">Y</text>'
'<path d="M 30 50 A 40 40 0 0 1 70 90" fill="none" stroke="#1e5a78" stroke-width="4"/>'
'<text x="75" y="58" fill="#1e5a78" font-family="sans-serif" font-weight="bold" font-size="15">A</text>'
'<circle cx="30" cy="90" r="4" fill="#1e5a78"/>'
'</svg></div>',
            unsafe_allow_html=True,
        )

    containers = [st.container()] if n_units == 1 else st.tabs(labels)
    measured = []
    
    for i, box in enumerate(containers):
        with box:
            st.markdown(f"**🎯 Mục tiêu bản vẽ {i + 1}**")
            c_t1, c_t2 = st.columns(2)
            draw_x = c_t1.number_input("KT bản vẽ X", value=1.67, format="%.3f", key=f"draw_x_{i}_{mode}")
            draw_y = c_t2.number_input("KT bản vẽ Y", value=1.89, format="%.3f", key=f"draw_y_{i}_{mode}")
            d_target = st.number_input("KT bản vẽ góc (KT1 & KT2)", value=1.89, format="%.3f", key=f"d_target_{i}_{mode}")

            st.markdown(f"**📏 Số đo thực tế vị trí {i + 1}**")
            c1, c2 = st.columns(2)
            ax = c1.number_input("Thực tế X", value=1.99, format="%.3f", key=f"ax_{i}_{mode}")
            ay = c2.number_input("Thực tế Y", value=1.77, format="%.3f", key=f"ay_{i}_{mode}")
            c3, c4 = st.columns(2)
            k1 = c3.number_input("KT1 (trên)", value=1.99, format="%.3f", key=f"k1_{i}_{mode}")
            k2 = c4.number_input("KT2 (dưới)", value=1.89, format="%.3f", key=f"k2_{i}_{mode}")
            
            measured.append((ax, ay, k1, k2, draw_x, draw_y, d_target))

    with st.expander("Offset đang có trên máy (tùy chọn)"):
        f1, f2, f3 = st.columns(3)
        cur_x = f1.number_input("X hiện tại", value=0.000, format="%.3f")
        cur_y = f2.number_input("Y hiện tại", value=0.000, format="%.3f")
        cur_a = f3.number_input("Góc hiện tại", value=0.00, format="%.3f")
        use_cur = st.checkbox("Cộng dồn vào offset hiện tại", value=False)

    note = st.text_input("Ghi chú", placeholder="Mã hàng / lô / ca...")
    submitted = stretch(st.form_submit_button, "🚀 XEM KẾT QUẢ", type="primary")

# ---------------- TÍNH TOÁN ----------------
if submitted:
    rows = compute_rows(measured, angles, L_DIST, ccw,
                        (inv_x, inv_y, inv_a), tol_pos, tol_ang)
    if use_cur:
        for r in rows:
            r["mx"] = cur_x + r["mx"]
            r["my"] = cur_y + r["my"]
            r["ma"] = cur_a + r["ma"]
            
    st.session_state.result = {
        "rows": rows, "use_cur": use_cur, "mode": mode, "n": n_units,
        "unit": unit, "dp": decimals_pos, "da": decimals_ang,
        "tol_pos": tol_pos, "tol_ang": tol_ang,
    }
    now = datetime.now().strftime("%H:%M:%S")
    new = [{
        "Giờ": now, "Ghi chú": note, "Chế độ": mode,
        "Vị trí": f"{r['i']}/{n_units} ({r['angle']}°)",
        "Lệch X": clean(r["err_x"], 4), "Lệch Y": clean(r["err_y"], 4),
        "KT1": clean(r["k1"], 4), "KT2": clean(r["k2"], 4),
        "Lệch góc (°)": clean(r["err_a"], 4),
        "Bù X": clean(r["mx"], decimals_pos), "Bù Y": clean(r["my"], decimals_pos),
        "Bù góc (°)": clean(r["ma"], decimals_ang), "Đạt": "OK" if r["ok"] else "NG",
    } for r in rows]
    st.session_state.history = (new + st.session_state.history)[:200]


# ---------------- HIỂN THỊ KẾT QUẢ THEO TAB TỪNG BẢN VẼ ----------------
def show_result(res):
    n, unit_, dp, da = res["n"], res["unit"], res["dp"], res["da"]
    rows = res["rows"]

    st.markdown("---")
    st.markdown("### 🎯 Kết quả bù trừ theo từng bản vẽ")

    result_tabs = [st.container()] if n == 1 else st.tabs([f"Bản vẽ {r['i']} ({r['angle']}°)" for r in rows])

    for i, tab in enumerate(result_tabs):
        r = rows[i]
        with tab:
            m1, m2, m3 = st.columns(3)
            m1.metric("Bù X", fmt(r["mx"], dp))
            m2.metric("Bù Y", fmt(r["my"], dp))
            m3.metric("Bù góc (°)", fmt(r["ma"], da))
            st.caption(f"Vị trí {r['i']}/{n} • đơn vị {unit_}")
            st.code(f"X {fmt(r['mx'], dp)}   Y {fmt(r['my'], dp)}   A {fmt(r['ma'], da)}", language=None)

            if r["ok"]:
                st.success(f"Bản vẽ số {r['i']} nằm trong dung sai – không cần bù trừ.")
            else:
                st.warning(f"Bản vẽ số {r['i']} ngoài dung sai (NG).")

# KÍCH HOẠT HIỂN THỊ KẾT QUẢ KHI ĐÃ CÓ DỮ LIỆU TRONG SESSION_STATE
if "result" in st.session_state:
    show_result(st.session_state.result)

# ---------------- LỊCH SỬ ----------------
if st.session_state.history:
    with st.expander(f"🕘 Lịch sử ({len(st.session_state.history)})"):
        df = pd.DataFrame(st.session_state.history)
        stretch(st.dataframe, df.head(30), hide_index=True)
        h1, h2 = st.columns(2)
        with h1:
            stretch(
                st.download_button, "⬇️ Tải CSV",
                df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"bu_tru_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv",
            )
        with h2:
            if stretch(st.button, "🗑️ Xóa"):
                st.session_state.history = []
                st.session_state.pop("result", None)
                st.rerun()
