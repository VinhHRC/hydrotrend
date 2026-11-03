import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import kendalltau, gaussian_kde
import plotly.graph_objects as go
import io
import os

# ------------------------------------------------------------------------------
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Hydrological Trend Explorer - KC08.15",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. CẤU HÌNH METADATA 8 TRẠM THỦY VĂN TRÊN 5 LƯU VỰC SÔNG
# ------------------------------------------------------------------------------
STATION_MAP = {
    "Sông Vu Gia": {
        "Thành Mỹ": {"col": "ThanhMy", "desc": "Thượng lưu Vu Gia, chịu tác động thủy điện Sông Bung & A Vương"},
        "Hội Khách": {"col": "HoiKhach", "desc": "Hạ du Thủy điện Đak Mi 4"},
        "Ái Nghĩa": {"col": "AiNghia", "desc": "Khu vực phân lũ Vu Gia - Thu Bồn và lấy nước tưới Đại Lộc"}
    },
    "Sông Thu Bồn": {
        "Nông Sơn": {"col": "NongSon", "desc": "Khống chế dòng chính Thu Bồn, hạ lưu hồ Sông Tranh 2"},
        "Giao Thủy": {"col": "GiaoThuy", "desc": "Hạ lưu ngã ba hợp lưu Vu Gia - Thu Bồn"}
    },
    "Sông Trà Khúc": {
        "Trà Khúc": {"col": "TraKhuc", "desc": "Hạ lưu đập dâng Trà Khúc & hồ Đakdrinh"}
    },
    "Sông Kôn": {
        "Bình Nghi": {"col": "BinhNghi", "desc": "Hạ du đập dâng Văn Phong và hồ Định Bình"}
    },
    "Sông Ba": {
        "Củng Sơn": {"col": "CungSon", "desc": "Hạ du liên hồ Sông Ba Hạ - Sông Hinh"}
    }
}

# ------------------------------------------------------------------------------
# 3. TẢI DỮ LIỆU TỪ 1 FILE DUY NHẤT (Waterlevel.xlsx)
# ------------------------------------------------------------------------------
@st.cache_data
def load_all_waterlevel(file_path="Waterlevel.xlsx"):
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_excel(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file Excel: {e}")
        return None

def calc_sens_slope(x, y):
    """Tính độ dốc Sen's slope phi tham số"""
    slopes = []
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    if len(slopes) == 0:
        return 0.0, 0.0
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return slope, intercept

# Tải dữ liệu
df_all = load_all_waterlevel("Waterlevel.xlsx")

if df_all is None:
    st.error("⚠️ Không tìm thấy file `Waterlevel.xlsx` trong cùng thư mục với `app.py`.")
    st.info("Vui lòng đặt file `Waterlevel.xlsx` vào thư mục dự án và khởi chạy lại.")
    st.stop()

# ------------------------------------------------------------------------------
# 4. GIAO DIỆN THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# ------------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/water.png", width=60)
st.sidebar.title("Bảng Điều Khiển")
st.sidebar.caption("Sản phẩm của Đề tài KC08.15/21-30")

selected_river = st.sidebar.selectbox("1. Chọn Lưu vực sông", list(STATION_MAP.keys()))
selected_station_name = st.sidebar.selectbox("2. Chọn Trạm Thủy văn", list(STATION_MAP[selected_river].keys()))

station_info = STATION_MAP[selected_river][selected_station_name]
col_name = station_info["col"]
st.sidebar.info(f"📍 **Vị trí:** {station_info['desc']}")

# Lọc dữ liệu hợp lệ cho trạm được chọn (bỏ các ngày NaN)
df_station = df_all[['Date', 'Year', 'Month', col_name]].dropna(subset=[col_name]).copy()
df_station = df_station.rename(columns={col_name: 'H'}).sort_values('Date').reset_index(drop=True)

station_min_year = int(df_station['Year'].min())
station_max_year = int(df_station['Year'].max())

# Đặt mặc định cho Bình Nghi là từ 2011 (sau khi chuyển trạm), các trạm khác là từ 2004
default_start = 2011 if col_name == "BinhNghi" else station_min_year

start_yr, end_yr = st.sidebar.slider(
    "3. Giới hạn giai đoạn phân tích:",
    min_value=station_min_year,
    max_value=station_max_year,
    value=(default_start, station_max_year)
)

if col_name == "BinhNghi" and start_yr < 2011:
    st.sidebar.warning("⚠️ Lưu ý: Trước 01/01/2011, trạm đặt tại Bình Tường (cao trình mốc lệch ~6,5 m so với Bình Nghi).")

delta_threshold = st.sidebar.slider("4. Ngưỡng xung dao động ngày (|ΔH| cm):", 5, 30, 10, step=5)

# Lọc dữ liệu theo năm đã chọn
df_filtered = df_station[(df_station['Year'] >= start_yr) & (df_station['Year'] <= end_yr)].copy()

# ------------------------------------------------------------------------------
# 5. TIÊU ĐỀ CHÍNH & METRIC CARDS
# ------------------------------------------------------------------------------
st.title("🌊 Phân tích xu thế mực nước & biến động dòng chảy hạ du các sông ven biển Nam Trung Bộ")
st.markdown(f"**Lưu vực:** `{selected_river}` | **Trạm quan trắc:** `{selected_station_name}` | **Giai đoạn:** `{start_yr} – {end_yr}` ({len(df_filtered):,} ngày quan sát)")

# Tính toán đặc trưng năm
annual_stats = df_filtered.groupby('Year')['H'].agg(
    H_min='min',
    H_p10=lambda x: np.percentile(x, 10),
    H_p50='median',
    H_mean='mean',
    H_p90=lambda x: np.percentile(x, 90),
    H_max='max'
).reset_index()

years_arr = annual_stats['Year'].values
s_min, _ = calc_sens_slope(years_arr, annual_stats['H_min'].values)
s_p50, _ = calc_sens_slope(years_arr, annual_stats['H_p50'].values)
s_max, _ = calc_sens_slope(years_arr, annual_stats['H_max'].values)

tau_min, p_min = kendalltau(years_arr, annual_stats['H_min'].values)
tau_p50, p_p50 = kendalltau(years_arr, annual_stats['H_p50'].values)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Xu thế Mực nước kiệt (H_min)",
        value=f"{s_min * 100:+.2f} cm/năm",
        delta=f"p = {p_min:.2e}" if p_min < 0.001 else f"p = {p_min:.3f}",
        delta_color="inverse" if s_min < 0 else "normal"
    )
with col2:
    st.metric(
        label="Xu thế Mực nước trung vị (H_50%)",
        value=f"{s_p50 * 100:+.2f} cm/năm",
        delta=f"p = {p_p50:.2e}" if p_p50 < 0.001 else f"p = {p_p50:.3f}",
        delta_color="inverse" if s_p50 < 0 else "normal"
    )
with col3:
    st.metric(
        label="Xu thế Đỉnh lũ (H_max)",
        value=f"{s_max * 100:+.2f} cm/năm",
        delta="Ít biến đổi" if abs(s_max * 100) < 5 else "Có biến đổi"
    )
with col4:
    h_drop = annual_stats['H_min'].iloc[0] - annual_stats['H_min'].iloc[-1]
    st.metric(
        label=f"Độ sụt đáy kiệt ({start_yr}–{end_yr})",
        value=f"{h_drop:+.2f} m",
        delta="Tụt dốc" if h_drop > 0 else "Dâng cao",
        delta_color="inverse" if h_drop > 0 else "normal"
    )

st.markdown("---")

# ------------------------------------------------------------------------------
# 6. CÁC TABS PHÂN TÍCH CHUYÊN SÂU
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Xu Thế Mực Nước Dài Hạn",
    "⚡ Dao Động Mực Nước Ngày (|ΔH|)",
    "📊 Đường Tần Suất Duy Trì (WLDC)",
    "📑 Bảng Thống Kê & Tải Dữ Liệu"
])

# ---------------------- TAB 1: XU THẾ DÀI HẠN ----------------------
with tab1:
    st.subheader(f"Đường hồi quy xu thế Sen's Slope tại trạm {selected_station_name}")
    years_fit = np.linspace(start_yr, end_yr, 50)
    _, i_min = calc_sens_slope(years_arr, annual_stats['H_min'].values)
    _, i_p50 = calc_sens_slope(years_arr, annual_stats['H_p50'].values)
    _, i_max = calc_sens_slope(years_arr, annual_stats['H_max'].values)

    fig_trend = go.Figure()

    # H_max
    fig_trend.add_trace(go.Scatter(
        x=annual_stats['Year'], y=annual_stats['H_max'],
        mode='markers', name=f'H_max (Đỉnh lũ) [{s_max*100:+.2f} cm/năm]',
        marker=dict(color='#d62728', size=9)
    ))
    fig_trend.add_trace(go.Scatter(
        x=years_fit, y=i_max + s_max * years_fit,
        mode='lines', name='Đường xu thế H_max',
        line=dict(color='#d62728', dash='dash', width=1.5), showlegend=False
    ))

    # H_50%
    fig_trend.add_trace(go.Scatter(
        x=annual_stats['Year'], y=annual_stats['H_p50'],
        mode='markers', name=f'H_50% (Trung vị) [{s_p50*100:+.2f} cm/năm]',
        marker=dict(color='#2ca02c', size=9)
    ))
    fig_trend.add_trace(go.Scatter(
        x=years_fit, y=i_p50 + s_p50 * years_fit,
        mode='lines', name='Đường xu thế H_50%',
        line=dict(color='#2ca02c', width=2), showlegend=False
    ))

    # H_min
    fig_trend.add_trace(go.Scatter(
        x=annual_stats['Year'], y=annual_stats['H_min'],
        mode='markers', name=f'H_min (Đáy kiệt) [{s_min*100:+.2f} cm/năm]',
        marker=dict(color='#1f77b4', size=9)
    ))
    fig_trend.add_trace(go.Scatter(
        x=years_fit, y=i_min + s_min * years_fit,
        mode='lines', name='Đường xu thế H_min',
        line=dict(color='#1f77b4', width=2.5), showlegend=False
    ))

    fig_trend.update_layout(
        xaxis=dict(title='Năm', dtick=1 if (end_yr - start_yr) <= 15 else 2),
        yaxis=dict(title='Mực nước H (m)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.info(f"""
    **Nhận định nhanh:**
    - Tốc độ suy giảm mực nước kiệt ($H_{{min}}$) đạt **{s_min*100:+.2f} cm/năm** (mức ý nghĩa $p = {p_min:.2e}$).
    - Tốc độ suy giảm trung vị ($H_{{50\\%}}$) đạt **{s_p50*100:+.2f} cm/năm** ($p = {p_p50:.2e}$).
    - Nếu $H_{{min}}$ và $H_{{50\\%}}$ suy giảm sâu trong khi $H_{{max}}$ ít thay đổi, đây là bằng chứng điển hình của hiện tượng **hạ thấp đáy sông/xói lòng dẫn** hạ du.
    """)

# ---------------------- TAB 2: DAO ĐỘNG NGÀY ----------------------
with tab2:
    st.subheader("Đặc trưng dao động mực nước ngày (|ΔH| = |H(t) - H(t-1)|)")
    
    df_diff = df_filtered.copy()
    df_diff['dH'] = df_diff['H'].diff()
    df_diff['abs_dH'] = df_diff['dH'].abs()
    df_diff['date_diff'] = df_diff['Date'].diff().dt.days
    df_consec = df_diff[df_diff['date_diff'] == 1].dropna(subset=['abs_dH']).copy()
    df_consec['Season'] = np.where(df_consec['Month'].isin([9, 10, 11, 12]), 'Mùa lũ (T9-T12)', 'Mùa kiệt (T1-T8)')

    c1, c2 = st.columns([2, 1])
    with c1:
        fig_kde = go.Figure()
        for season, color in [('Mùa kiệt (T1-T8)', '#1f77b4'), ('Mùa lũ (T9-T12)', '#d62728')]:
            sub = df_consec[df_consec['Season'] == season]['abs_dH'].values
            if len(sub) > 10:
                kde = gaussian_kde(sub)
                x_vals = np.linspace(0, min(1.0, np.percentile(sub, 99.5)), 200)
                y_vals = kde(x_vals)
                
                fig_kde.add_trace(go.Scatter(
                    x=x_vals * 100, y=y_vals,
                    mode='lines', name=f"{season} (TB = {np.mean(sub)*100:.1f} cm/ngày)",
                    line=dict(color=color, width=2),
                    fill='tozeroy', fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}"
                ))

        fig_kde.add_vline(x=delta_threshold, line_dash="dash", line_color="orange",
                          annotation_text=f"Ngưỡng xung {delta_threshold} cm/ngày", annotation_position="top right")

        fig_kde.update_layout(
            title="Đường phân phối mật độ xác suất dao động ngày (|ΔH|)",
            xaxis=dict(title='Biên độ dao động ngày |ΔH| (cm/ngày)'),
            yaxis=dict(title='Mật độ xác suất (Density)'),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_kde, use_container_width=True)

    with c2:
        st.markdown("#### Thống kê dao động ngày theo mùa:")
        for season in ['Mùa kiệt (T1-T8)', 'Mùa lũ (T9-T12)']:
            sub = df_consec[df_consec['Season'] == season]['abs_dH']
            n_tot = len(sub)
            n_exceed = (sub >= delta_threshold / 100.0).sum()
            pct_exceed = (n_exceed / n_tot * 100) if n_tot > 0 else 0
            
            st.markdown(f"""
            <div class="metric-card">
                <b>{season}</b> ({n_tot:,} ngày quan sát)<br>
                • Trung bình: <b>{sub.mean()*100:.2f} cm/ngày</b><br>
                • Trung vị: <b>{sub.median()*100:.2f} cm/ngày</b><br>
                • Phân vị 90%: <b>{sub.quantile(0.90)*100:.2f} cm/ngày</b><br>
                • Tần suất |ΔH| ≥ {delta_threshold} cm: <b style="color:#d9534f;">{pct_exceed:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

# ---------------------- TAB 3: ĐƯỜNG TẦN SUẤT WLDC ----------------------
with tab3:
    st.subheader("Đường tần suất duy trì mực nước (WLDC) theo 2 phân kỳ")
    mid_yr = start_yr + (end_yr - start_yr) // 2
    df_filtered['SubPeriod'] = np.where(df_filtered['Year'] <= mid_yr, f"{start_yr} - {mid_yr}", f"{mid_yr + 1} - {end_yr}")
    
    col_wldc1, col_wldc2 = st.columns(2)
    for idx, (col_ui, s_name) in enumerate([(col_wldc1, 'Mùa kiệt (T1-T8)'), (col_wldc2, 'Mùa lũ (T9-T12)')]):
        with col_ui:
            s_mask = df_filtered['Month'].isin([9, 10, 11, 12]) if 'lũ' in s_name else ~df_filtered['Month'].isin([9, 10, 11, 12])
            sub_df = df_filtered[s_mask]
            
            fig_wldc = go.Figure()
            colors = ['#2ca02c', '#d62728']
            for p_idx, p_name in enumerate(sorted(sub_df['SubPeriod'].unique())):
                vals = sub_df[sub_df['SubPeriod'] == p_name]['H'].sort_values(ascending=False).values
                n_v = len(vals)
                if n_v > 0:
                    probs = (np.arange(1, n_v + 1) / (n_v + 1)) * 100
                    fig_wldc.add_trace(go.Scatter(
                        x=probs, y=vals, mode='lines',
                        name=f"Thời kỳ {p_name}",
                        line=dict(color=colors[p_idx % len(colors)], width=2)
                    ))
            
            fig_wldc.update_layout(
                title=f"Đường tần suất duy trì - {s_name}",
                xaxis=dict(title='Tần suất duy trì P (%)', range=[0, 100]),
                yaxis=dict(title='Mực nước H (m)'),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_wldc, use_container_width=True)

# ---------------------- TAB 4: BẢNG SỐ LIỆU & TẢI XUỐNG ----------------------
with tab4:
    st.subheader(f"Bảng tổng hợp đặc trưng mực nước năm tại trạm {selected_station_name}")
    display_table = annual_stats.copy()
    display_table.columns = ['Năm', 'H_min (m)', 'H_10% (m)', 'H_50% (m)', 'H_tb (m)', 'H_90% (m)', 'H_max (m)']
    st.dataframe(display_table.style.format("{:.2f}", subset=display_table.columns[1:]), use_container_width=True)
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        display_table.to_excel(writer, sheet_name=f'Stats_{selected_station_name}', index=False)
        df_filtered[['Date', 'Year', 'Month', 'H']].to_excel(writer, sheet_name='Daily_Data', index=False)
    
    st.download_button(
        label=f"📥 Tải Bảng Số Liệu Trạm {selected_station_name} (Excel)",
        data=excel_buffer.getvalue(),
        file_name=f"So_lieu_{selected_station_name}_{start_yr}_{end_yr}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ------------------------------------------------------------------------------
# 7. CHÂN TRANG (FOOTER)
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Tác giả: Hoàng Đức Vinh - Phòng TNTĐQG về động lực học sông biển - "
    "Sản phẩm khoa học thuộc Đề tài cấp Nhà nước: "
    "*'Nghiên cứu ảnh hưởng của tình trạng hạ thấp mực nước đến ổn định và vận hành các công trình trên sông vùng hạ du sông chính Nam Trung Bộ và đề xuất các giải pháp khắc phục'* (Mã số: **KC08.15/21-30**)."
)
