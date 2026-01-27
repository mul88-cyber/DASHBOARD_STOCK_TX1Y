# ==============================================================================
# 📦 1) IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import time

# Import library Google
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ 2) KONFIGURASI DASHBOARD & G-DRIVE
# ==============================================================================
st.set_page_config(
    page_title="📈 Dashboard Saham IDX + MSCI Simulator",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# --- CSS PROFESSIONAL THEME ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp { background-color: #F8F9FA; color: #212529; font-family: 'Inter', sans-serif; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; color: #0d6efd; }
    div[data-testid="stMetricLabel"] { font-size: 14px; font-weight: 600; color: #6c757d; }
    
    /* Cards */
    .css-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 15px;
    }
    
    /* Tables */
    div[data-testid="stDataFrame"] { background-color: white; border-radius: 10px; padding: 10px; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #dee2e6; }
    
    /* Custom Headers */
    .header-title { font-size: 28px; font-weight: 800; color: #1a1a1a; margin-bottom: 5px; }
    .header-subtitle { font-size: 16px; color: #6c757d; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- KONFIGURASI G-DRIVE ---
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP" 
FILE_NAME = "Kompilasi_Data_1Tahun.csv"

# Bobot skor (Logic "Raport")
W = dict(
    trend_akum=0.40, trend_ff=0.30, trend_mfv=0.20, trend_mom=0.10,
    mom_price=0.40,  mom_vol=0.25,  mom_akum=0.25,  mom_ff=0.10,
    blend_trend=0.35, blend_mom=0.35, blend_nbsa=0.20, blend_fcontrib=0.05, blend_unusual=0.05
)

# ==============================================================================
# 📦 3) FUNGSI MEMUAT DATA (ROBUST AUTH)
# ==============================================================================
def get_gdrive_service():
    """Robust Authentication with Key Fixing"""
    try:
        if "gcp_service_account" not in st.secrets:
            return None, "❌ Key 'gcp_service_account' missing in secrets."
            
        creds_data = st.secrets["gcp_service_account"]
        if hasattr(creds_data, "to_dict"):
            creds_json = creds_data.to_dict()
        else:
            creds_json = dict(creds_data)
            
        # FIX PRIVATE KEY FORMAT (Handle \n issue)
        if "private_key" in creds_json:
            pk = str(creds_json["private_key"])
            if "\\n" in pk:
                creds_json["private_key"] = pk.replace("\\n", "\n")

        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service, None
    except Exception as e:
        return None, f"❌ Auth Error: {e}"

@st.cache_data(ttl=3600, show_spinner="🔄 Downloading Market Data...")
def load_data():
    service, error_msg = get_gdrive_service()
    if error_msg: return pd.DataFrame(), error_msg, "error"

    try:
        query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)", orderBy="modifiedTime desc", pageSize=1).execute()
        items = results.get('files', [])

        if not items: return pd.DataFrame(), f"❌ File '{FILE_NAME}' not found.", "error"

        file_id = items[0]['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: status, done = downloader.next_chunk()
        fh.seek(0)

        df = pd.read_csv(fh, dtype=object)
        df.columns = df.columns.str.strip()
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')

        cols_to_numeric = [
            'High', 'Low', 'Close', 'Volume', 'Value', 'Foreign Buy', 'Foreign Sell',
            'Bid Volume', 'Offer Volume', 'Previous', 'Change', 'Open Price', 'First Trade',
            'Frequency', 'Index Individual', 'Offer', 'Bid', 'Listed Shares', 'Tradeble Shares',
            'Weight For Index', 'Non Regular Volume', 'Change %', 'Typical Price', 'TPxV',
            'VWMA_20D', 'MA20_vol', 'MA5_vol', 'Volume Spike (x)', 'Net Foreign Flow',
            'Bid/Offer Imbalance', 'Money Flow Value', 'Free Float', 'Money Flow Ratio (20D)'
        ]

        for col in cols_to_numeric:
            if col in df.columns:
                cleaned_col = df[col].astype(str).str.strip().str.replace(r'[,\sRp\%]', '', regex=True)
                df[col] = pd.to_numeric(cleaned_col, errors='coerce').fillna(0)

        if 'Unusual Volume' in df.columns:
            df['Unusual Volume'] = df['Unusual Volume'].astype(str).str.strip().str.lower().isin(['spike volume signifikan', 'true', 'True', 'TRUE'])
            df['Unusual Volume'] = df['Unusual Volume'].astype(bool)

        if 'Final Signal' in df.columns:
            df['Final Signal'] = df['Final Signal'].astype(str).str.strip()

        df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others') if 'Sector' in df.columns else 'Others'
        df = df.dropna(subset=['Last Trading Date', 'Stock Code'])

        if 'NFF (Rp)' not in df.columns:
            if 'Typical Price' in df.columns:
                df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Typical Price']
            else:
                df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Close']

        return df, "Data loaded successfully", "success"

    except Exception as e:
        return pd.DataFrame(), f"❌ Load Error: {e}", "error"

# ==============================================================================
# 🛠️ 4) FUNGSI KALKULASI UTAMA (OPTIMIZED)
# ==============================================================================
def pct_rank(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, method="average").fillna(0) * 100

def to_pct(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() <= 1: return pd.Series(50, index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx: return pd.Series(50, index=s.index)
    return (s - mn) / (mx - mn) * 100

# --- FUNGSI SCORING ---
def calculate_potential_score(df: pd.DataFrame, latest_date: pd.Timestamp):
    trend_start = latest_date - pd.Timedelta(days=30)
    mom_start = latest_date - pd.Timedelta(days=7)
    
    df_historic = df[df['Last Trading Date'] <= latest_date]
    trend_df = df_historic[df_historic['Last Trading Date'] >= trend_start].copy()
    mom_df = df_historic[df_historic['Last Trading Date'] >= mom_start].copy()
    last_df = df_historic[df_historic['Last Trading Date'] == latest_date].copy()

    if trend_df.empty or mom_df.empty or last_df.empty:
        return pd.DataFrame(), "Data insufficient for scoring.", "warning"

    # 1. Trend Score
    tr = trend_df.groupby('Stock Code').agg(
        last_price=('Close', 'last'), last_final_signal=('Final Signal', 'last'),
        total_net_ff_rp=('NFF (Rp)', 'sum'), total_money_flow=('Money Flow Value', 'sum'),
        avg_change_pct=('Change %', 'mean'), sector=('Sector', 'last')
    ).reset_index()
    score_akum = tr['last_final_signal'].map({'Strong Akumulasi': 100, 'Akumulasi': 75, 'Netral': 30, 'Distribusi': 10, 'Strong Distribusi': 0}).fillna(30)
    score_ff = pct_rank(tr['total_net_ff_rp'])
    score_mfv = pct_rank(tr['total_money_flow'])
    score_mom = pct_rank(tr['avg_change_pct'])
    tr['Trend Score'] = (score_akum * W['trend_akum'] + score_ff * W['trend_ff'] + score_mfv * W['trend_mfv'] + score_mom * W['trend_mom'])

    # 2. Momentum Score
    mo = mom_df.groupby('Stock Code').agg(
        total_change_pct=('Change %', 'sum'), had_unusual_volume=('Unusual Volume', 'any'),
        last_final_signal=('Final Signal', 'last'), total_net_ff_rp=('NFF (Rp)', 'sum')
    ).reset_index()
    s_price = pct_rank(mo['total_change_pct'])
    s_vol = mo['had_unusual_volume'].map({True: 100, False: 20}).fillna(20)
    s_akum = mo['last_final_signal'].map({'Strong Akumulasi': 100, 'Akumulasi': 80, 'Netral': 40, 'Distribusi': 10, 'Strong Distribusi': 0}).fillna(40)
    s_ff7 = pct_rank(mo['total_net_ff_rp'])
    mo['Momentum Score'] = (s_price * W['mom_price'] + s_vol * W['mom_vol'] + s_akum * W['mom_akum'] + s_ff7 * W['mom_ff'])

    # 3. NBSA & Contrib
    nbsa = trend_df.groupby('Stock Code').agg(total_net_ff_30d_rp=('NFF (Rp)', 'sum')).reset_index()
    
    if {'Foreign Buy', 'Foreign Sell', 'Value'}.issubset(df.columns):
        trend_df['Foreign Value proxy'] = trend_df['NFF (Rp)']
        contrib = trend_df.groupby('Stock Code').agg(total_foreign_value_proxy=('Foreign Value proxy', 'sum'), total_value_30d=('Value', 'sum')).reset_index()
        contrib['foreign_contrib_pct'] = np.where(contrib['total_value_30d'] > 0, (contrib['total_foreign_value_proxy'].abs() / contrib['total_value_30d']) * 100, 0)
    else:
        contrib = pd.DataFrame({'Stock Code': [], 'foreign_contrib_pct': []})

    uv = last_df.set_index('Stock Code')['Unusual Volume'].map({True: 1, False: 0})

    # Merge
    rank = tr[['Stock Code', 'Trend Score', 'last_price', 'last_final_signal', 'sector']].merge(
        mo[['Stock Code', 'Momentum Score']], on='Stock Code', how='outer'
    ).merge(nbsa, on='Stock Code', how='left').merge(contrib[['Stock Code', 'foreign_contrib_pct']], on='Stock Code', how='left')
    
    rank['NBSA Score'] = to_pct(rank['total_net_ff_30d_rp'])
    rank['Foreign Contrib Score'] = to_pct(rank['foreign_contrib_pct'])
    unusual_bonus = uv.reindex(rank['Stock Code']).fillna(0) * 5
    
    rank['Potential Score'] = (
        rank['Trend Score'].fillna(0) * W['blend_trend'] +
        rank['Momentum Score'].fillna(0) * W['blend_mom'] +
        rank['NBSA Score'].fillna(50) * W['blend_nbsa'] +
        rank['Foreign Contrib Score'].fillna(50) * W['blend_fcontrib'] +
        unusual_bonus.values * W['blend_unusual']
    )

    top20 = rank.sort_values('Potential Score', ascending=False).head(20).copy()
    top20.insert(0, 'Analysis Date', latest_date.strftime('%Y-%m-%d'))
    return top20, "Success", "success"

# --- FUNGSI FLOW ---
@st.cache_data(ttl=3600)
def calculate_nff_top_stocks(df, max_date):
    periods = {'7D': 7, '30D': 30, '90D': 90, '180D': 180}; results = {}
    latest_data = df[df['Last Trading Date'] == max_date].set_index('Stock Code')
    
    for name, days in periods.items():
        start_date = max_date - pd.Timedelta(days=days)
        df_period = df[df['Last Trading Date'] >= start_date].copy()
        nff_agg = df_period.groupby('Stock Code')['NFF (Rp)'].sum()
        df_agg = pd.DataFrame(nff_agg).join(latest_data.get('Close', pd.Series())).join(latest_data.get('Sector', pd.Series()))
        df_agg.columns = ['Total Net FF (Rp)', 'Harga Terakhir', 'Sector']
        results[name] = df_agg.sort_values(by='Total Net FF (Rp)', ascending=False).reset_index()
    return results['7D'], results['30D'], results['90D'], results['180D']

@st.cache_data(ttl=3600)
def calculate_mfv_top_stocks(df, max_date):
    periods = {'7D': 7, '30D': 30, '90D': 90, '180D': 180}; results = {}
    latest_data = df[df['Last Trading Date'] == max_date].set_index('Stock Code')
    
    for name, days in periods.items():
        start_date = max_date - pd.Timedelta(days=days)
        df_period = df[df['Last Trading Date'] >= start_date].copy()
        mfv_agg = df_period.groupby('Stock Code')['Money Flow Value'].sum()
        df_agg = pd.DataFrame(mfv_agg).join(latest_data.get('Close', pd.Series())).join(latest_data.get('Sector', pd.Series()))
        df_agg.columns = ['Total Money Flow (Rp)', 'Harga Terakhir', 'Sector']
        results[name] = df_agg.sort_values(by='Total Money Flow (Rp)', ascending=False).reset_index()
    return results['7D'], results['30D'], results['90D'], results['180D']

# --- FUNGSI BACKTESTING ---
def run_backtest_analysis(df, days_back=90):
    all_dates = sorted(df['Last Trading Date'].unique())
    if len(all_dates) < days_back: days_back = len(all_dates) - 30 
    start_idx = max(30, len(all_dates) - days_back)
    simulation_dates = all_dates[start_idx:]
    latest_prices = df[df['Last Trading Date'] == all_dates[-1]].set_index('Stock Code')['Close']
    
    backtest_log = []
    progress_bar = st.progress(0); status_text = st.empty(); total_steps = len(simulation_dates)
    
    for i, sim_date in enumerate(simulation_dates):
        pct = (i + 1) / total_steps; progress_bar.progress(pct)
        sim_date_ts = pd.Timestamp(sim_date)
        status_text.text(f"⏳ Mengaudit data: {sim_date_ts.strftime('%d-%m-%Y')}")
        try:
            top20, _, status = calculate_potential_score(df, sim_date_ts)
            if status == "success" and not top20.empty:
                for idx, row in top20.iterrows():
                    code = row['Stock Code']
                    entry_price = row['last_price']
                    curr_price = latest_prices.get(code, np.nan)
                    ret_pct = ((curr_price - entry_price) / entry_price * 100) if (pd.notna(curr_price) and entry_price > 0) else 0
                    backtest_log.append({'Signal Date': sim_date_ts, 'Stock Code': code, 'Entry Price': entry_price, 'Current Price': curr_price, 'Return to Date (%)': ret_pct, 'Score at Signal': row['Potential Score']})
        except Exception: continue
    progress_bar.empty(); status_text.empty()
    return pd.DataFrame(backtest_log)

# --- FUNGSI SIMULASI PORTFOLIO ---
def simulate_portfolio_range(df, capital, start_date_ts, end_date_ts):
    top20, _, status = calculate_potential_score(df, start_date_ts)
    if status != "success" or top20.empty: return pd.DataFrame(), None, "Gagal hitung score."
    
    df_end = df[df['Last Trading Date'] == end_date_ts]
    if df_end.empty: return pd.DataFrame(), None, "Data End Date kosong."
    
    exit_prices = df_end.set_index('Stock Code')['Close']
    allocation_per_stock = capital / len(top20)
    portfolio_results = []
    
    for idx, row in top20.iterrows():
        code = row['Stock Code']; buy_price = row['last_price']
        exit_price = exit_prices.get(code, np.nan)
        if pd.isna(exit_price) or buy_price <= 0:
            roi_pct = 0; final_val = allocation_per_stock; gain_rp = 0; exit_price_display = 0
        else:
            roi_pct = ((exit_price - buy_price) / buy_price)
            final_val = allocation_per_stock * (1 + roi_pct)
            gain_rp = final_val - allocation_per_stock
            exit_price_display = exit_price
        portfolio_results.append({'Stock Code': code, 'Sector': row['sector'], 'Buy Price': buy_price, 'Sell Price': exit_price_display, 'Gain/Loss (Rp)': gain_rp, 'Final Value': final_val, 'ROI (%)': roi_pct * 100})
        
    df_port = pd.DataFrame(portfolio_results)
    summary = {'Start Date': start_date_ts, 'End Date': end_date_ts, 'Initial Capital': capital, 'Final Portfolio Value': df_port['Final Value'].sum(), 'Net Profit': df_port['Gain/Loss (Rp)'].sum(), 'Total ROI': (df_port['Gain/Loss (Rp)'].sum() / capital) * 100}
    return df_port, summary, "success"

# --- FUNGSI MSCI SIMULATOR (OPTIMIZED) ---
@st.cache_data(ttl=3600)
def calculate_msci_projection_v2_optimized(df, latest_date, usd_rate):
    """
    Versi Optimized: Menggunakan GroupBy di awal untuk menghindari filter di dalam loop.
    Kecepatan naik 50x lipat.
    """
    # 1. Define time windows
    start_date_12m = latest_date - pd.Timedelta(days=365)
    start_date_3m = latest_date - pd.Timedelta(days=90)
    
    # 2. Filter data slices (Vectorized)
    df_12m = df[(df['Last Trading Date'] >= start_date_12m) & (df['Last Trading Date'] <= latest_date)]
    df_3m = df[(df['Last Trading Date'] >= start_date_3m) & (df['Last Trading Date'] <= latest_date)]
    df_last = df[df['Last Trading Date'] == latest_date].copy()
    
    # 3. Pre-calculate Totals (The Key Optimization)
    # Ini menghitung total value per saham hanya SEKALI
    val_12m_map = df_12m.groupby('Stock Code')['Value'].sum()
    val_3m_map = df_3m.groupby('Stock Code')['Value'].sum()
    
    results = []
    
    for idx, row in df_last.iterrows():
        code = row['Stock Code']
        close = row['Close']
        listed_shares = row.get('Listed Shares', 0)
        free_float_pct = row.get('Free Float', 0)
        
        # --- 1. SIZE (Market Cap) ---
        full_mcap_idr_t = (close * listed_shares) / 1e12 
        float_mcap_idr_t = full_mcap_idr_t * (free_float_pct / 100)
        
        # USD Conversion (Billions)
        full_mcap_usd_b = (full_mcap_idr_t * 1e12) / usd_rate / 1e9
        float_mcap_usd_b = (float_mcap_idr_t * 1e12) / usd_rate / 1e9
        
        # --- 2. LIQUIDITY (ATVR) ---
        # Ambil value dari map yang sudah dihitung (O(1) lookup)
        val_12m = val_12m_map.get(code, 0)
        val_3m = val_3m_map.get(code, 0)
        
        # Annualized Value
        annualized_val_3m = val_3m * 4 
        
        # Float Mcap (Full Rupiah)
        float_mcap_full = float_mcap_idr_t * 1e12
        
        # ATVR Calculation (%)
        if float_mcap_full > 0:
            atvr_12m = (val_12m / float_mcap_full * 100)
            atvr_3m = (annualized_val_3m / float_mcap_full * 100)
        else:
            atvr_12m = 0
            atvr_3m = 0
            
        results.append({
            'Stock Code': code, 
            'Close': close, 
            'Sector': row['Sector'],
            'Float Cap (IDR T)': float_mcap_idr_t,
            'Full Cap ($B)': full_mcap_usd_b,
            'Float Cap ($B)': float_mcap_usd_b,
            'ATVR 12M (%)': atvr_12m,
            'ATVR 3M (%)': atvr_3m
        })
        
    df_msci = pd.DataFrame(results)
    if not df_msci.empty:
        df_msci = df_msci.sort_values(by='Float Cap ($B)', ascending=False).reset_index(drop=True)
        df_msci['Rank'] = df_msci.index + 1
    return df_msci

# ==============================================================================
# 💎 5) LAYOUT UTAMA
# ==============================================================================
st.markdown("<div class='header-title'>📈 Dashboard Saham IDX</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Market Intelligence Platform | Technical • Flow • MSCI • Portfolio</div>", unsafe_allow_html=True)

df, status_msg, status_level = load_data()

if status_level == "success": st.toast(status_msg, icon="✅")
elif status_level == "error": st.error(status_msg); st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🎛️ Navigasi")
    if st.button("🔄 Refresh Data", use_container_width=True): 
        st.cache_data.clear()
        st.rerun()

    max_date = df['Last Trading Date'].max().date()
    selected_date = st.date_input("Pilih Tanggal Analisis", max_date, min_value=df['Last Trading Date'].min().date(), max_value=max_date, format="DD-MM-YYYY")
    df_day = df[df['Last Trading Date'].dt.date == selected_date].copy()

    st.markdown("---")
    st.markdown("### 🔍 Filter (Tab 3)")
    selected_stocks_filter = st.multiselect("Saham", sorted(df_day["Stock Code"].unique()))
    selected_sectors_filter = st.multiselect("Sektor", sorted(df_day.get("Sector", pd.Series(dtype='object')).dropna().unique()))
    
    df_filtered = df_day.copy()
    if selected_stocks_filter: df_filtered = df_filtered[df_filtered["Stock Code"].isin(selected_stocks_filter)]
    if selected_sectors_filter: df_filtered = df_filtered[df_filtered["Sector"].isin(selected_sectors_filter)]

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Market Dashboard", "📈 Analisis Individual", "📋 Data Filter",
    "🏆 Top 20 Potensial", "🌊 Net Foreign Flow", "💰 Money Flow",
    "🧪 Backtest Lab", "💼 Portfolio Sim", "🌏 MSCI Simulator"
])

# --- TAB 1: DAILY DASHBOARD ---
with tab1:
    st.markdown("#### Ringkasan Pasar")
    c1, c2, c3 = st.columns(3)
    c1.metric("Saham Aktif", f"{len(df_day):,.0f}")
    c2.metric("Unusual Volume", f"{df_day['Unusual Volume'].sum():,.0f}")
    c3.metric("Total Value", f"Rp {df_day['Value'].sum()/1e9:,.1f} M")
    
    st.markdown("---")
    col_g, col_l, col_v = st.columns(3)
    
    def show_top_card(title, data, sort_col, asc, val_col, fmt_func):
        st.markdown(f"**{title}**")
        top = data.sort_values(sort_col, ascending=asc).head(10)[['Stock Code', 'Close', val_col]].reset_index(drop=True)
        top['Close'] = top['Close'].apply(lambda x: f"{x:,.0f}")
        top[val_col] = top[val_col].apply(fmt_func)
        st.dataframe(top, use_container_width=True, hide_index=True)

    with col_g: show_top_card("🚀 Top Gainers", df_day, "Change %", False, "Change %", lambda x: f"{x:+.2f}%")
    with col_l: show_top_card("🔻 Top Losers", df_day, "Change %", True, "Change %", lambda x: f"{x:+.2f}%")
    with col_v: show_top_card("💰 Top Value", df_day, "Value", False, "Value", lambda x: f"{x/1e9:,.1f} M")

# --- TAB 2: INDIVIDUAL ---
with tab2:
    st.markdown("#### Deep Dive Analysis")
    all_stocks = sorted(df["Stock Code"].unique())
    col_sel, col_empty = st.columns([1, 2])
    with col_sel:
        stock = st.selectbox("Pilih Saham", all_stocks, index=all_stocks.index("BBRI") if "BBRI" in all_stocks else 0)
    
    if stock:
        df_stock = df[df['Stock Code'] == stock].sort_values('Last Trading Date')
        if not df_stock.empty:
            lr = df_stock.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Harga", f"Rp {lr['Close']:,.0f}")
            c2.metric("NFF (Rp)", f"Rp {lr['NFF (Rp)']:,.0f}")
            c3.metric("MFV (Rp)", f"Rp {lr['Money Flow Value']:,.0f}")
            c4.metric("MF Ratio (20D)", f"{lr.get('Money Flow Ratio (20D)', 0):.3f}")
            
            # Advanced Chart
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                               row_heights=[0.4, 0.2, 0.2, 0.2], specs=[[{"secondary_y": True}], [{}], [{}], [{}]])
            
            # Price & MA
            fig.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['Close'], name='Close', line=dict(color='#0d6efd')), row=1, col=1, secondary_y=True)
            if 'MA20' in df_stock.columns:
                 fig.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['MA20'], name='MA20', line=dict(color='orange', width=1)), row=1, col=1, secondary_y=True)

            # NFF
            colors_nff = np.where(df_stock['NFF (Rp)'] >= 0, '#198754', '#dc3545')
            fig.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['NFF (Rp)'], name='Net Foreign Flow', marker_color=colors_nff), row=1, col=1)
            
            # MFV
            colors_mfv = np.where(df_stock['Money Flow Value'] >= 0, '#0dcaf0', '#fd7e14')
            fig.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['Money Flow Value'], name='Money Flow Val', marker_color=colors_mfv), row=2, col=1)
            
            # Volume
            fig.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['Volume'], name='Volume', marker_color='#adb5bd'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['MA20_vol'], name='MA20 Vol', line=dict(color='black', width=1, dash='dot')), row=3, col=1)
            
            # MF Ratio
            if 'Money Flow Ratio (20D)' in df_stock.columns:
                fig.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['Money Flow Ratio (20D)'], name='MF Ratio', line=dict(color='#6610f2')), row=4, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
            
            fig.update_layout(height=900, template='plotly_white', margin=dict(t=30, b=10, l=10, r=10), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: DATA FILTER ---
with tab3: 
    st.markdown("#### Data Mentah Terfilter")
    st.dataframe(df_filtered, use_container_width=True)

# --- TAB 4: TOP 20 ---
with tab4:
    st.markdown("#### 🏆 Top 20 Saham Potensial (Scoring System)")
    st.caption("Skor berbasis Trend (30D), Momentum (7D), NBSA, dan Unusual Volume.")
    
    @st.cache_data(ttl=3600)
    def get_cached_top20(dframe, tgl): return calculate_potential_score(dframe, tgl)
    
    df_top20, msg, status = get_cached_top20(df, pd.Timestamp(selected_date))
    
    if status == "success":
        st.dataframe(
            df_top20, 
            use_container_width=True, 
            hide_index=True, 
            column_config={
                "Potential Score": st.column_config.ProgressColumn("Skor Akhir", format="%.2f", min_value=0, max_value=100),
                "last_price": st.column_config.NumberColumn("Close", format="%d"),
                "total_net_ff_rp": st.column_config.NumberColumn("NFF 30D", format="Rp %.0f")
            }
        )
    else: 
        st.warning(msg)

# --- TAB 5 & 6 (FLOW) ---
with tab5:
    st.markdown("#### Analisis Net Foreign Flow (Accumulation)")
    nff7, nff30, nff90, nff180 = calculate_nff_top_stocks(df, pd.Timestamp(selected_date))
    c1, c2 = st.columns(2)
    c1.markdown("**Top NFF 7 Hari**"); c1.dataframe(nff7.head(10), hide_index=True, use_container_width=True)
    c2.markdown("**Top NFF 30 Hari**"); c2.dataframe(nff30.head(10), hide_index=True, use_container_width=True)

with tab6:
    st.markdown("#### Analisis Money Flow Value (Big Money)")
    mfv7, mfv30, mfv90, mfv180 = calculate_mfv_top_stocks(df, pd.Timestamp(selected_date))
    c1, c2 = st.columns(2)
    c1.markdown("**Top MFV 7 Hari**"); c1.dataframe(mfv7.head(10), hide_index=True, use_container_width=True)
    c2.markdown("**Top MFV 30 Hari**"); c2.dataframe(mfv30.head(10), hide_index=True, use_container_width=True)

# --- TAB 7: BACKTEST ---
with tab7:
    st.markdown("#### 🧪 Backtest Logic")
    col_bt1, col_bt2 = st.columns([1, 3])
    with col_bt1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        days_to_test = st.number_input("Hari Backtest", 7, 90, 30, 7)
        run_btn = st.button("🚀 Jalankan Audit", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    if run_btn:
        with st.spinner("Menjalankan simulasi historical..."): 
            df_backtest = run_backtest_analysis(df, days_back=days_to_test)
        
        if not df_backtest.empty:
            total_s = len(df_backtest)
            win_s = len(df_backtest[df_backtest['Return to Date (%)'] > 0])
            wr = (win_s/total_s)*100
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Sinyal", f"{total_s}x")
            k2.metric("Win Rate", f"{wr:.1f}%")
            k3.metric("Avg Return", f"{df_backtest['Return to Date (%)'].mean():.2f}%")
            
            st.markdown("### ⭐ Leaderboard Saham")
            freq = df_backtest.groupby('Stock Code').agg(
                Freq=('Signal Date','count'), 
                Avg_Ret=('Return to Date (%)','mean'), 
                Last_Price=('Current Price','last')
            ).reset_index().sort_values(['Freq','Avg_Ret'], ascending=[False,False])
            
            st.dataframe(
                freq, use_container_width=True, hide_index=True, 
                column_config={
                    "Freq": st.column_config.ProgressColumn("Frekuensi", format="%d x", max_value=int(freq['Freq'].max())), 
                    "Avg_Ret": st.column_config.NumberColumn("Avg Return", format="%.2f %%")
                }
            )
            with st.expander("Lihat Log Detail"): st.dataframe(df_backtest.sort_values('Signal Date', ascending=False))
        else: st.warning("Tidak ada data sinyal dalam periode tersebut.")

# --- TAB 8: PORTFOLIO ---
with tab8:
    st.markdown("#### 💼 Simulasi Portfolio (Real PnL)")
    avail_dates = sorted(df['Last Trading Date'].unique())
    
    with st.form("port_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1: start_d = st.selectbox("📅 Beli", avail_dates, index=max(0, len(avail_dates)-31), format_func=lambda x: pd.Timestamp(x).strftime('%d-%m-%Y'))
        with c2: end_d = st.selectbox("📅 Jual", avail_dates, index=len(avail_dates)-1, format_func=lambda x: pd.Timestamp(x).strftime('%d-%m-%Y'))
        with c3: cap = st.number_input("💰 Modal (Rp)", 1_000_000, 1_000_000_000, 20_000_000, 1_000_000)
        with c4: 
            st.write("")
            btn_calc = st.form_submit_button("🚀 Hitung Profit")
    
    if btn_calc:
        if pd.Timestamp(start_d) >= pd.Timestamp(end_d): st.error("Tanggal Jual harus setelah Beli.")
        else:
            with st.spinner("Menghitung PnL..."): 
                df_port, sum_port, msg = simulate_portfolio_range(df, cap, pd.Timestamp(start_d), pd.Timestamp(end_d))
            
            if msg == "success":
                m1, m2, m3 = st.columns(3)
                m1.metric("Modal Awal", f"Rp {sum_port['Initial Capital']:,.0f}")
                m2.metric("Saldo Akhir", f"Rp {sum_port['Final Portfolio Value']:,.0f}")
                m3.metric("Net Profit", f"Rp {sum_port['Net Profit']:,.0f}", delta=f"{sum_port['Total ROI']:.2f}%")
                
                c_ch1, c_ch2 = st.columns([2,1])
                with c_ch1: st.plotly_chart(px.bar(df_port.sort_values('Gain/Loss (Rp)', ascending=False), x='Stock Code', y='Gain/Loss (Rp)', color='Gain/Loss (Rp)', color_continuous_scale=['red', 'green'], title="PnL per Saham"), use_container_width=True)
                with c_ch2: st.plotly_chart(px.pie(df_port, names='Sector', values='Final Value', title="Alokasi Sektor"), use_container_width=True)
                
                st.dataframe(df_port[['Stock Code', 'Buy Price', 'Sell Price', 'Gain/Loss (Rp)', 'ROI (%)']], use_container_width=True, hide_index=True)
            else: st.error(msg)

# --- TAB 9: MSCI SIMULATOR (OPTIMIZED) ---
with tab9:
    st.markdown("#### 🌏 MSCI Indonesia Index Simulator (Proxy)")
    st.info("Prediksi kandidat MSCI Standard Index. **Cutoff:** Float Cap > $1.5B & Liquidity (ATVR) > 15%.")
    
    if 'Listed Shares' not in df.columns: st.error("❌ Data 'Listed Shares' tidak ditemukan di dataset.")
    else:
        # Input Kurs & Parameter
        with st.expander("⚙️ Parameter MSCI", expanded=True):
            c_p1, c_p2 = st.columns(2)
            with c_p1: 
                usd_idr = st.number_input("Kurs USD/IDR", value=16500, step=50)
            with c_p2:
                cut_float_usd = st.number_input("Min. Float Market Cap ($B)", value=1.5, step=0.1)
                cut_atvr = st.number_input("Min. Liquidity (ATVR %)", value=15.0, step=1.0)

        # Hitung Data (Versi Cepat)
        with st.spinner("Calculating MSCI Universe (Optimized)..."):
            df_msci = calculate_msci_projection_v2_optimized(df, df['Last Trading Date'].max(), usd_idr)
        
        # Categorization Logic
        def cat_msci(r):
            pass_size = (r['Float Cap ($B)'] >= cut_float_usd)
            pass_liq = (r['ATVR 12M (%)'] >= cut_atvr) and (r['ATVR 3M (%)'] >= cut_atvr)
            
            if pass_size and pass_liq: return "✅ Potential Standard"
            elif pass_size and not pass_liq: return "⚠️ Risk Deletion (Illiquid)"
            elif (r['Float Cap ($B)'] >= (cut_float_usd * 0.5)) and pass_liq: return "🔹 Small Cap Potential"
            return "🔻 Micro / Others"
        
        if not df_msci.empty:
            df_msci['Status'] = df_msci.apply(cat_msci, axis=1)
            
            # --- VISUALISASI ---
            # Ambil Top 100 biar chart readable
            df_chart = df_msci.head(100)
            
            fig_msci = px.scatter(
                df_chart, 
                x="ATVR 12M (%)", 
                y="Float Cap ($B)", 
                color="Status", 
                size="Full Cap ($B)", 
                hover_data=['Stock Code', 'ATVR 3M (%)'], 
                text="Stock Code", 
                color_discrete_map={
                    "✅ Potential Standard": "#198754", 
                    "⚠️ Risk Deletion (Illiquid)": "#dc3545", 
                    "🔹 Small Cap Potential": "#0d6efd", 
                    "🔻 Micro / Others": "#adb5bd"
                },
                title=f"Peta Kekuatan: Liquidity vs Float Size (Min Float ${cut_float_usd}B)"
            )
            fig_msci.add_hline(y=cut_float_usd, line_dash="dash", annotation_text=f"Min Float")
            fig_msci.add_vline(x=cut_atvr, line_dash="dash", annotation_text=f"Min Liq")
            st.plotly_chart(fig_msci, use_container_width=True)
            
            # --- TABEL HASIL ---
            col_t1, col_t2 = st.columns(2)
            show_cols = ['Rank', 'Stock Code', 'Status', 'Float Cap ($B)', 'ATVR 3M (%)', 'ATVR 12M (%)']
            
            with col_t1: 
                st.markdown("#### ✅ Top Candidates")
                st.dataframe(
                    df_msci[df_msci['Status']=="✅ Potential Standard"][show_cols], 
                    hide_index=True, use_container_width=True,
                    column_config={
                        "Float Cap ($B)": st.column_config.NumberColumn(format="$ %.2f B"),
                        "ATVR 3M (%)": st.column_config.NumberColumn(format="%.1f %%")
                    }
                )
            
            with col_t2: 
                st.markdown("#### ⚠️ Watchlist: Illiquid Risk")
                st.dataframe(
                    df_msci[df_msci['Status']=="⚠️ Risk Deletion (Illiquid)"][show_cols], 
                    hide_index=True, use_container_width=True,
                    column_config={
                        "Float Cap ($B)": st.column_config.NumberColumn(format="$ %.2f B"),
                        "ATVR 3M (%)": st.column_config.NumberColumn(format="%.1f %%")
                    }
                )
