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
    page_title="Terminal Saham Pro v2",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# --- ✨ MODERN PREMIUM THEME (LIGHT MODE) ---
st.markdown("""
<style>
    /* 1. Global Setup */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp {
        background-color: #F3F4F6; /* Light Gray Background */
        color: #1F2937; /* Dark Gray Text */
        font-family: 'Inter', sans-serif;
    }

    /* 2. Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    
    /* 3. Cards (Container) */
    .css-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* 4. Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 8px;
        # border: 1px solid #F3F4F6;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #6B7280 !important; /* Cool Gray */
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #111827 !important; /* Almost Black */
    }
    div[data-testid="stMetricDelta"] {
        font-weight: 600;
    }

    /* 5. Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        background-color: #FFFFFF;
    }

    /* 6. Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        color: #6B7280;
        border: 1px solid #E5E7EB;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important; /* Royal Blue */
        color: #FFFFFF !important;
        border-color: #2563EB !important;
    }

    /* 7. Buttons */
    div.stButton > button {
        background-color: #2563EB;
        color: white !important;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
    }

    /* 8. Custom Headers */
    .header-title { 
        font-size: 36px; 
        font-weight: 800; 
        color: #111827; 
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .header-subtitle { 
        font-size: 16px; 
        color: #6B7280; 
        font-weight: 500;
        margin-bottom: 32px; 
    }
    
    /* 9. Utilities */
    hr { margin: 2em 0; border-color: #E5E7EB; }
</style>
""", unsafe_allow_html=True)

# --- KONFIGURASI G-DRIVE ---
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP" 
FILE_NAME = "Kompilasi_Data_1Tahun.csv"

# ==============================================================================
# 📦 3) FUNGSI MEMUAT DATA (FIXED ROBUST)
# ==============================================================================
def get_gdrive_service():
    try:
        if "gcp_service_account" not in st.secrets:
            return None, "❌ Key 'gcp_service_account' missing in secrets."
        creds_data = st.secrets["gcp_service_account"]
        creds_json = creds_data.to_dict() if hasattr(creds_data, "to_dict") else dict(creds_data)
        if "private_key" in creds_json:
            pk = str(creds_json["private_key"])
            if "\\n" in pk: creds_json["private_key"] = pk.replace("\\n", "\n")
        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service, None
    except Exception as e: return None, f"❌ Auth Error: {e}"

@st.cache_data(ttl=3600, show_spinner="🔄 Memuat Data Pasar...")
def load_data():
    service, error_msg = get_gdrive_service()
    if error_msg: return pd.DataFrame(), error_msg, "error"
    try:
        results = service.files().list(q=f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false", fields="files(id, name)", orderBy="modifiedTime desc", pageSize=1).execute()
        items = results.get('files', [])
        if not items: return pd.DataFrame(), f"❌ File '{FILE_NAME}' not found.", "error"
        
        request = service.files().get_media(fileId=items[0]['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: status, done = downloader.next_chunk()
        fh.seek(0)
        
        df = pd.read_csv(fh, dtype=object)
        df.columns = df.columns.str.strip()
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
        
        # Numeric Conversion
        cols_to_numeric = [
            'High', 'Low', 'Close', 'Volume', 'Value', 'Foreign Buy', 'Foreign Sell', 'Bid Volume', 'Offer Volume', 
            'Change', 'Open Price', 'Listed Shares', 'Change %', 'Typical Price', 'Net Foreign Flow', 'Money Flow Value', 
            'Free Float', 'Volume Spike (x)', 'Money Flow Ratio (20D)'
        ]
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.strip().str.replace(r'[,\sRp\%]', '', regex=True), errors='coerce').fillna(0)
        
        # --- FIX: LOGIKA UNUSUAL VOLUME ---
        # Kita buat kolom ini JIKA belum ada, atau bersihkan JIKA sudah ada
        if 'Unusual Volume' not in df.columns:
            # Skenario 1: Ambil dari 'Volume Spike (x)' > 2
            if 'Volume Spike (x)' in df.columns:
                df['Unusual Volume'] = df['Volume Spike (x)'] > 2.0
            # Skenario 2: Ambil dari 'Big_Player_Anomaly'
            elif 'Big_Player_Anomaly' in df.columns:
                df['Unusual Volume'] = df['Big_Player_Anomaly'].astype(str).str.strip().str.lower() == 'true'
            else:
                df['Unusual Volume'] = False # Default False
        else:
            # Jika kolom sudah ada, bersihkan formatnya
            df['Unusual Volume'] = df['Unusual Volume'].astype(str).str.strip().str.lower().isin(['spike volume signifikan', 'true', '1'])
        
        # --- FIX: LOGIKA FINAL SIGNAL & SECTOR ---
        if 'Final Signal' in df.columns:
            df['Final Signal'] = df['Final Signal'].astype(str).str.strip()
        else:
            df['Final Signal'] = 'Neutral'

        if 'Sector' in df.columns:
            df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
        else:
            df['Sector'] = 'Others'

        df = df.dropna(subset=['Last Trading Date', 'Stock Code'])
        
        if 'NFF (Rp)' not in df.columns:
            if 'Net Foreign Flow' in df.columns:
                price_col = 'Typical Price' if 'Typical Price' in df.columns else 'Close'
                df['NFF (Rp)'] = df['Net Foreign Flow'] * df[price_col]
            else:
                df['NFF (Rp)'] = 0
            
        return df, "Market Data Synced", "success"
    except Exception as e: return pd.DataFrame(), f"❌ Load Error: {e}", "error"

# ==============================================================================
# 🛠️ 4) FUNGSI KALKULASI UTAMA
# ==============================================================================
def pct_rank(s): return pd.to_numeric(s, errors="coerce").rank(pct=True, method="average").fillna(0) * 100

def calculate_potential_score(df, latest_date):
    trend_start, mom_start = latest_date - pd.Timedelta(days=30), latest_date - pd.Timedelta(days=7)
    df_hist = df[df['Last Trading Date'] <= latest_date]
    trend_df = df_hist[df_hist['Last Trading Date'] >= trend_start]
    mom_df = df_hist[df_hist['Last Trading Date'] >= mom_start]
    
    if trend_df.empty: return pd.DataFrame(), "Data insufficient", "warning"
    
    tr = trend_df.groupby('Stock Code').agg(
        last_price=('Close', 'last'), 
        total_net_ff_rp=('NFF (Rp)', 'sum'), 
        total_money_flow=('Money Flow Value', 'sum'),
        avg_change_pct=('Change %', 'mean'), 
        sector=('Sector', 'last')
    ).reset_index()
    
    mo = mom_df.groupby('Stock Code').agg(
        total_change_pct=('Change %', 'sum'), 
        had_unusual_volume=('Unusual Volume', 'any'),
        total_net_ff_rp=('NFF (Rp)', 'sum')
    ).reset_index()
    
    tr['Trend Score'] = pct_rank(tr['total_net_ff_rp']) * 0.5 + pct_rank(tr['total_money_flow']) * 0.5
    mo['Momentum Score'] = pct_rank(mo['total_change_pct']) * 0.6 + pct_rank(mo['total_net_ff_rp']) * 0.4
    
    rank = tr.merge(mo[['Stock Code', 'Momentum Score', 'had_unusual_volume']], on='Stock Code', how='outer')
    rank['Potential Score'] = rank['Trend Score'].fillna(0)*0.5 + rank['Momentum Score'].fillna(0)*0.5
    
    if 'had_unusual_volume' in rank.columns:
        bonus = rank['had_unusual_volume'].fillna(False).astype(int) * 10
        rank['Potential Score'] += bonus

    top20 = rank.sort_values('Potential Score', ascending=False).head(20).copy()
    return top20, "Success", "success"

@st.cache_data(ttl=3600)
def calculate_nff_summary_enhanced(df, max_date):
    periods = {'1 Bulan': 30, '3 Bulan': 90, '6 Bulan': 180}
    results = {}
    latest_data = df[df['Last Trading Date'] == max_date].set_index('Stock Code')
    
    for name, days in periods.items():
        start_date = max_date - pd.Timedelta(days=days)
        df_period = df[(df['Last Trading Date'] >= start_date) & (df['Last Trading Date'] <= max_date)].copy()
        
        agg = df_period.groupby('Stock Code').agg(
            Total_Net_Buy=('NFF (Rp)', 'sum'),
            Trading_Days=('Last Trading Date', 'nunique'),
            Pos_Days=('NFF (Rp)', lambda x: (x > 0).sum())
        )
        agg['Konsistensi (%)'] = (agg['Pos_Days'] / agg['Trading_Days']) 
        df_final = agg.join(latest_data[['Close', 'Sector']], how='inner').reset_index()
        df_final = df_final[df_final['Total_Net_Buy'] > 0]
        results[name] = df_final.sort_values(by='Total_Net_Buy', ascending=False)
        
    return results

@st.cache_data(ttl=3600)
def calculate_mfv_top_stocks(df, max_date):
    periods = {'7D': 7, '30D': 30}; results = {}
    latest_data = df[df['Last Trading Date'] == max_date].set_index('Stock Code')
    
    for name, days in periods.items():
        start_date = max_date - pd.Timedelta(days=days)
        df_period = df[df['Last Trading Date'] >= start_date].copy()
        mfv_agg = df_period.groupby('Stock Code')['Money Flow Value'].sum()
        df_agg = pd.DataFrame(mfv_agg).join(latest_data.get('Close', pd.Series())).join(latest_data.get('Sector', pd.Series()))
        df_agg.columns = ['Total Money Flow (Rp)', 'Harga Terakhir', 'Sector']
        results[name] = df_agg.sort_values(by='Total Money Flow (Rp)', ascending=False).reset_index()
    return results['7D'], results['30D']

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
        status_text.text(f"⏳ Auditing: {sim_date_ts.strftime('%d-%m-%Y')}")
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

@st.cache_data(ttl=3600)
def calculate_msci_projection_v2_optimized(df, latest_date, usd_rate):
    start_date_12m = latest_date - pd.Timedelta(days=365)
    start_date_3m = latest_date - pd.Timedelta(days=90)
    
    df_12m = df[(df['Last Trading Date'] >= start_date_12m) & (df['Last Trading Date'] <= latest_date)]
    df_3m = df[(df['Last Trading Date'] >= start_date_3m) & (df['Last Trading Date'] <= latest_date)]
    df_last = df[df['Last Trading Date'] == latest_date].copy()
    
    val_12m_map = df_12m.groupby('Stock Code')['Value'].sum()
    val_3m_map = df_3m.groupby('Stock Code')['Value'].sum()
    
    results = []
    for idx, row in df_last.iterrows():
        code = row['Stock Code']; close = row['Close']
        listed_shares = row.get('Listed Shares', 0)
        free_float_pct = row.get('Free Float', 0)
        
        full_mcap_idr_t = (close * listed_shares) / 1e12 
        float_mcap_idr_t = full_mcap_idr_t * (free_float_pct / 100)
        
        full_mcap_usd_b = (full_mcap_idr_t * 1e12) / usd_rate / 1e9
        float_mcap_usd_b = (float_mcap_idr_t * 1e12) / usd_rate / 1e9
        
        val_12m = val_12m_map.get(code, 0)
        val_3m = val_3m_map.get(code, 0)
        annualized_val_3m = val_3m * 4 
        float_mcap_full = float_mcap_idr_t * 1e12
        
        if float_mcap_full > 0:
            atvr_12m = (val_12m / float_mcap_full * 100)
            atvr_3m = (annualized_val_3m / float_mcap_full * 100)
        else:
            atvr_12m = 0; atvr_3m = 0
            
        results.append({
            'Stock Code': code, 'Close': close, 'Sector': row['Sector'],
            'Float Cap (IDR T)': float_mcap_idr_t, 'Full Cap ($B)': full_mcap_usd_b,
            'Float Cap ($B)': float_mcap_usd_b, 'ATVR 12M (%)': atvr_12m, 'ATVR 3M (%)': atvr_3m
        })
        
    df_msci = pd.DataFrame(results)
    if not df_msci.empty:
        df_msci = df_msci.sort_values(by='Float Cap ($B)', ascending=False).reset_index(drop=True)
        df_msci['Rank'] = df_msci.index + 1
    return df_msci

# ==============================================================================
# 💎 5) LAYOUT UTAMA (PREMIUM LIGHT)
# ==============================================================================
st.markdown("<div class='header-title'>TERMINAL SAHAM PRO</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Advanced Market Intelligence • Flow Analysis • MSCI Proxy</div>", unsafe_allow_html=True)

try:
    df, status_msg, status_level = load_data()
except Exception as e:
    st.error(f"Critical Data Error: {e}")
    st.stop()

if status_level == "error": 
    st.error(status_msg)
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🎛️ SYSTEM CONTROL")
    if st.button("🔄 REFRESH DATA", use_container_width=True): 
        st.cache_data.clear()
        st.rerun()
        
    max_date = df['Last Trading Date'].max().date()
    selected_date = st.date_input("ANALYSIS DATE", max_date, min_value=df['Last Trading Date'].min().date(), max_value=max_date, format="DD-MM-YYYY")
    df_day = df[df['Last Trading Date'].dt.date == selected_date].copy()
    
    st.markdown("---")
    st.markdown("### 🔍 WATCHLIST FILTER")
    selected_stocks_filter = st.multiselect("Stock Ticker", sorted(df_day["Stock Code"].unique()))
    selected_sectors_filter = st.multiselect("Sector", sorted(df_day.get("Sector", pd.Series(dtype='object')).dropna().unique()))
    
    df_filtered = df_day.copy()
    if selected_stocks_filter: df_filtered = df_filtered[df_filtered["Stock Code"].isin(selected_stocks_filter)]
    if selected_sectors_filter: df_filtered = df_filtered[df_filtered["Sector"].isin(selected_sectors_filter)]

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 DASHBOARD", "📈 DEEP DIVE", "📋 RAW DATA",
    "🏆 TOP 20", "🌊 NET FLOW (ASING)", "💰 MONEY FLOW",
    "🧪 BACKTEST", "💼 PORTFOLIO", "🌏 MSCI PROXY"
])

# --- TAB 1: DAILY DASHBOARD ---
with tab1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("ACTIVE STOCKS", f"{len(df_day):,.0f}")
    
    # Safely get unusual volume sum
    unusual_sum = df_day['Unusual Volume'].sum() if 'Unusual Volume' in df_day.columns else 0
    c2.metric("UNUSUAL VOLUME", f"{unusual_sum:,.0f}")
    
    c3.metric("TOTAL VALUE", f"Rp {df_day['Value'].sum()/1e9:,.1f} M")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_g, col_l, col_v = st.columns(3)
    def show_top_card(title, data, sort_col, asc, val_col, fmt):
        with st.container():
            st.markdown(f"**{title}**")
            top = data.sort_values(sort_col, ascending=asc).head(10)[['Stock Code', 'Close', val_col]].reset_index(drop=True)
            st.dataframe(top, use_container_width=True, hide_index=True, column_config={
                "Close": st.column_config.NumberColumn(format="%d"),
                val_col: st.column_config.NumberColumn(format=fmt)
            })

    with col_g: show_top_card("🚀 TOP GAINERS", df_day, "Change %", False, "Change %", "%.2f %%")
    with col_l: show_top_card("🔻 TOP LOSERS", df_day, "Change %", True, "Change %", "%.2f %%")
    with col_v: show_top_card("💰 TOP VALUE", df_day, "Value", False, "Value", "Rp %.0f")

# --- TAB 2: INDIVIDUAL ---
with tab2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    all_stocks = sorted(df["Stock Code"].unique())
    col_sel, _ = st.columns([1, 2])
    with col_sel: stock = st.selectbox("SEARCH STOCK", all_stocks, index=all_stocks.index("BBRI") if "BBRI" in all_stocks else 0)
    
    if stock:
        df_stock = df[df['Stock Code'] == stock].sort_values('Last Trading Date')
        if not df_stock.empty:
            lr = df_stock.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CLOSE", f"Rp {lr['Close']:,.0f}")
            c2.metric("NFF (Rp)", f"Rp {lr['NFF (Rp)']:,.0f}")
            c3.metric("MFV (Rp)", f"Rp {lr['Money Flow Value']:,.0f}")
            
            raw_mf = lr.get('Money Flow Ratio (20D)', 0)
            try: mf_val = float(raw_mf)
            except: mf_val = 0.0
            c4.metric("MF RATIO", f"{mf_val:.3f}")
            
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.2, 0.2, 0.2], specs=[[{"secondary_y": True}], [{}], [{}], [{}]])
            fig.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['Close'], name='Close', line=dict(color='#2563EB', width=2)), row=1, col=1, secondary_y=True)
            
            colors_nff = np.where(df_stock['NFF (Rp)'] >= 0, '#10B981', '#EF4444')
            fig.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['NFF (Rp)'], name='NFF', marker_color=colors_nff), row=1, col=1)
            
            colors_mfv = np.where(df_stock['Money Flow Value'] >= 0, '#3B82F6', '#F59E0B')
            fig.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['Money Flow Value'], name='MFV', marker_color=colors_mfv), row=2, col=1)
            
            fig.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['Volume'], name='Volume', marker_color='#9CA3AF'), row=3, col=1)
            
            if 'Money Flow Ratio (20D)' in df_stock.columns:
                fig.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['Money Flow Ratio (20D)'], name='MF Ratio', line=dict(color='#8B5CF6')), row=4, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
            
            # Update Layout for Light Theme
            fig.update_layout(
                height=900, 
                template='plotly_white', 
                margin=dict(t=30, b=10, l=10, r=10), 
                hovermode="x unified", 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: DATA FILTER ---
with tab3: 
    st.markdown("#### FILTERED RAW DATA")
    st.dataframe(df_filtered, use_container_width=True, column_config={
        "Close": st.column_config.NumberColumn(format="%d"),
        "Volume": st.column_config.NumberColumn(format="%d"),
        "Value": st.column_config.NumberColumn(format="Rp %.0f")
    })

# --- TAB 4: TOP 20 ---
with tab4:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("#### 🏆 TOP 20 POTENTIAL STOCKS")
    df_top20, msg, status = calculate_potential_score(df, pd.Timestamp(selected_date))
    if status == "success":
        st.dataframe(df_top20, use_container_width=True, hide_index=True, column_config={
            "Potential Score": st.column_config.ProgressColumn("Final Score", format="%.2f", min_value=0, max_value=100),
            "last_price": st.column_config.NumberColumn("Close", format="%d"),
            "total_net_ff_rp": st.column_config.NumberColumn("NFF 30D", format="Rp %.0f")
        })
    else: st.warning(msg)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 5: NET FOREIGN FLOW ---
with tab5:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("#### 🌊 NET FOREIGN FLOW SUMMARY")
    nff_summary = calculate_nff_summary_enhanced(df, pd.Timestamp(selected_date))
    
    def show_nff_table(data):
        st.dataframe(
            data.head(20), 
            hide_index=True, 
            use_container_width=True, 
            column_config={
                "Close": st.column_config.NumberColumn(format="%d"),
                "Total_Net_Buy": st.column_config.ProgressColumn("Net Buy (Rp)", format="Rp %.0f", min_value=0, max_value=data['Total_Net_Buy'].max()),
                "Pos_Days": st.column_config.NumberColumn("Freq (Hari)", format="%d"),
                "Konsistensi (%)": st.column_config.NumberColumn("Konsistensi", format="%.1f %%")
            }
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🗓️ 1 BULAN (30 Hari)**")
        show_nff_table(nff_summary['1 Bulan'])
    with c2:
        st.markdown("**🗓️ 3 BULAN (90 Hari)**")
        show_nff_table(nff_summary['3 Bulan'])
    with c3:
        st.markdown("**🗓️ 6 BULAN (180 Hari)**")
        show_nff_table(nff_summary['6 Bulan'])
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 6: MONEY FLOW ---
with tab6:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("#### 💰 MONEY FLOW VALUE (BIG MONEY)")
    mfv7, mfv30 = calculate_mfv_top_stocks(df, pd.Timestamp(selected_date))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top MFV 7 Days**")
        st.dataframe(mfv7.head(10), hide_index=True, use_container_width=True, column_config={"Total Money Flow (Rp)": st.column_config.NumberColumn(format="Rp %.0f"), "Harga Terakhir": st.column_config.NumberColumn(format="%d")})
    with c2:
        st.markdown("**Top MFV 30 Days**")
        st.dataframe(mfv30.head(10), hide_index=True, use_container_width=True, column_config={"Total Money Flow (Rp)": st.column_config.NumberColumn(format="Rp %.0f"), "Harga Terakhir": st.column_config.NumberColumn(format="%d")})
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 7: BACKTEST ---
with tab7:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    col_bt1, col_bt2 = st.columns([1, 3])
    with col_bt1:
        days_to_test = st.number_input("Lookback Days", 7, 90, 30, 7)
        run_btn = st.button("🚀 RUN AUDIT", type="primary", use_container_width=True)
    if run_btn:
        with st.spinner("Processing..."): 
            df_backtest = run_backtest_analysis(df, days_back=days_to_test)
        if not df_backtest.empty:
            total_s = len(df_backtest)
            win_s = len(df_backtest[df_backtest['Return to Date (%)'] > 0])
            k1, k2, k3 = st.columns(3)
            k1.metric("SIGNALS", f"{total_s}x")
            k2.metric("WIN RATE", f"{(win_s/total_s)*100:.1f}%")
            k3.metric("AVG RETURN", f"{df_backtest['Return to Date (%)'].mean():.2f}%")
            st.dataframe(df_backtest, use_container_width=True, column_config={
                "Entry Price": st.column_config.NumberColumn(format="%d"),
                "Current Price": st.column_config.NumberColumn(format="%d"),
                "Return to Date (%)": st.column_config.NumberColumn(format="%.2f %%")
            })
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 8: PORTFOLIO ---
with tab8:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("#### 💼 PORTFOLIO SIMULATOR")
    avail_dates = sorted(df['Last Trading Date'].unique())
    with st.form("port_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1: start_d = st.selectbox("BUY DATE", avail_dates, index=max(0, len(avail_dates)-31), format_func=lambda x: pd.Timestamp(x).strftime('%d-%m-%Y'))
        with c2: end_d = st.selectbox("SELL DATE", avail_dates, index=len(avail_dates)-1, format_func=lambda x: pd.Timestamp(x).strftime('%d-%m-%Y'))
        with c3: cap = st.number_input("CAPITAL (Rp)", 1_000_000, 1_000_000_000, 20_000_000, 1_000_000)
        with c4: 
            st.write("")
            btn_calc = st.form_submit_button("🚀 CALCULATE PROFIT")
    
    if btn_calc:
        if pd.Timestamp(start_d) >= pd.Timestamp(end_d): st.error("Sell Date must be after Buy Date.")
        else:
            with st.spinner("Calculating PnL..."): 
                df_port, sum_port, msg = simulate_portfolio_range(df, cap, pd.Timestamp(start_d), pd.Timestamp(end_d))
            if msg == "success":
                m1, m2, m3 = st.columns(3)
                m1.metric("INITIAL CAPITAL", f"Rp {sum_port['Initial Capital']:,.0f}")
                m2.metric("FINAL VALUE", f"Rp {sum_port['Final Portfolio Value']:,.0f}")
                m3.metric("NET PROFIT", f"Rp {sum_port['Net Profit']:,.0f}", delta=f"{sum_port['Total ROI']:.2f}%")
                st.dataframe(df_port[['Stock Code', 'Buy Price', 'Sell Price', 'Gain/Loss (Rp)', 'ROI (%)']], use_container_width=True, hide_index=True, column_config={
                    "Buy Price": st.column_config.NumberColumn(format="%d"),
                    "Sell Price": st.column_config.NumberColumn(format="%d"),
                    "Gain/Loss (Rp)": st.column_config.NumberColumn(format="Rp %.0f"),
                    "ROI (%)": st.column_config.NumberColumn(format="%.2f %%")
                })
            else: st.error(msg)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 9: MSCI ---
with tab9:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("#### 🌏 MSCI PROXY SIMULATOR")
    st.info("Target: Float Cap > $1.5B & Liquidity (ATVR) > 15%")
    if 'Listed Shares' not in df.columns: st.error("❌ Data 'Listed Shares' missing.")
    else:
        with st.expander("⚙️ CONFIGURATION", expanded=True):
            c_p1, c_p2 = st.columns(2)
            with c_p1: usd_idr = st.number_input("USD/IDR RATE", value=16500, step=50)
            with c_p2:
                cut_float_usd = st.number_input("MIN FLOAT CAP ($B)", value=1.5, step=0.1)
                cut_atvr = st.number_input("MIN LIQUIDITY (ATVR %)", value=15.0, step=1.0)
        with st.spinner("Calculating MSCI Universe..."):
            df_msci = calculate_msci_projection_v2_optimized(df, df['Last Trading Date'].max(), usd_idr)
        
        def cat_msci(r):
            pass_size = (r['Float Cap ($B)'] >= cut_float_usd)
            pass_liq = (r['ATVR 12M (%)'] >= cut_atvr) and (r['ATVR 3M (%)'] >= cut_atvr)
            if pass_size and pass_liq: return "✅ Potential Standard"
            elif pass_size and not pass_liq: return "⚠️ Risk Deletion (Illiquid)"
            elif (r['Float Cap ($B)'] >= (cut_float_usd * 0.5)) and pass_liq: return "🔹 Small Cap Potential"
            return "🔻 Micro / Others"
        
        if not df_msci.empty:
            df_msci['Status'] = df_msci.apply(cat_msci, axis=1)
            col_t1, col_t2 = st.columns(2)
            show_cols = ['Rank', 'Stock Code', 'Status', 'Float Cap ($B)', 'ATVR 3M (%)', 'ATVR 12M (%)']
            with col_t1: 
                st.markdown("#### ✅ TOP CANDIDATES")
                st.dataframe(df_msci[df_msci['Status']=="✅ Potential Standard"][show_cols], hide_index=True, use_container_width=True, column_config={"Float Cap ($B)": st.column_config.NumberColumn(format="$ %.2f B"), "ATVR 3M (%)": st.column_config.NumberColumn(format="%.1f %%")})
            with col_t2: 
                st.markdown("#### ⚠️ RISK LIST (ILLIQUID)")
                st.dataframe(df_msci[df_msci['Status']=="⚠️ Risk Deletion (Illiquid)"][show_cols], hide_index=True, use_container_width=True, column_config={"Float Cap ($B)": st.column_config.NumberColumn(format="$ %.2f B"), "ATVR 3M (%)": st.column_config.NumberColumn(format="%.1f %%")})
    st.markdown('</div>', unsafe_allow_html=True)
