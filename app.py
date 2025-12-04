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
    page_title="📊 Dashboard Analisis Saham IDX + Portfolio Sim",
    layout="wide",
    page_icon="📈"
)

# --- KONFIGURASI G-DRIVE ---
# Ganti dengan Folder ID dan Nama File yang sesuai
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP" 
FILE_NAME = "Kompilasi_Data_1Tahun.csv"

# Bobot skor (Logic "Raport")
W = dict(
    trend_akum=0.40, trend_ff=0.30, trend_mfv=0.20, trend_mom=0.10,
    mom_price=0.40,  mom_vol=0.25,  mom_akum=0.25,  mom_ff=0.10,
    blend_trend=0.35, blend_mom=0.35, blend_nbsa=0.20, blend_fcontrib=0.05, blend_unusual=0.05
)

# ==============================================================================
# 📦 3) FUNGSI MEMUAT DATA (via SERVICE ACCOUNT)
# ==============================================================================
def get_gdrive_service():
    try:
        creds_json = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service, None
    except KeyError:
        msg = "❌ Gagal otentikasi: 'st.secrets' tidak menemukan key [gcp_service_account]. Pastikan 'secrets.toml' sudah benar."
        return None, msg
    except Exception as e:
        msg = f"❌ Gagal otentikasi Google Drive: {e}."
        return None, msg

@st.cache_data(ttl=3600)
def load_data():
    """Mencari file transaksi, men-download, membersihkan, dan membacanya ke Pandas."""
    service, error_msg = get_gdrive_service()
    if error_msg:
        return pd.DataFrame(), error_msg, "error"

    try:
        query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
        results = service.files().list(
            q=query, fields="files(id, name)", orderBy="modifiedTime desc", pageSize=1
        ).execute()
        items = results.get('files', [])

        if not items:
            msg = f"❌ File '{FILE_NAME}' tidak ditemukan di folder GDrive."
            return pd.DataFrame(), msg, "error"

        file_id = items[0]['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)

        df = pd.read_csv(fh, dtype=object)

        df.columns = df.columns.str.strip()
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')

        # Daftar kolom numerik yang harus dibersihkan
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
                cleaned_col = df[col].astype(str).str.strip()
                cleaned_col = cleaned_col.str.replace(r'[,\sRp\%]', '', regex=True)
                df[col] = pd.to_numeric(cleaned_col, errors='coerce').fillna(0)

        if 'Unusual Volume' in df.columns:
            # Handle variasi string boolean
            df['Unusual Volume'] = df['Unusual Volume'].astype(str).str.strip().str.lower().isin(['spike volume signifikan', 'true', 'True', 'TRUE'])
            df['Unusual Volume'] = df['Unusual Volume'].astype(bool)

        if 'Final Signal' in df.columns:
            df['Final Signal'] = df['Final Signal'].astype(str).str.strip()

        if 'Sector' in df.columns:
             df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
        else:
             df['Sector'] = 'Others'

        df = df.dropna(subset=['Last Trading Date', 'Stock Code'])

        # Hitung NFF (Rp) jika belum ada
        if 'NFF (Rp)' not in df.columns:
             if 'Typical Price' in df.columns:
                 df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Typical Price']
             else:
                 df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Close']

        msg = f"Data Transaksi Harian berhasil dimuat (file ID: {file_id})."
        return df, msg, "success"

    except Exception as e:
        msg = f"❌ Terjadi error saat memuat data: {e}."
        return pd.DataFrame(), msg, "error"

# ==============================================================================
# 🛠️ 4) FUNGSI KALKULASI SKOR & SIMULASI
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

# Fungsi Utama Perhitungan Skor
def calculate_potential_score(df: pd.DataFrame, latest_date: pd.Timestamp):
    """Menjalankan logika scoring 'Raport' pada data cutoff tanggal tertentu."""
    trend_start = latest_date - pd.Timedelta(days=30)
    mom_start = latest_date - pd.Timedelta(days=7)
    
    # Filter data historis berdasarkan tanggal cutoff (latest_date)
    # Penting untuk backtest: Jangan melihat data masa depan!
    df_historic = df[df['Last Trading Date'] <= latest_date]
    
    trend_df = df_historic[df_historic['Last Trading Date'] >= trend_start].copy()
    mom_df = df_historic[df_historic['Last Trading Date'] >= mom_start].copy()
    last_df = df_historic[df_historic['Last Trading Date'] == latest_date].copy()

    if trend_df.empty or mom_df.empty or last_df.empty:
        msg = "Data tidak cukup."
        return pd.DataFrame(), msg, "warning"

    # --- 1. Trend Score (30 Hari) ---
    tr = trend_df.groupby('Stock Code').agg(
        last_price=('Close', 'last'), last_final_signal=('Final Signal', 'last'),
        total_net_ff_rp=('NFF (Rp)', 'sum'), total_money_flow=('Money Flow Value', 'sum'),
        avg_change_pct=('Change %', 'mean'), sector=('Sector', 'last')
    ).reset_index()
    score_akum = tr['last_final_signal'].map({'Strong Akumulasi': 100, 'Akumulasi': 75, 'Netral': 30, 'Distribusi': 10, 'Strong Distribusi': 0}).fillna(30)
    score_ff = pct_rank(tr['total_net_ff_rp'])
    score_mfv = pct_rank(tr['total_money_flow'])
    score_mom = pct_rank(tr['avg_change_pct'])
    tr['Trend Score'] = (score_akum * W['trend_akum'] + score_ff * W['trend_ff'] +
                         score_mfv * W['trend_mfv'] + score_mom * W['trend_mom'])

    # --- 2. Momentum Score (7 Hari) ---
    mo = mom_df.groupby('Stock Code').agg(
        total_change_pct=('Change %', 'sum'), had_unusual_volume=('Unusual Volume', 'any'),
        last_final_signal=('Final Signal', 'last'), total_net_ff_rp=('NFF (Rp)', 'sum')
    ).reset_index()
    s_price = pct_rank(mo['total_change_pct'])
    s_vol = mo['had_unusual_volume'].map({True: 100, False: 20}).fillna(20)
    s_akum = mo['last_final_signal'].map({'Strong Akumulasi': 100, 'Akumulasi': 80, 'Netral': 40, 'Distribusi': 10, 'Strong Distribusi': 0}).fillna(40)
    s_ff7 = pct_rank(mo['total_net_ff_rp'])
    mo['Momentum Score'] = (s_price * W['mom_price'] + s_vol * W['mom_vol'] +
                            s_akum * W['mom_akum'] + s_ff7 * W['mom_ff'])

    # --- 3. NBSA & Foreign Contribution ---
    nbsa = trend_df.groupby('Stock Code').agg(total_net_ff_30d_rp=('NFF (Rp)', 'sum')).reset_index()
    
    if {'Foreign Buy', 'Foreign Sell', 'Value'}.issubset(df.columns):
        tmp = trend_df.copy()
        tmp['Foreign Value proxy'] = tmp['NFF (Rp)']
        contrib = tmp.groupby('Stock Code').agg(
            total_foreign_value_proxy=('Foreign Value proxy', 'sum'),
            total_value_30d=('Value', 'sum')
        ).reset_index()
        contrib['foreign_contrib_pct'] = np.where(contrib['total_value_30d'] > 0,
                                                (contrib['total_foreign_value_proxy'].abs() / contrib['total_value_30d']) * 100,
                                                0)
    else:
        contrib = pd.DataFrame({'Stock Code': [], 'foreign_contrib_pct': []})

    uv = last_df.set_index('Stock Code')['Unusual Volume'].map({True: 1, False: 0})

    # --- 4. Merge & Final Score ---
    rank = tr[['Stock Code', 'Trend Score', 'last_price', 'last_final_signal', 'sector']].merge(
        mo[['Stock Code', 'Momentum Score']], on='Stock Code', how='outer'
    ).merge(
        nbsa, on='Stock Code', how='left'
    ).merge(
        contrib[['Stock Code', 'foreign_contrib_pct']], on='Stock Code', how='left'
    )
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
    
    score_cols = ['Potential Score', 'Trend Score', 'Momentum Score', 'NBSA Score', 'Foreign Contrib Score']
    for c in score_cols:
        if c in top20.columns: top20[c] = pd.to_numeric(top20[c], errors='coerce').round(2)

    cols_order = ['Analysis Date', 'Stock Code', 'Potential Score', 'Trend Score', 'Momentum Score',
                  'total_net_ff_30d_rp', 'foreign_contrib_pct', 'last_price', 'last_final_signal', 'sector']
    for c in cols_order:
        if c not in top20.columns: top20[c] = np.nan
    top20 = top20[cols_order]

    msg = "Skor berhasil dihitung."
    return top20, msg, "success"

@st.cache_data(ttl=3600)
def calculate_nff_top_stocks(df: pd.DataFrame, max_date: pd.Timestamp):
    """Menghitung agregat NFF (Rp) untuk beberapa periode."""
    periods = {'7D': 7, '30D': 30, '90D': 90, '180D': 180}
    results = {}
    latest_data = df[df['Last Trading Date'] == max_date].set_index('Stock Code')
    latest_prices = latest_data.get('Close', pd.Series(dtype='float64'))
    latest_sectors = latest_data.get('Sector', pd.Series(dtype='object'))

    for name, days in periods.items():
        start_date = max_date - pd.Timedelta(days=days)
        df_period = df[df['Last Trading Date'] >= start_date].copy()
        nff_agg = df_period.groupby('Stock Code')['NFF (Rp)'].sum()
        df_agg = pd.DataFrame(nff_agg)
        df_agg.columns = ['Total Net FF (Rp)']
        df_agg = df_agg.join(latest_prices).join(latest_sectors)
        df_agg.rename(columns={'Close': 'Harga Terakhir'}, inplace=True)
        df_agg = df_agg.sort_values(by='Total Net FF (Rp)', ascending=False)
        results[name] = df_agg.reset_index()

    return results.get('7D', pd.DataFrame()), results.get('30D', pd.DataFrame()), \
           results.get('90D', pd.DataFrame()), results.get('180D', pd.DataFrame())

@st.cache_data(ttl=3600)
def calculate_mfv_top_stocks(df: pd.DataFrame, max_date: pd.Timestamp):
    """Menghitung agregat MFV (Rp) untuk beberapa periode."""
    periods = {'7D': 7, '30D': 30, '90D': 90, '180D': 180}
    results = {}
    latest_data = df[df['Last Trading Date'] == max_date].set_index('Stock Code')
    latest_prices = latest_data.get('Close', pd.Series(dtype='float64'))
    latest_sectors = latest_data.get('Sector', pd.Series(dtype='object'))

    for name, days in periods.items():
        start_date = max_date - pd.Timedelta(days=days)
        df_period = df[df['Last Trading Date'] >= start_date].copy()
        mfv_agg = df_period.groupby('Stock Code')['Money Flow Value'].sum()
        df_agg = pd.DataFrame(mfv_agg)
        df_agg.columns = ['Total Money Flow (Rp)']
        df_agg = df_agg.join(latest_prices).join(latest_sectors)
        df_agg.rename(columns={'Close': 'Harga Terakhir'}, inplace=True)
        df_agg = df_agg.sort_values(by='Total Money Flow (Rp)', ascending=False)
        results[name] = df_agg.reset_index()

    return results.get('7D', pd.DataFrame()), results.get('30D', pd.DataFrame()), \
           results.get('90D', pd.DataFrame()), results.get('180D', pd.DataFrame())

# --- FUNGSI BACKTESTING (AUDIT LOGIC) ---
def run_backtest_analysis(df, days_back=90):
    all_dates = sorted(df['Last Trading Date'].unique())
    if len(all_dates) < days_back: days_back = len(all_dates) - 30 
    start_idx = max(30, len(all_dates) - days_back)
    simulation_dates = all_dates[start_idx:]
    
    latest_date = all_dates[-1]
    latest_prices_series = df[df['Last Trading Date'] == latest_date].set_index('Stock Code')['Close']
    
    backtest_log = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(simulation_dates)
    
    for i, sim_date in enumerate(simulation_dates):
        pct = (i + 1) / total_steps
        progress_bar.progress(pct)
        sim_date_ts = pd.Timestamp(sim_date)
        status_text.text(f"⏳ Mengaudit data tanggal: {sim_date_ts.strftime('%d-%m-%Y')}...")
        
        try:
            top20_daily, _, status = calculate_potential_score(df, sim_date_ts)
            if status == "success" and not top20_daily.empty:
                for idx, row in top20_daily.iterrows():
                    code = row['Stock Code']
                    entry_price = row['last_price']
                    score = row['Potential Score']
                    curr_price = latest_prices_series.get(code, np.nan)
                    ret_pct = 0.0
                    if pd.notna(curr_price) and entry_price > 0:
                        ret_pct = ((curr_price - entry_price) / entry_price) * 100
                    backtest_log.append({
                        'Signal Date': sim_date_ts, 'Stock Code': code,
                        'Entry Price': entry_price, 'Current Price': curr_price,
                        'Return to Date (%)': ret_pct, 'Score at Signal': score
                    })
        except Exception: continue
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(backtest_log)

# --- FUNGSI SIMULASI PORTFOLIO (BARU) ---
def simulate_portfolio(df, capital, sim_date_ts):
    """Simulasi beli Top 20 pada tanggal tertentu dengan bobot rata."""
    top20, _, status = calculate_potential_score(df, sim_date_ts)
    
    if status != "success" or top20.empty:
        return pd.DataFrame(), None, "Gagal mendapatkan Top 20 pada tanggal tersebut."
    
    # Ambil harga terakhir di database
    latest_date = df['Last Trading Date'].max()
    latest_prices = df[df['Last Trading Date'] == latest_date].set_index('Stock Code')['Close']
    
    allocation_per_stock = capital / len(top20) # Equal weight
    
    portfolio_results = []
    
    for idx, row in top20.iterrows():
        code = row['Stock Code']
        buy_price = row['last_price']
        
        # Current price lookup
        curr_price = latest_prices.get(code, np.nan)
        
        if pd.isna(curr_price) or buy_price <= 0:
            roi_pct = 0
            final_val = allocation_per_stock # Asumsi uang kembali kalau data error
            gain_rp = 0
        else:
            roi_pct = ((curr_price - buy_price) / buy_price)
            final_val = allocation_per_stock * (1 + roi_pct)
            gain_rp = final_val - allocation_per_stock
            
        portfolio_results.append({
            'Stock Code': code,
            'Buy Price': buy_price,
            'Current Price': curr_price,
            'Allocation': allocation_per_stock,
            'Final Value': final_val,
            'Gain/Loss (Rp)': gain_rp,
            'ROI (%)': roi_pct * 100,
            'Sector': row['sector']
        })
        
    df_port = pd.DataFrame(portfolio_results)
    
    summary = {
        'Start Date': sim_date_ts,
        'End Date': latest_date,
        'Initial Capital': capital,
        'Final Portfolio Value': df_port['Final Value'].sum(),
        'Net Profit': df_port['Gain/Loss (Rp)'].sum(),
        'Total ROI': (df_port['Gain/Loss (Rp)'].sum() / capital) * 100
    }
    
    return df_port, summary, "success"

# ==============================================================================
# 💎 5) LAYOUT DASHBOARD
# ==============================================================================
st.title("📈 Dashboard Analisis Saham IDX")
st.caption("Analisis Teknikal + Bandarmology + Backtest + Portfolio Simulation")

df, status_msg, status_level = load_data()

if status_level == "success":
    st.toast(status_msg, icon="✅")
elif status_level == "error":
    st.error(status_msg)
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("🎛️ Filter & Navigasi")

if st.sidebar.button("🔄 Refresh Data GDrive"):
    st.cache_data.clear()
    st.rerun()

max_date = df['Last Trading Date'].max().date()
selected_date = st.sidebar.date_input(
    "Pilih Tanggal Analisis (Dashboard)",
    max_date,
    min_value=df['Last Trading Date'].min().date(),
    max_value=max_date,
    format="DD-MM-YYYY"
)

df_day = df[df['Last Trading Date'].dt.date == selected_date].copy()

st.sidebar.markdown("---")
st.sidebar.header("Filter Tampilan (Tab 3)")
selected_stocks_filter = st.sidebar.multiselect("Pilih Saham", options=sorted(df_day["Stock Code"].dropna().unique()))
selected_sectors_filter = st.sidebar.multiselect("Pilih Sektor", options=sorted(df_day.get("Sector", pd.Series(dtype='object')).dropna().unique()))

df_filtered = df_day.copy()
if selected_stocks_filter:
    df_filtered = df_filtered[df_filtered["Stock Code"].isin(selected_stocks_filter)]
if selected_sectors_filter and 'Sector' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Sector"].isin(selected_sectors_filter)]


# --- TABS VISUALISASI ---
st.caption(f"Menampilkan data per: **{selected_date.strftime('%d %B %Y')}**")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 **Dashboard Harian**",
    "📈 **Analisis Individual**",
    "📋 **Data Filter**",
    "🏆 **Top 20 Potensial**",
    "🌊 **Analisis NFF (Rp)**",
    "💰 **Analisis Money Flow**",
    "🧪 **Backtest Logic**",
    "💼 **Simulasi Portfolio**" # <-- TAB BARU
])

# --- TAB 1: DASHBOARD HARIAN ---
with tab1:
    st.subheader("Ringkasan Pasar")
    if df_day.empty:
        st.warning(f"Tidak ada data transaksi untuk tanggal {selected_date}.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Saham Aktif", f"{len(df_day):,.0f}")
        c2.metric("Unusual Volume", f"{df_day['Unusual Volume'].sum():,.0f}")
        c3.metric("Total Nilai Transaksi", f"Rp {df_day['Value'].sum():,.0f}")
        st.markdown("---")
        col_g, col_l, col_v = st.columns(3)
        def show_top(dframe, sort_col, asc, title, val_col="Change %"):
            st.markdown(f"**{title}**")
            top = dframe.sort_values(sort_col, ascending=asc).head(10)[['Stock Code', 'Close', val_col]]
            top['Close'] = top['Close'].apply(lambda x: f"{x:,.0f}")
            if val_col == "Value": top[val_col] = top[val_col].apply(lambda x: f"{x/1e9:,.2f} M")
            st.dataframe(top, hide_index=True, use_container_width=True)
        with col_g: show_top(df_day, "Change %", False, "Top Gainers (%)")
        with col_l: show_top(df_day, "Change %", True, "Top Losers (%)")
        with col_v: show_top(df_day, "Value", False, "Top Value (Miliar)", "Value")

# --- TAB 2: ANALISIS INDIVIDUAL ---
with tab2:
    st.subheader("Deep Dive Analysis")
    all_stocks = sorted(df["Stock Code"].unique())
    stock = st.selectbox("Pilih Saham", all_stocks, index=all_stocks.index("BBRI") if "BBRI" in all_stocks else 0)
    if stock:
        df_stock = df[df['Stock Code'] == stock].sort_values('Last Trading Date')
        if not df_stock.empty:
            last_row = df_stock.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Harga", f"Rp {last_row['Close']:,.0f}")
            c2.metric("NFF (Rp)", f"Rp {last_row['NFF (Rp)']:,.0f}")
            c3.metric("MFV (Rp)", f"Rp {last_row['Money Flow Value']:,.0f}")
            c4.metric("MF Ratio (20D)", f"{last_row.get('Money Flow Ratio (20D)', 0):.3f}")
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                                row_heights=[0.35, 0.2, 0.2, 0.25], specs=[[{"secondary_y": True}], [{}], [{}], [{}]])
            colors_nff = np.where(df_stock['NFF (Rp)'] >= 0, 'green', 'red')
            fig.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['NFF (Rp)'], name='NFF (Rp)', marker_color=colors_nff), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['Close'], name='Close', line=dict(color='blue')), row=1, col=1, secondary_y=True)
            colors_mfv = np.where(df_stock['Money Flow Value'] >= 0, 'teal', 'salmon')
            fig.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['Money Flow Value'], name='MFV (Rp)', marker_color=colors_mfv), row=2, col=1)
            fig.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['Volume'], name='Volume', marker_color='gray'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['MA20_vol'], name='MA20 Vol', line=dict(color='orange', dash='dot')), row=3, col=1)
            if 'Money Flow Ratio (20D)' in df_stock.columns:
                fig.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['Money Flow Ratio (20D)'], name='MF Ratio (20D)', line=dict(color='purple')), row=4, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="black", row=4, col=1)
            fig.update_layout(height=800, title=f"Analisis Teknikal & Flow: {stock}", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: DATA FILTER ---
with tab3:
    st.subheader("Data Mentah Terfilter")
    st.dataframe(df_filtered, use_container_width=True)

# --- TAB 4: TOP 20 POTENSIAL ---
with tab4:
    st.subheader("🏆 Top 20 Saham Potensial (Scoring Logic)")
    @st.cache_data(ttl=3600)
    def get_cached_top20(dframe, tgl): return calculate_potential_score(dframe, tgl)
    df_top20, msg, status = get_cached_top20(df, pd.Timestamp(selected_date))
    if status == "success":
        st.dataframe(df_top20, use_container_width=True, hide_index=True, column_config={
            "Potential Score": st.column_config.ProgressColumn("Skor Akhir", format="%.2f", min_value=0, max_value=100),
            "total_net_ff_30d_rp": st.column_config.NumberColumn("NFF 30D (Rp)", format="Rp %.0f"),
            "last_price": st.column_config.NumberColumn("Harga", format="Rp %.0f")})
    else: st.warning(msg)

# --- TAB 5 & 6 (FLOW) ---
with tab5:
    st.subheader("Analisis Net Foreign Flow (Rp)")
    nff7, nff30, nff90, nff180 = calculate_nff_top_stocks(df, pd.Timestamp(selected_date))
    c1, c2 = st.columns(2)
    c1.dataframe(nff7.head(10), hide_index=True, use_container_width=True); c1.dataframe(nff90.head(10), hide_index=True, use_container_width=True)
    c2.dataframe(nff30.head(10), hide_index=True, use_container_width=True); c2.dataframe(nff180.head(10), hide_index=True, use_container_width=True)

with tab6:
    st.subheader("Analisis Money Flow Value & Ratio")
    mfv7, mfv30, mfv90, mfv180 = calculate_mfv_top_stocks(df, pd.Timestamp(selected_date))
    c1, c2 = st.columns(2)
    c1.dataframe(mfv7.head(10), hide_index=True, use_container_width=True)
    c2.dataframe(mfv30.head(10), hide_index=True, use_container_width=True)
    st.markdown("---")
    if 'Money Flow Ratio (20D)' in df_day.columns:
        cr1, cr2 = st.columns(2)
        cr1.markdown("Top Rasio Positif"); cr1.dataframe(df_day.sort_values('Money Flow Ratio (20D)', ascending=False).head(10)[['Stock Code','Close','Money Flow Ratio (20D)']], hide_index=True, use_container_width=True)
        cr2.markdown("Top Rasio Negatif"); cr2.dataframe(df_day.sort_values('Money Flow Ratio (20D)', ascending=True).head(10)[['Stock Code','Close','Money Flow Ratio (20D)']], hide_index=True, use_container_width=True)

# --- TAB 7: BACKTEST LOGIC (FULL) ---
with tab7:
    st.subheader("🧪 Backtest: Audit Performa Algoritma")
    st.markdown("Audit sinyal mundur ke masa lalu untuk melihat Win Rate asli.")
    col_bt1, col_bt2 = st.columns([1, 3])
    with col_bt1:
        days_to_test = st.number_input("Hari Backtest", min_value=7, max_value=90, value=30, step=7)
        run_btn = st.button("🚀 Jalankan Audit")
    if run_btn:
        with st.spinner("Processing..."): df_backtest = run_backtest_analysis(df, days_back=days_to_test)
        if not df_backtest.empty:
            st.success("Selesai!")
            win_rate = (len(df_backtest[df_backtest['Return to Date (%)'] > 0]) / len(df_backtest)) * 100
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Sinyal", f"{len(df_backtest)}x"); k2.metric("Win Rate", f"{win_rate:.1f}%"); k3.metric("Avg Return", f"{df_backtest['Return to Date (%)'].mean():.2f}%")
            
            st.markdown("---")
            st.subheader("Analisis Frekuensi")
            freq = df_backtest.groupby('Stock Code').agg(Freq=('Signal Date','count'), Avg_Ret=('Return to Date (%)','mean')).reset_index()
            freq['Kategori'] = pd.cut(freq['Freq'], bins=[0,2,5,100], labels=['Jarang','Sedang','Sering'])
            sum_freq = freq.groupby('Kategori', observed=False).agg(Count=('Stock Code','count'), Avg_Ret=('Avg_Ret','mean')).reset_index()
            
            c1, c2 = st.columns(2)
            c1.dataframe(sum_freq, use_container_width=True)
            c2.plotly_chart(px.bar(sum_freq, x='Kategori', y='Avg_Ret', title="Return by Frequency", color='Avg_Ret'), use_container_width=True)
            
            st.subheader("Detail Sinyal")
            st.dataframe(df_backtest.sort_values('Signal Date', ascending=False), use_container_width=True)
        else: st.warning("No data.")

# --- TAB 8: SIMULASI PORTFOLIO (BARU) ---
with tab8:
    st.subheader("💼 Simulasi Portfolio: Cuan Berapa?")
    st.markdown("Pilih tanggal di masa lalu. Asumsikan kita beli Top 20 hari itu, hold sampai sekarang.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    # Cari tanggal valid untuk simulasi (minimal 7 hari ke belakang dari data terakhir)
    valid_dates = sorted(df['Last Trading Date'].unique())
    latest_avail = valid_dates[-1]
    default_sim_idx = max(0, len(valid_dates) - 30) # Default 30 hari lalu
    
    with col_p1:
        sim_date_input = st.selectbox(
            "Pilih Tanggal Mulai Beli",
            options=valid_dates[:-1], # Jangan pilih hari terakhir
            index=default_sim_idx,
            format_func=lambda x: pd.Timestamp(x).strftime('%d-%m-%Y')
        )
    with col_p2:
        capital_input = st.number_input("Modal Awal (Rp)", min_value=1_000_000, value=20_000_000, step=1_000_000)
    with col_p3:
        st.write("") # Spacer
        st.write("") # Spacer
        calc_port_btn = st.button("💰 Hitung Profit/Loss")
        
    if calc_port_btn:
        df_port, summary_port, msg_port = simulate_portfolio(df, capital_input, pd.Timestamp(sim_date_input))
        
        if msg_port == "success":
            # 1. TAMPILAN KARTU RINGKASAN
            st.success(f"Simulasi dari tanggal {pd.Timestamp(sim_date_input).strftime('%d-%m-%Y')} s/d {pd.Timestamp(summary_port['End Date']).strftime('%d-%m-%Y')}")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Modal Awal", f"Rp {summary_port['Initial Capital']:,.0f}")
            
            # Warna untuk profit/loss
            delta_color = "normal"
            if summary_port['Net Profit'] > 0: delta_color = "normal" # Streamlit auto green for positive delta
            
            m2.metric("Nilai Aset Sekarang", f"Rp {summary_port['Final Portfolio Value']:,.0f}")
            m3.metric("Net Profit/Loss", f"Rp {summary_port['Net Profit']:,.0f}", delta=f"{summary_port['Total ROI']:.2f}%")
            
            # Menghitung saham untung vs rugi
            win_stocks = len(df_port[df_port['Gain/Loss (Rp)'] > 0])
            m4.metric("Saham Cuan / Rugi", f"{win_stocks} / {len(df_port) - win_stocks}")
            
            st.markdown("---")
            
            # 2. CHART KOMPOSISI PROFIT
            c_chart1, c_chart2 = st.columns([2, 1])
            
            with c_chart1:
                st.subheader("Kontribusi Profit per Saham")
                # Bar chart sorted by Gain amount
                df_port_sorted = df_port.sort_values('Gain/Loss (Rp)', ascending=False)
                fig_port = px.bar(
                    df_port_sorted, 
                    x='Stock Code', 
                    y='Gain/Loss (Rp)',
                    color='Gain/Loss (Rp)',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    text_auto='.2s',
                    title="Siapa Pahlawan & Siapa Beban?"
                )
                fig_port.update_layout(xaxis_title="Kode Saham", yaxis_title="Profit/Loss (Rp)")
                st.plotly_chart(fig_port, use_container_width=True)
                
            with c_chart2:
                st.subheader("Distribusi Sektor")
                fig_pie = px.pie(df_port, names='Sector', values='Allocation', title="Alokasi Sektor")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # 3. TABEL DETAIL
            st.subheader("Rincian Transaksi")
            st.dataframe(
                df_port_sorted[['Stock Code', 'Sector', 'Buy Price', 'Current Price', 'ROI (%)', 'Gain/Loss (Rp)', 'Final Value']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Buy Price": st.column_config.NumberColumn("Harga Beli", format="Rp %.0f"),
                    "Current Price": st.column_config.NumberColumn("Harga Jual", format="Rp %.0f"),
                    "Gain/Loss (Rp)": st.column_config.NumberColumn("Cuan (Rp)", format="Rp %.0f"),
                    "Final Value": st.column_config.NumberColumn("Nilai Aset", format="Rp %.0f"),
                    "ROI (%)": st.column_config.NumberColumn("ROI %", format="%.2f %%")
                }
            )
            
        else:
            st.error(msg_port)
