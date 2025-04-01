# sp_vix_app_plotly_single_fixedaxis_square_30y_slider_main_v16_left_title_fixed_axes.py
# FINAL VERSION BASED ON USER REQUESTS + 3-COLUMN LAYOUT + LEFT TITLE + FIXED AXES (FULL HISTORY)
# Features:
# - Select two tickers (expanded list including sectors, Russell 2000, and Ibovespa)
# - Calculate rolling % change
# - Display scatter plot with fixed axes (based on full history) and square aspect ratio # <<< CONFIRMADO/REFORÇADO >>>
# - Left-aligned plot title # <<< MODIFICADO >>>
# - Quadrant percentage METRICS in side columns (based on displayed data)
# - Main plot in center column (dynamic width)
# - Two view modes (selectable in sidebar):
#   1. Date Range: Slider selects start/end date, colored by year, defaults to FULL range.
#   2. Fixed Window: Slider selects end date, shows last N days, single color.
# - Optional Confidence Ellipse (calculated on displayed data, selectable level in sidebar).
# - Optional data table (below main layout, controlled by sidebar checkbox).
# - Layout: Controls in sidebar, slider + 3-col(metrics-plot-metrics) + table in main area.

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from collections import OrderedDict
import numpy as np
from scipy.stats import chi2 # Needed for confidence level scaling

# --- Configuration ---
APP_TITLE = "Analisador de Variação Móvel com Elipse de Confiança e Quadrantes"

# Mapeamento Ticker -> Nome (Inglês/Referência) - Updated List with IBOVESPA
AVAILABLE_TICKERS = OrderedDict([
    # Commodities / Moedas
    ("DX-Y.NYB", "US Dollar Index (DXY)"),
    ("CL=F", "WTI Crude Oil Futures"),
    ("TIO=F", "Iron Ore Futures (62% FE)"),
    ("GC=F", "Gold Futures"),
    # Índices Principais
    ("^GSPC", "S&P 500"),
    ("^IXIC", "NASDAQ Composite"),
    ("^DJI", "Dow Jones Industrial Average"),
    ("^RUT", "Russell 2000"),
    ("^VIX", "VIX"),
    ("^GDAXI", "DAX (Germany)"),
    ("^BVSP", "Bovespa (Brazil)"),
    # Setores SPDR ETFs
    ("XLP", "Consumer Staples Sector SPDR"),
    ("XLF", "Financial Sector SPDR"),
    ("XLU", "Utilities Sector SPDR"),
    ("XLE", "Energy Sector SPDR"),
    ("XLB", "Materials Sector SPDR"),
    ("XLRE", "Real Estate Sector SPDR"),
    ("XLV", "Health Care Sector SPDR"),
    ("XLC", "Communication Services Sector SPDR"),
    ("XLI", "Industrial Sector SPDR"),
    ("XLY", "Consumer Discretionary Sector SPDR"),
    ("XLK", "Technology Sector SPDR"),
])

# Mapeamento Ticker -> Nome (Português para UI) - Updated List with IBOVESPA
AVAILABLE_TICKERS_PT = OrderedDict([
    # Commodities / Moedas
    ("DX-Y.NYB", "Índice Dólar Americano (DXY)"),
    ("CL=F", "Petróleo Cru WTI (Futuros)"),
    ("TIO=F", "Minério de Ferro (Futuros)"),
    ("GC=F", "Ouro (Futuros)"),
    # Índices Principais
    ("^GSPC", "S&P 500"),
    ("^IXIC", "NASDAQ Composite"),
    ("^DJI", "Dow Jones Industrial Average"),
    ("^RUT", "Russell 2000"),
    ("^VIX", "VIX"),
    ("^GDAXI", "DAX (Alemanha)"),
    ("^BVSP", "Ibovespa (Brasil)"),
    # Setores SPDR ETFs
    ("XLP", "Setor Consumo Básico (XLP)"),
    ("XLF", "Setor Financeiro (XLF)"),
    ("XLU", "Setor Utilidade Pública (XLU)"),
    ("XLE", "Setor Energia (XLE)"),
    ("XLB", "Setor Materiais (XLB)"),
    ("XLRE", "Setor Imobiliário (XLRE)"),
    ("XLV", "Setor Saúde (XLV)"),
    ("XLC", "Setor Serv. Comunicação (XLC)"),
    ("XLI", "Setor Industrial (XLI)"),
    ("XLY", "Setor Consumo Discricionário (XLY)"),
    ("XLK", "Setor Tecnologia (XLK)"),
])

INITIAL_YEARS_FETCH = 30
PLOT_FIXED_HEIGHT = 700 # Mantém a altura fixa para o gráfico
AXIS_PADDING_FACTOR = 0.05
FIXED_WINDOW_TRADING_DAYS = 252
CONFIDENCE_LEVELS = {
    "68% (~1σ)": 0.68,
    "95% (~2σ)": 0.95,
    "99.7% (~3σ)": 0.997
}
DEFAULT_CONFIDENCE_KEY = "95% (~2σ)"
DEFAULT_ELLIPSE_CONF_COLOR = "#A9A9A9" # DarkGray

# --- Caching Functions ---
@st.cache_data(ttl=3600)
def load_data(tickers_tuple, start_date, end_date):
    """Downloads historical stock data for a tuple of tickers."""
    tickers_list = list(tickers_tuple)
    valid_tickers = [t for t in tickers_list if t in AVAILABLE_TICKERS]
    if not valid_tickers:
        st.error("Nenhum ticker válido selecionado.")
        return pd.DataFrame()
    if len(valid_tickers) < len(tickers_list):
        missing = [t for t in tickers_list if t not in valid_tickers]
        st.warning(f"Tickers inválidos ignorados: {', '.join(missing)}")

    print(f"Buscando dados para: {', '.join(valid_tickers)}")
    try:
        data = yf.download(valid_tickers, start=start_date, end=end_date, progress=False)
        if data.empty: return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data.get('Close', pd.DataFrame())
            close_prices = close_prices[[col for col in valid_tickers if col in close_prices.columns]]
            close_prices = close_prices.dropna(axis=1, how='all')
        else:
            if 'Close' in data.columns:
                close_prices = data[['Close']]
                if len(valid_tickers) == 1: close_prices.columns = valid_tickers
            elif len(valid_tickers) == 1 and hasattr(data, 'name') and data.name == 'Close':
                 close_prices = data.to_frame(name=valid_tickers[0])
            else:
                 st.sidebar.warning(f"Não foi possível extrair 'Close' dos dados para {', '.join(valid_tickers)}.")
                 close_prices = pd.DataFrame()

        if close_prices.empty: return pd.DataFrame()

        if not isinstance(close_prices.index, pd.DatetimeIndex):
            close_prices.index = pd.to_datetime(close_prices.index)

        close_prices = close_prices.reindex(columns=tickers_list)
        initial_rows = len(close_prices);
        close_prices = close_prices.ffill().bfill()
        close_prices = close_prices.dropna(how='all', subset=tickers_list)
        rows_after_cleaning = len(close_prices)

        if initial_rows > rows_after_cleaning and rows_after_cleaning > 0:
            st.sidebar.info(f"Limpeza: {initial_rows - rows_after_cleaning} linhas com NaN preenchidas/removidas.")
        elif rows_after_cleaning == 0 and initial_rows > 0:
            st.sidebar.warning("Limpeza resultou em DataFrame vazio.")

        if close_prices.empty: return pd.DataFrame()

        missing_cols = [t for t in tickers_list if t not in close_prices.columns or close_prices[t].isnull().all()]
        if missing_cols:
             st.sidebar.warning(f"Dados não encontrados ou inutilizáveis para: {', '.join(missing_cols)}")
             cols_to_keep = [t for t in tickers_list if t in close_prices.columns and not close_prices[t].isnull().all()]
             if not cols_to_keep: return pd.DataFrame()
             close_prices = close_prices[cols_to_keep]

        return close_prices
    except Exception as e:
        st.error(f"Erro Crítico ao buscar dados para {', '.join(valid_tickers)}: {e}")
        return pd.DataFrame()


@st.cache_data
def calculate_rolling_change(data, window):
    """Calculates rolling percentage change and adds 'Ano' column."""
    if data.empty or window <= 1:
        return pd.DataFrame()
    try:
        numeric_data = data.apply(pd.to_numeric, errors='coerce')
        numeric_data = numeric_data.dropna(subset=data.columns)
        if numeric_data.empty:
            st.sidebar.warning("Não há dados numéricos válidos para cálculo da variação móvel.")
            return pd.DataFrame()

        change_pct = numeric_data.pct_change(periods=window) * 100
        if not change_pct.empty:
             if isinstance(change_pct.index, pd.DatetimeIndex):
                 change_pct['Ano'] = change_pct.index.year
             else:
                 try:
                     change_pct.index = pd.to_datetime(change_pct.index)
                     change_pct['Ano'] = change_pct.index.year
                 except Exception:
                     st.sidebar.error("Não foi possível converter índice para Datetime para adicionar 'Ano'.")
                     pass

        cols_to_check_na = [col for col in change_pct.columns if col != 'Ano']
        return change_pct.dropna(how='all', subset=cols_to_check_na)
    except Exception as e:
        st.error(f"Erro ao calcular variação móvel: {e}")
        return pd.DataFrame()


# --- Function to Calculate Confidence Ellipse Parameters ---
def get_confidence_ellipse(x_data, y_data, confidence=0.95):
    """Calculates parameters for a confidence ellipse based on data."""
    if len(x_data) < 2 or len(y_data) < 2: return None
    try:
        x_data_clean = pd.to_numeric(x_data, errors='coerce').dropna()
        y_data_clean = pd.to_numeric(y_data, errors='coerce').dropna()
        common_index = x_data_clean.index.intersection(y_data_clean.index)
        if len(common_index) < 2: return None
        x_data_aligned = x_data_clean.loc[common_index]
        y_data_aligned = y_data_clean.loc[common_index]

        cov_matrix = np.cov(x_data_aligned, y_data_aligned)
        if np.isclose(np.linalg.det(cov_matrix), 0) or np.isnan(cov_matrix).any(): return None

        center_x, center_y = np.mean(x_data_aligned), np.mean(y_data_aligned)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        if not (0 < confidence < 1):
            st.sidebar.error(f"Nível de confiança inválido: {confidence}")
            return None
        try: scale_factor = np.sqrt(chi2.ppf(confidence, df=2))
        except ValueError as chi_err:
            st.sidebar.error(f"Erro no cálculo Chi-quadrado: {chi_err}")
            return None

        eigenvalues = np.maximum(eigenvalues, 1e-12)
        width = 2 * scale_factor * np.sqrt(eigenvalues[0])
        height = 2 * scale_factor * np.sqrt(eigenvalues[1])
        angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        angle_degrees = np.degrees(angle_rad)

        if any(np.isnan([center_x, center_y, width, height, angle_degrees])): return None
        return center_x, center_y, width, height, angle_degrees
    except np.linalg.LinAlgError: return None
    except Exception: return None

# --- Streamlit App Layout ---
st.set_page_config(page_title=APP_TITLE, layout="wide")

# --- Sidebar Controls ---
st.sidebar.header("Configuração")
ticker_list_pt = list(AVAILABLE_TICKERS_PT.values())

default_y_name = "Índice Dólar Americano (DXY)"
default_x_name = "S&P 500"
try: default_y_index = ticker_list_pt.index(default_y_name)
except ValueError: default_y_index = 0
try: default_x_index = ticker_list_pt.index(default_x_name)
except ValueError: default_x_index = 1 if len(ticker_list_pt) > 1 else 0
if default_x_index == default_y_index and len(ticker_list_pt) > 1:
    default_x_index = (default_y_index + 1) % len(ticker_list_pt)

y_ticker_name_pt = st.sidebar.selectbox("Ticker Eixo Y", options=ticker_list_pt, index=default_y_index, help="Selecione o índice/ativo para o eixo Y.")
y_ticker = [k for k, v in AVAILABLE_TICKERS_PT.items() if v == y_ticker_name_pt][0]
x_ticker_name_pt = st.sidebar.selectbox("Ticker Eixo X", options=ticker_list_pt, index=default_x_index, help="Selecione o índice/ativo para o eixo X.")
x_ticker = [k for k, v in AVAILABLE_TICKERS_PT.items() if v == x_ticker_name_pt][0]

if x_ticker == y_ticker:
    st.sidebar.error("Por favor, selecione tickers diferentes para os eixos X e Y.")
    st.stop()

rolling_window_days = st.sidebar.number_input("Janela Móvel (Dias Úteis)", min_value=2, max_value=252 * 5, value=50, step=1, help="Número de dias úteis para o cálculo da variação percentual móvel.")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Modo de Visualização")
view_mode = st.sidebar.radio(
    "Selecione o Modo:",
    ('Intervalo de Datas (cor por ano)', f'Janela Fixa ({FIXED_WINDOW_TRADING_DAYS} dias úteis, cor única)'),
    key="view_mode_radio", index=0,
    help="Escolha como filtrar os dados: por intervalo ou janela fixa."
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Elipse de Confiança")
show_conf_ellipse = st.sidebar.checkbox("Mostrar Elipse de Confiança", value=True, key="show_conf_ellipse_cb")
confidence_key = st.sidebar.selectbox(
    "Nível de Confiança",
    options=list(CONFIDENCE_LEVELS.keys()),
    index=list(CONFIDENCE_LEVELS.keys()).index(DEFAULT_CONFIDENCE_KEY),
    key="conf_level_sb",
    help="Define o nível de confiança para a elipse.",
    disabled=not show_conf_ellipse
)
selected_confidence_level = CONFIDENCE_LEVELS[confidence_key]

st.sidebar.markdown("---")
show_table = st.sidebar.checkbox("Mostrar tabela de dados filtrados", value=False)

# --- Data Loading & Full Calculation ---
end_date_max = datetime.date.today()
start_date_fetch = end_date_max - datetime.timedelta(days=INITIAL_YEARS_FETCH * 365.25 + 90)
tickers_to_fetch = tuple(sorted(list(set([x_ticker, y_ticker]))))

close_prices_full = pd.DataFrame()
rolling_changes_full = pd.DataFrame()
axis_limits = None # <<< Será calculado aqui, baseado no histórico COMPLETO >>>
data_load_successful = False
min_available_date_calc = None
max_available_date_calc = None

data_load_placeholder = st.empty()
data_load_placeholder.info(f"Carregando dados históricos ({INITIAL_YEARS_FETCH} anos) para {x_ticker_name_pt} e {y_ticker_name_pt}...")

close_prices_full = load_data(tickers_to_fetch, start_date_fetch, end_date_max)

if not close_prices_full.empty:
    missing_in_loaded = [t for t in tickers_to_fetch if t not in close_prices_full.columns or close_prices_full[t].isnull().all()]
    if not missing_in_loaded:
        data_load_successful = True
        data_load_placeholder.success(f"Dados carregados. Calculando variação móvel ({rolling_window_days} dias)...")

        cols_for_calc = [t for t in [x_ticker, y_ticker] if t in close_prices_full.columns]
        if len(cols_for_calc) == 2:
            rolling_changes_full = calculate_rolling_change(close_prices_full[cols_for_calc], rolling_window_days)

            if not rolling_changes_full.empty and x_ticker in rolling_changes_full.columns and y_ticker in rolling_changes_full.columns:
                if not isinstance(rolling_changes_full.index, pd.DatetimeIndex):
                    try: rolling_changes_full.index = pd.to_datetime(rolling_changes_full.index)
                    except Exception as e:
                        st.error(f"Falha ao converter índice de R.Changes para Datetime: {e}")
                        rolling_changes_full = pd.DataFrame()

                if not rolling_changes_full.empty:
                    # <<< CALCULAR LIMITES DOS EIXOS A PARTIR DAQUI (USA rolling_changes_full ANTES DE FILTRAR) >>>
                    valid_data_for_axes = rolling_changes_full[[x_ticker, y_ticker]].dropna()
                    if not valid_data_for_axes.empty:
                        min_available_date_calc = valid_data_for_axes.index.min().date()
                        max_available_date_calc = valid_data_for_axes.index.max().date()

                        # Adiciona a coluna 'Ano' de volta se existir originalmente
                        if 'Ano' in rolling_changes_full.columns:
                            rolling_changes_full = valid_data_for_axes.join(rolling_changes_full['Ano'], how='left')
                        else:
                            rolling_changes_full = valid_data_for_axes # Mantem apenas os dados válidos

                        can_color_by_year = 'Ano' in rolling_changes_full.columns and not rolling_changes_full['Ano'].isnull().all()

                        # --- Cálculo dos Limites Fixos dos Eixos ---
                        try:
                            min_x = valid_data_for_axes[x_ticker].min(); max_x = valid_data_for_axes[x_ticker].max()
                            min_y = valid_data_for_axes[y_ticker].min(); max_y = valid_data_for_axes[y_ticker].max()
                            x_range = max_x - min_x; y_range = max_y - min_y
                            x_pad = x_range * AXIS_PADDING_FACTOR if x_range > 1e-9 else 1
                            y_pad = y_range * AXIS_PADDING_FACTOR if y_range > 1e-9 else 1
                            # Armazena os limites calculados globalmente
                            axis_limits = {'x': [min_x - x_pad, max_x + x_pad], 'y': [min_y - y_pad, max_y + y_pad]}
                            #axis_limits = {'x': [-10, 10], 'y': [-5, 5]}
                            data_load_placeholder.success(f"Dados processados. Período disponível: {min_available_date_calc.strftime('%d/%m/%Y')} a {max_available_date_calc.strftime('%d/%m/%Y')}")
                        except Exception as e:
                            st.sidebar.error(f"Erro ao calcular limites dos eixos: {e}")
                            data_load_placeholder.error("Erro ao calcular limites dos eixos.")
                            axis_limits = None # Garante que axis_limits é None em caso de erro
                    else:
                        st.sidebar.warning("Sem dados válidos após cálculo da variação móvel e limpeza de NaNs.")
                        data_load_placeholder.warning("Sem dados válidos após cálculo.")
                        rolling_changes_full = pd.DataFrame() # Garante que está vazio se não houver dados válidos
            else:
                 if data_load_successful: st.sidebar.warning("Não foi possível calcular variações móveis ou resultado vazio.")
                 data_load_placeholder.warning("Falha no cálculo da variação móvel.")
                 rolling_changes_full = pd.DataFrame() # Garante que está vazio
        else:
            st.error(f"Dados insuficientes/faltando para cálculo ({', '.join(cols_for_calc)}). Verifique se ambos os tickers selecionados tiveram dados carregados.")
            data_load_placeholder.error("Dados cruciais ausentes.")
    else:
        st.error(f"Falha crítica ao carregar dados cruciais: {', '.join(missing_in_loaded)}. Verifique os tickers selecionados.")
        data_load_placeholder.error(f"Falha ao carregar: {', '.join(missing_in_loaded)}.")
else:
    if not close_prices_full.empty:
         loaded_tickers = list(close_prices_full.columns)
         st.error(f"Falha no carregamento inicial. Dados obtidos para: {', '.join(loaded_tickers) if loaded_tickers else 'Nenhum'}. Verifique a conexão ou os tickers não listados.")
    else:
         st.error("Falha ao carregar dados iniciais. Verifique a conexão ou os tickers.")
    data_load_placeholder.error("Falha no carregamento inicial.")


# --- Main Page Area ---
# Só continua se o carregamento foi bem-sucedido E os limites dos eixos foram calculados
if data_load_successful and not rolling_changes_full.empty and axis_limits and min_available_date_calc and max_available_date_calc:
    data_load_placeholder.empty()

    # --- Date Slider (Always Visible at Top) ---
    date_slider_container = st.container()
    with date_slider_container:
        all_available_dates = sorted(list(set(rolling_changes_full.index.date)))
        min_available_date = min_available_date_calc
        max_available_date = max_available_date_calc
        can_color_by_year = 'Ano' in rolling_changes_full.columns and not rolling_changes_full['Ano'].isnull().all()

        start_date_filter, end_date_filter = None, None
        end_date_fixed_window = None
        color_by_year_active = False # Será reavaliado após filtrar

        if view_mode == 'Intervalo de Datas (cor por ano)':
            if can_color_by_year: color_by_year_active = True
            date_options = all_available_dates
            if date_options:
                default_start = min_available_date
                default_end = max_available_date
                selected_date_range = st.select_slider(
                    "Selecione o Intervalo de Datas:",
                    options=date_options, value=(default_start, default_end),
                    format_func=lambda date: date.strftime('%d/%m/%Y'),
                    key="date_range_slider_main",
                    help="Filtre o período exibido no gráfico."
                )
                if selected_date_range and len(selected_date_range) == 2:
                    start_date_filter, end_date_filter = selected_date_range
                else:
                     st.warning("Intervalo inválido selecionado, usando padrão.")
                     start_date_filter, end_date_filter = default_start, default_end
            else: # Deve ser raro chegar aqui se axis_limits foi calculado
                st.warning("Não há datas disponíveis para seleção após cálculo.")
                start_date_filter, end_date_filter = min_available_date, max_available_date

        elif view_mode == f'Janela Fixa ({FIXED_WINDOW_TRADING_DAYS} dias úteis, cor única)':
            color_by_year_active = False
            fixed_window_date_options = all_available_dates
            if fixed_window_date_options:
                default_fixed_end = max_available_date
                if default_fixed_end not in fixed_window_date_options:
                    default_fixed_end = fixed_window_date_options[-1] if fixed_window_date_options else None

                if default_fixed_end:
                    end_date_fixed_window = st.select_slider(
                        f"Selecione a Data Final da Janela ({FIXED_WINDOW_TRADING_DAYS} dias para trás):",
                        options=fixed_window_date_options, value=default_fixed_end,
                        format_func=lambda date: date.strftime('%d/%m/%Y'),
                        key="end_date_fixed_window_slider",
                        help="Arraste para selecionar a data final da janela móvel."
                    )
            # Mensagens de erro/aviso se não houver opções são tratadas implicitamente

    # --- Filter Data Based on Slider Selection ---
    plot_data = pd.DataFrame()
    data_up_to_end = pd.DataFrame()

    if start_date_filter and end_date_filter:
        try:
            mask_date = (rolling_changes_full.index.date >= start_date_filter) & (rolling_changes_full.index.date <= end_date_filter)
            plot_data = rolling_changes_full.loc[mask_date]
        except Exception as filter_err: st.error(f"Erro ao filtrar por data: {filter_err}")
    elif end_date_fixed_window:
        try:
             mask_end_date = rolling_changes_full.index.date <= end_date_fixed_window
             data_up_to_end = rolling_changes_full.loc[mask_end_date]
             if not data_up_to_end.empty: plot_data = data_up_to_end.tail(FIXED_WINDOW_TRADING_DAYS)
        except Exception as filter_err: st.error(f"Erro ao filtrar por janela fixa: {filter_err}")
    else: # Caso inicial ou slider inválido, usar tudo? Ou esperar seleção? Vamos esperar seleção.
        if view_mode == 'Intervalo de Datas (cor por ano)' and not start_date_filter:
             pass # Não plotar nada até selecionar um intervalo válido
        elif view_mode == f'Janela Fixa ({FIXED_WINDOW_TRADING_DAYS} dias úteis, cor única)' and not end_date_fixed_window:
             pass # Não plotar nada até selecionar uma data final válida

    # --- Clean Filtered Data ---
    if not plot_data.empty:
        cols_to_plot = [x_ticker, y_ticker]
        if all(col in plot_data.columns for col in cols_to_plot):
            plot_data = plot_data.dropna(subset=cols_to_plot)
            # Reavaliar se pode colorir por ano baseado nos dados FILTRADOS
            if color_by_year_active and ('Ano' not in plot_data.columns or plot_data['Ano'].isnull().all()):
                 color_by_year_active = False # Desativa se não houver anos nos dados filtrados
        else:
            st.warning(f"Colunas necessárias ({', '.join(cols_to_plot)}) não encontradas nos dados filtrados.")
            plot_data = pd.DataFrame()

    # --- Calculate Metrics (Based on Filtered Data) ---
    tl_perc, tr_perc, bl_perc, br_perc = 0.0, 0.0, 0.0, 0.0
    metrics_calculated = False
    if not plot_data.empty and x_ticker in plot_data.columns and y_ticker in plot_data.columns:
        total_points = len(plot_data)
        if total_points > 0:
            tr_count = len(plot_data[(plot_data[x_ticker] > 0) & (plot_data[y_ticker] > 0)])
            tl_count = len(plot_data[(plot_data[x_ticker] < 0) & (plot_data[y_ticker] > 0)])
            bl_count = len(plot_data[(plot_data[x_ticker] < 0) & (plot_data[y_ticker] < 0)])
            br_count = len(plot_data[(plot_data[x_ticker] > 0) & (plot_data[y_ticker] < 0)])

            if total_points > 0:
                tr_perc = (tr_count / total_points) * 100
                tl_perc = (tl_count / total_points) * 100
                bl_perc = (bl_count / total_points) * 100
                br_perc = (br_count / total_points) * 100
                metrics_calculated = True

    # --- Create the 3-Column Layout ---
    col_left, col_plot, col_right = st.columns([2, 6, 2], gap="large")

    # --- Populate Left Column (Metrics) ---
    with col_left:
        if metrics_calculated:
            st.metric(label="Sup Esq", value=f"{tl_perc:.1f}%")
            st.divider()
            st.metric(label="Inf Esq", value=f"{bl_perc:.1f}%")
        else:
            st.markdown("<div style='height: 150px;'></div>", unsafe_allow_html=True)

    # --- Populate Middle Column (Plot) ---
    with col_plot:
        if not plot_data.empty:
            # --- Plotting Logic ---
            hover_df = plot_data.reset_index()
            date_col_name = hover_df.columns[0]
            try:
                hover_df[date_col_name] = pd.to_datetime(hover_df[date_col_name])
                date_format_ok = True
            except Exception: date_format_ok = False

            hover_data_config = { x_ticker: ':.2f', y_ticker: ':.2f' }
            if date_format_ok: hover_data_config[date_col_name] = '|%d/%m/%Y'
            custom_data_list = [date_col_name]

            scatter_color_arg = None
            plot_labels = {
                x_ticker: f'{AVAILABLE_TICKERS_PT[x_ticker]} (% Móvel {rolling_window_days}d)',
                y_ticker: f'{AVAILABLE_TICKERS_PT[y_ticker]} (% Móvel {rolling_window_days}d)',
                date_col_name: 'Data'
            }
            if color_by_year_active and 'Ano' in plot_data.columns:
                scatter_color_arg = 'Ano'
                hover_data_config['Ano'] = True
                custom_data_list.append('Ano')
                plot_labels['Ano'] = 'Ano'

            fig_main = px.scatter(
                hover_df, x=x_ticker, y=y_ticker, color=scatter_color_arg,
                labels=plot_labels, custom_data=custom_data_list,
                hover_name=date_col_name if date_format_ok else None,
                hover_data=hover_data_config, opacity=0.7
            )
            fig_main.update_traces(marker=dict(size=6))
            fig_main.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5, layer="below")
            fig_main.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5, layer="below")

            ellipse_params = None
            if show_conf_ellipse:
                if len(plot_data) >= 5:
                     ellipse_params = get_confidence_ellipse(
                         plot_data[x_ticker], plot_data[y_ticker], confidence=selected_confidence_level
                     )
                elif len(plot_data) > 1:
                     st.info(f"Dados insuficientes ({len(plot_data)} pontos) para calcular elipse de confiança para o período.", icon="ℹ️")

            if ellipse_params:
                center_x, center_y, width, height, angle_degrees = ellipse_params
                theta = np.linspace(0, 2 * np.pi, 100)
                x_unit = (width / 2) * np.cos(theta); y_unit = (height / 2) * np.sin(theta)
                angle_rad = np.radians(angle_degrees)
                cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
                ellipse_x = center_x + x_unit * cos_angle - y_unit * sin_angle
                ellipse_y = center_y + x_unit * sin_angle + y_unit * cos_angle
                fig_main.add_trace(go.Scatter(
                    x=ellipse_x, y=ellipse_y, mode='lines',
                    line=dict(color=DEFAULT_ELLIPSE_CONF_COLOR, width=2.5),
                    name=f'Elipse {confidence_key}', showlegend=False, hoverinfo='skip'
                ))

            # --- Layout Updates for Plot ---
            layout_updates = {
                 'margin': dict(l=20, r=20, t=50, b=20), # Margens podem precisar de ajuste com título à esquerda
                 'hovermode': 'closest',
                 # <<< USAR OS LIMITES FIXOS CALCULADOS GLOBALMENTE >>>
                 'xaxis_range': axis_limits['x'],
                 'yaxis_range': axis_limits['y'],
                 #'yaxis_scaleanchor': "x",
                 #'yaxis_scaleratio': 1,
                 'height': PLOT_FIXED_HEIGHT,
                 'title_text': f"{AVAILABLE_TICKERS_PT[y_ticker]} vs {AVAILABLE_TICKERS_PT[x_ticker]} ({rolling_window_days}-dias móveis %)",
                 # <<< ALINHAR TÍTULO À ESQUERDA >>>
                 'title_x': 0, # 0 para esquerda, 0.5 para centro, 1 para direita
                 'title_font_size': 16
            }
            if color_by_year_active and 'Ano' in plot_data.columns:
                layout_updates['legend_title_text'] = 'Ano'; layout_updates['showlegend'] = True
            else: layout_updates['showlegend'] = False

            fig_main.update_layout(**layout_updates)
            st.plotly_chart(fig_main, use_container_width=True)

        # Se plot_data estiver vazio, mas o carregamento inicial foi ok...
        elif not rolling_changes_full.empty: # Verifica se há dados subjacentes, mesmo que filtrados para zero
             current_mode_label = view_mode.split('(')[0].strip()
             if current_mode_label == 'Intervalo de Datas':
                  if start_date_filter and end_date_filter:
                       st.info(f"Nenhum dado encontrado para plotar no intervalo selecionado ({start_date_filter.strftime('%d/%m/%Y')} a {end_date_filter.strftime('%d/%m/%Y')}).", icon="ℹ️")
                  else:
                      # Adiciona uma mensagem se nenhum intervalo foi selecionado ainda
                      st.info("Selecione um intervalo de datas no slider acima para visualizar o gráfico.", icon="ℹ️")
             elif current_mode_label == 'Janela Fixa':
                  if end_date_fixed_window:
                     points_found_before_tail = len(data_up_to_end.index.unique()) if not data_up_to_end.empty else 0
                     st.info(f"Nenhum dado encontrado para plotar terminando em {end_date_fixed_window.strftime('%d/%m/%Y')} ou insuficientes ({points_found_before_tail} pts encontrados antes do corte) para a janela de {FIXED_WINDOW_TRADING_DAYS} dias.", icon="ℹ️")
                  else:
                      # Adiciona uma mensagem se nenhuma data final foi selecionada ainda
                      st.info("Selecione uma data final no slider acima para visualizar o gráfico de janela fixa.", icon="ℹ️")
             # Nenhuma mensagem genérica "nenhum dado" aqui, pois é coberto acima

    # --- Populate Right Column (Metrics) ---
    with col_right:
        if metrics_calculated:
            st.metric(label="Sup Dir", value=f"{tr_perc:.1f}%")
            st.divider()
            st.metric(label="Inf Dir", value=f"{br_perc:.1f}%")
        else:
            st.markdown("<div style='height: 150px;'></div>", unsafe_allow_html=True)

    # --- Data Table (Below the 3-column layout) ---
    table_container = st.container()
    with table_container:
        if show_table:
            st.markdown("---")
            if not plot_data.empty:
                table_period_label = ""
                if start_date_filter and end_date_filter:
                    table_period_label = f"Intervalo: {start_date_filter.strftime('%d/%m/%Y')} a {end_date_filter.strftime('%d/%m/%Y')}"
                elif end_date_fixed_window and not plot_data.empty:
                     actual_start_date_in_plot = plot_data.index.min().strftime('%d/%m/%Y')
                     actual_end_date_in_plot = plot_data.index.max().strftime('%d/%m/%Y')
                     table_period_label = f"Janela Fixa: {actual_start_date_in_plot} a {actual_end_date_in_plot}"
                elif end_date_fixed_window:
                     table_period_label = f"Janela Fixa terminando em {end_date_fixed_window.strftime('%d/%m/%Y')}"

                st.markdown(f"### Tabela de Dados ({table_period_label} - Máx 100 Pontos)")
                cols_to_show_in_table = [x_ticker, y_ticker]
                if 'Ano' in plot_data.columns: cols_to_show_in_table.append('Ano')

                display_df = plot_data[[col for col in cols_to_show_in_table if col in plot_data.columns]].sort_index(ascending=False)
                format_dict = {col: "{:.2f}%" for col in [x_ticker, y_ticker] if col in display_df.columns}
                if 'Ano' in display_df.columns: format_dict['Ano'] = "{:}"

                st.dataframe(display_df.head(100).style.format(format_dict))
            else:
                st.info("Tabela habilitada, mas sem dados filtrados para exibir com as configurações atuais.", icon="ℹ️")

# --- Handle Initial Load Failures or Empty Results ---
elif not data_load_successful or rolling_changes_full.empty:
     pass # Messages handled by data_load_placeholder or specific checks
elif not axis_limits:
     # Esta mensagem agora indica que o cálculo inicial dos limites falhou
     st.warning("Não foi possível definir os limites dos eixos a partir do histórico completo. Verifique os dados carregados e mensagens de erro/aviso.", icon="⚠️")
else:
    st.error("Ocorreu um erro inesperado na preparação dos dados para exibição.", icon="🚨")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Fonte:** Yahoo Finance (`yfinance`)")
st.sidebar.markdown(f"**Versão:** Quadrants v1.4 - LeftTitle/FixedAxes") # <<< MODIFICADO: Atualizar versão >>>