# sp_vix_app_plotly_single_fixedaxis_square_30y_slider_main_v12_full_period_default.py
# FINAL VERSION BASED ON USER REQUESTS
# Features:
# - Select two tickers (expanded list including sectors and Russell 2000)
# - Calculate rolling % change
# - Display scatter plot with fixed axes (based on full history) and square aspect ratio
# - Two view modes (selectable in sidebar):
#   1. Date Range: Slider selects start/end date, colored by year, defaults to FULL range.
#   2. Fixed Window: Slider selects end date, shows last N days, single color.
# - Optional Confidence Ellipse (calculated on displayed data, selectable level in sidebar).
# - Optional data table (in main area, controlled by sidebar checkbox).
# - Layout: Controls in sidebar, slider+plot+table in main area.

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
APP_TITLE = "Analisador de Variação Móvel com Elipse de Confiança"

# Mapeamento Ticker -> Nome (Inglês/Referência) - Updated List
AVAILABLE_TICKERS = OrderedDict([
    # Índices Principais
    ("^GSPC", "S&P 500"),
    ("^IXIC", "NASDAQ Composite"),
    ("^DJI", "Dow Jones Industrial Average"),
    ("^RUT", "Russell 2000"),
    ("^VIX", "VIX"),
    ("^GDAXI", "DAX (Germany)"),
    # Commodities / Moedas
    ("GC=F", "Gold Futures"),
    ("DX-Y.NYB", "US Dollar Index (DXY)"),
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

# Mapeamento Ticker -> Nome (Português para UI) - Updated List
AVAILABLE_TICKERS_PT = OrderedDict([
    # Índices Principais
    ("^GSPC", "S&P 500"),
    ("^IXIC", "NASDAQ Composite"),
    ("^DJI", "Dow Jones Industrial Average"),
    ("^RUT", "Russell 2000"),
    ("^VIX", "VIX"),
    ("^GDAXI", "DAX (Alemanha)"),
     # Commodities / Moedas
    ("GC=F", "Ouro (Futuros)"),
    ("DX-Y.NYB", "Índice Dólar Americano (DXY)"),
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
PLOT_FIXED_HEIGHT = 700
AXIS_PADDING_FACTOR = 0.05
FIXED_WINDOW_TRADING_DAYS = 252
CONFIDENCE_LEVELS = {
    "68% (~1σ)": 0.68,
    "90%": 0.90,
    "95% (~2σ)": 0.95,
    "99% (~3σ)": 0.99,
    "99.7%": 0.997
}
DEFAULT_CONFIDENCE_KEY = "95% (~2σ)"
DEFAULT_ELLIPSE_CONF_COLOR = "#A9A9A9" # DarkGray (was LightGray D3D3D3)

# --- Caching Functions ---
@st.cache_data(ttl=3600)
def load_data(tickers_tuple, start_date, end_date):
    """Downloads historical stock data for a tuple of tickers."""
    tickers_list = list(tickers_tuple)
    print(f"Buscando dados para: {', '.join(tickers_list)}")
    try:
        data = yf.download(tickers_list, start=start_date, end=end_date, progress=False)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data.get('Close', pd.DataFrame())
            close_prices = close_prices.dropna(axis=1, how='all')
        else:
             if 'Close' in data.columns:
                 close_prices = data[['Close']]
                 if len(tickers_list) == 1: close_prices.columns = tickers_list
             elif len(tickers_list) == 1 and hasattr(data, 'name') and data.name == 'Close':
                 close_prices = data.to_frame(name=tickers_list[0])
             else:
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
    except Exception as e: st.error(f"Erro Crítico ao buscar dados: {e}"); return pd.DataFrame()


@st.cache_data
def calculate_rolling_change(data, window):
    """Calculates rolling percentage change and adds 'Ano' column."""
    if data.empty or window <= 1: return pd.DataFrame()
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
                     pass # Continue without 'Ano' if conversion fails

        cols_to_check_na = [col for col in change_pct.columns if col != 'Ano']
        return change_pct.dropna(how='all', subset=cols_to_check_na)
    except Exception as e: st.error(f"Erro ao calcular variação móvel: {e}"); return pd.DataFrame()


# --- Function to Calculate Confidence Ellipse Parameters ---
def get_confidence_ellipse(x_data, y_data, confidence=0.95):
    """Calculates parameters for a confidence ellipse based on data."""
    if len(x_data) < 2 or len(y_data) < 2: return None
    try:
        cov_matrix = np.cov(x_data, y_data)
        if np.isclose(np.linalg.det(cov_matrix), 0):
             st.sidebar.warning("Variância próxima de zero, elipse não calculada.", icon="⚠️")
             return None

        center_x, center_y = np.mean(x_data), np.mean(y_data)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        scale_factor = np.sqrt(chi2.ppf(confidence, df=2))
        eigenvalues = np.maximum(eigenvalues, 0) # Ensure non-negative before sqrt
        width = 2 * scale_factor * np.sqrt(eigenvalues[0]) # Diameter along first axis
        height = 2 * scale_factor * np.sqrt(eigenvalues[1]) # Diameter along second axis
        angle_rad = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
        angle_degrees = np.degrees(angle_rad)

        # Ensure width corresponds to the larger eigenvalue for consistency if desired
        # This mostly affects the naming 'width'/'height', plot is correct either way
        # if eigenvalues[0] < eigenvalues[1]:
        #    width, height = height, width
        #    angle_rad = np.arctan2(eigenvectors[1, 1], eigenvectors[1, 0]) # Angle of the other axis
        #    angle_degrees = np.degrees(angle_rad)

        return center_x, center_y, width, height, angle_degrees
    except Exception as e:
        st.sidebar.error(f"Erro Elipse: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(page_title=APP_TITLE, layout="wide")

# --- Sidebar Controls ---
st.sidebar.header("Configuração")
ticker_list_pt = list(AVAILABLE_TICKERS_PT.values()) # Use PT names for selection
# Find default indices for selectboxes (e.g., S&P 500 and VIX)
default_y_index = 0 # S&P 500
default_x_index = 4 # VIX (adjust if order changes)
try:
    default_y_index = ticker_list_pt.index("S&P 500")
    default_x_index = ticker_list_pt.index("VIX")
except ValueError:
    pass # Keep 0 and 4 if not found

y_ticker_name_pt = st.sidebar.selectbox("Ticker Eixo Y", options=ticker_list_pt, index=default_y_index, help="Selecione o índice/ativo para o eixo Y.")
y_ticker = [k for k, v in AVAILABLE_TICKERS_PT.items() if v == y_ticker_name_pt][0]
x_ticker_name_pt = st.sidebar.selectbox("Ticker Eixo X", options=ticker_list_pt, index=default_x_index, help="Selecione o índice/ativo para o eixo X.")
x_ticker = [k for k, v in AVAILABLE_TICKERS_PT.items() if v == x_ticker_name_pt][0]

if x_ticker == y_ticker:
    st.sidebar.error("Por favor, selecione tickers diferentes para os eixos X e Y.")
    st.stop()

rolling_window_days = st.sidebar.number_input("Janela Móvel (Dias Úteis)", min_value=2, max_value=252 * 5, value=21, step=1, help="Número de dias úteis para o cálculo da variação percentual móvel.")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Modo de Visualização")
view_mode = st.sidebar.radio(
    "Selecione o Modo:",
    ('Intervalo de Datas (cor por ano)', f'Janela Fixa ({FIXED_WINDOW_TRADING_DAYS} dias úteis, cor única)'),
    key="view_mode_radio",
    help="Escolha como filtrar os dados: por intervalo ou janela fixa."
)

# --- Confidence Ellipse Controls ---
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
start_date_fetch = end_date_max - datetime.timedelta(days=INITIAL_YEARS_FETCH * 365 + (INITIAL_YEARS_FETCH // 4))
tickers_to_fetch = tuple(sorted(list(set([x_ticker, y_ticker]))))

close_prices_full = pd.DataFrame()
rolling_changes_full = pd.DataFrame()
axis_limits = None
data_load_successful = False
min_available_date_calc = None
max_available_date_calc = None

with st.spinner(f"Carregando e processando dados ({INITIAL_YEARS_FETCH} anos)..."):
    close_prices_full = load_data(tickers_to_fetch, start_date_fetch, end_date_max)
    if not close_prices_full.empty:
        missing_in_loaded = [t for t in tickers_to_fetch if t not in close_prices_full.columns]
        if not missing_in_loaded:
            data_load_successful = True
            cols_for_calc = [t for t in [x_ticker, y_ticker] if t in close_prices_full.columns]
            if len(cols_for_calc) == 2:
                rolling_changes_full = calculate_rolling_change(close_prices_full[cols_for_calc], rolling_window_days)
                if not rolling_changes_full.empty and x_ticker in rolling_changes_full.columns and y_ticker in rolling_changes_full.columns:
                    if not isinstance(rolling_changes_full.index, pd.DatetimeIndex):
                        try: rolling_changes_full.index = pd.to_datetime(rolling_changes_full.index)
                        except Exception as e:
                            st.error(f"Falha ao converter índice para Datetime: {e}")
                            rolling_changes_full = pd.DataFrame()
                    if not rolling_changes_full.empty:
                        min_available_date_calc = rolling_changes_full.index.min().date()
                        max_available_date_calc = rolling_changes_full.index.max().date()
                        can_color_by_year = 'Ano' in rolling_changes_full.columns
                        try:
                            valid_data_for_axes = rolling_changes_full[[x_ticker, y_ticker]].dropna()
                            if not valid_data_for_axes.empty:
                                min_x = valid_data_for_axes[x_ticker].min(); max_x = valid_data_for_axes[x_ticker].max()
                                min_y = valid_data_for_axes[y_ticker].min(); max_y = valid_data_for_axes[y_ticker].max()
                                x_range = max_x - min_x; y_range = max_y - min_y
                                x_pad = x_range * AXIS_PADDING_FACTOR if x_range > 1e-9 else 1
                                y_pad = y_range * AXIS_PADDING_FACTOR if y_range > 1e-9 else 1
                                axis_limits = {'x': [min_x - x_pad, max_x + x_pad], 'y': [min_y - y_pad, max_y + y_pad]}
                            else: st.sidebar.warning("Sem dados válidos para definir eixos.")
                        except Exception as e: st.sidebar.error(f"Erro ao calcular eixos: {e}")
                else:
                    if data_load_successful: st.sidebar.warning("Não foi possível calcular variações móveis.")
            else: st.error(f"Dados insuficientes/faltando ({', '.join(cols_for_calc)}).")
        else: st.error(f"Falha ao carregar dados cruciais: {', '.join(missing_in_loaded)}.")

# --- Main Page Area ---
date_slider_container = st.container()
plot_container = st.container()
table_container = st.container()

# Proceed only if data loaded, calculated, axes limits set, and dates available
if data_load_successful and not rolling_changes_full.empty and axis_limits and min_available_date_calc and max_available_date_calc:

    all_available_dates = sorted(list(set(rolling_changes_full.index.date)))
    # Ensure min/max dates are from actual calculated data
    min_available_date = min_available_date_calc
    max_available_date = max_available_date_calc

    can_color_by_year = 'Ano' in rolling_changes_full.columns

    start_date_filter, end_date_filter = None, None
    end_date_fixed_window = None
    color_by_year_active = False

    # --- Date Slider Placement ---
    with date_slider_container:
        if view_mode == 'Intervalo de Datas (cor por ano)':
            if can_color_by_year: color_by_year_active = True
            else:
                 st.warning("Coluna 'Ano' não disponível. Coloração desativada.", icon="⚠️")
                 color_by_year_active = False

            date_options = all_available_dates
            if date_options:
                 # *** Default to full range ***
                selected_date_range = st.select_slider(
                    "Selecione o Intervalo de Datas:",
                    options=date_options,
                    value=(min_available_date, max_available_date), # Default to full available range
                    format_func=lambda date: date.strftime('%d/%m/%Y'),
                    key="date_range_slider_main",
                    help="Filtre o período exibido no gráfico."
                )
                start_date_filter, end_date_filter = selected_date_range
            else:
                st.warning("Não há opções de data disponíveis.")
                start_date_filter, end_date_filter = min_available_date, max_available_date # Fallback

        elif view_mode == f'Janela Fixa ({FIXED_WINDOW_TRADING_DAYS} dias úteis, cor única)':
            color_by_year_active = False
            fixed_window_date_options = all_available_dates
            if fixed_window_date_options:
                end_date_fixed_window = st.select_slider(
                    f"Selecione a Data Final da Janela ({FIXED_WINDOW_TRADING_DAYS} dias para trás):",
                    options=fixed_window_date_options, value=max_available_date,
                    format_func=lambda date: date.strftime('%d/%m/%Y'), key="end_date_fixed_window_slider",
                    help="Arraste para selecionar a data final."
                )
            else:
                st.warning("Não há opções de data disponíveis.")
                end_date_fixed_window = max_available_date

    # --- Filter Data ---
    plot_data = pd.DataFrame()
    data_filtered_by_date = pd.DataFrame()
    data_up_to_end = pd.DataFrame() # Define scope for info message
    if start_date_filter and end_date_filter:
        mask_date = (rolling_changes_full.index >= pd.to_datetime(start_date_filter)) & (rolling_changes_full.index <= pd.to_datetime(end_date_filter))
        data_filtered_by_date = rolling_changes_full.loc[mask_date]
    elif end_date_fixed_window:
         mask_end_date = rolling_changes_full.index <= pd.to_datetime(end_date_fixed_window)
         data_up_to_end = rolling_changes_full.loc[mask_end_date]
         data_filtered_by_date = data_up_to_end.tail(FIXED_WINDOW_TRADING_DAYS)

    # --- Prepare Plot Data ---
    if not data_filtered_by_date.empty:
        if x_ticker in data_filtered_by_date.columns and y_ticker in data_filtered_by_date.columns:
            cols_to_plot = [x_ticker, y_ticker]
            required_cols = cols_to_plot + (['Ano'] if color_by_year_active and can_color_by_year else [])
            if all(col in data_filtered_by_date.columns for col in required_cols):
                plot_data = data_filtered_by_date[required_cols].dropna(subset=cols_to_plot)
            else:
                 plot_data = data_filtered_by_date[cols_to_plot].dropna(subset=cols_to_plot)
                 if color_by_year_active: color_by_year_active = False # Disable if Ano column missing after filter
        else: plot_data = pd.DataFrame()

    # --- Display Graph ---
    with plot_container:
        if not plot_data.empty and x_ticker in plot_data.columns and y_ticker in plot_data.columns:
            hover_df = plot_data.reset_index()
            date_col_name = hover_df.columns[0]
            hover_data_config = { x_ticker: ':.2f', y_ticker: ':.2f', date_col_name: '|%d/%m/%Y' }
            custom_data_list = [date_col_name]
            if 'Ano' in hover_df.columns:
                hover_data_config['Ano'] = True
                custom_data_list.append('Ano')

            scatter_color_arg = 'Ano' if color_by_year_active and 'Ano' in plot_data.columns else None
            plot_labels = { x_ticker: f'{AVAILABLE_TICKERS_PT[x_ticker]} (% Móvel {rolling_window_days}d)', y_ticker: f'{AVAILABLE_TICKERS_PT[y_ticker]} (% Móvel {rolling_window_days}d)', date_col_name: 'Data'}
            if color_by_year_active and 'Ano' in plot_data.columns: plot_labels['Ano'] = 'Ano'

            fig_main = px.scatter(
                hover_df, x=x_ticker, y=y_ticker, color=scatter_color_arg,
                labels=plot_labels, custom_data=custom_data_list,
                hover_name=date_col_name, hover_data=hover_data_config, opacity=0.7
            )
            fig_main.update_traces(marker=dict(size=6))
            fig_main.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig_main.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)

            # --- Add Confidence Ellipse Trace ---
            ellipse_params = None
            if show_conf_ellipse:
                if len(plot_data) >= 2: # Check length of the actual data to be plotted
                     ellipse_params = get_confidence_ellipse(
                         plot_data[x_ticker], plot_data[y_ticker], confidence=selected_confidence_level
                     )
                else:
                     st.info(f"Dados insuficientes ({len(plot_data)}) para calcular a elipse de confiança para o período selecionado.", icon="ℹ️")

            if ellipse_params:
                center_x, center_y, width, height, angle_degrees = ellipse_params
                theta = np.linspace(0, 2 * np.pi, 100)
                x_unit = (width / 2) * np.cos(theta)
                y_unit = (height / 2) * np.sin(theta)
                angle_rad = np.radians(angle_degrees)
                cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
                ellipse_x = center_x + x_unit * cos_angle - y_unit * sin_angle
                ellipse_y = center_y + x_unit * sin_angle + y_unit * cos_angle

                fig_main.add_trace(go.Scatter(
                    x=ellipse_x, y=ellipse_y, mode='lines',
                    line=dict(color=DEFAULT_ELLIPSE_CONF_COLOR, width=1.5, dash="dot"),
                    name=f'Elipse {confidence_key}', showlegend=False, hoverinfo='skip'
                ))

            # --- Layout Updates ---
            layout_updates = {
                 'margin': dict(l=10, r=10, t=30, b=10), 'hovermode': 'closest',
                 'xaxis_range': axis_limits['x'], 'yaxis_range': axis_limits['y'],
                 'yaxis_scaleanchor': "x", 'yaxis_scaleratio': 1,
                 'height': PLOT_FIXED_HEIGHT,
                 'title_text': f"{AVAILABLE_TICKERS_PT[y_ticker]} vs {AVAILABLE_TICKERS_PT[x_ticker]} ({rolling_window_days}-dias móveis %)",
                 'title_x': 0.5
            }
            # Show legend only if coloring by year AND the 'Ano' column is actually present
            if color_by_year_active and 'Ano' in plot_data.columns:
                layout_updates['legend_title_text'] = 'Ano'; layout_updates['showlegend'] = True
            else: layout_updates['showlegend'] = False

            fig_main.update_layout(**layout_updates)
            st.plotly_chart(fig_main, use_container_width=True)

        # --- Handle Empty Plot Data ---
        elif not rolling_changes_full.empty:
             current_mode_label = view_mode.split('(')[0].strip()
             if current_mode_label == 'Intervalo de Datas':
                  if start_date_filter and end_date_filter: st.info(f"Nenhum dado encontrado para o intervalo ({start_date_filter.strftime('%d/%m/%Y')} a {end_date_filter.strftime('%d/%m/%Y')}).", icon="ℹ️")
             elif current_mode_label == 'Janela Fixa':
                  if end_date_fixed_window:
                     points_found_before_tail = len(data_up_to_end.index.unique()) if not data_up_to_end.empty else 0
                     st.info(f"Nenhum dado encontrado terminando em {end_date_fixed_window.strftime('%d/%m/%Y')} ou insuficientes ({points_found_before_tail} pts antes do corte) para a janela de {FIXED_WINDOW_TRADING_DAYS} dias.", icon="ℹ️")

    # --- Optional Data Table ---
    with table_container:
        if show_table and not plot_data.empty:
           st.markdown("---")
           table_period_label = ""
           if start_date_filter and end_date_filter: table_period_label = f"Intervalo: {start_date_filter.strftime('%d/%m/%Y')} a {end_date_filter.strftime('%d/%m/%Y')}"
           elif end_date_fixed_window:
                actual_start_date_in_plot = plot_data.index.min().strftime('%d/%m/%Y') if not plot_data.empty else "N/A"
                table_period_label = f"Janela Fixa: {actual_start_date_in_plot} a {end_date_fixed_window.strftime('%d/%m/%Y')}"
           st.markdown(f"### Tabela de Dados ({table_period_label} - Máx 100 Pontos)")
           cols_to_show_in_table = [x_ticker, y_ticker]
           # Only include 'Ano' if it was actually used for coloring (i.e., exists and mode allows)
           if color_by_year_active and 'Ano' in plot_data.columns:
                cols_to_show_in_table.append('Ano')
           display_df = plot_data[cols_to_show_in_table].sort_index(ascending=False)
           st.dataframe(display_df.head(100).style.format({col: "{:.2f}%" for col in [x_ticker, y_ticker]}))
        elif show_table and plot_data.empty:
            st.markdown("---")
            st.info("Tabela habilitada, mas sem dados filtrados para exibir.", icon="ℹ️")

# --- Handle Initial Load Failures ---
elif not data_load_successful or rolling_changes_full.empty:
     st.warning("Não foi possível carregar ou processar os dados necessários. Verifique as configurações ou tente novamente.", icon="⚠️")
elif not axis_limits:
     st.warning("Não foi possível definir os limites dos eixos. Verifique os dados carregados e mensagens na barra lateral.", icon="⚠️")
else: # Catch all for unexpected states where data might seem ok but dates failed etc.
    st.error("Ocorreu um erro inesperado na preparação dos dados para exibição.", icon="🚨")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Fonte:** Yahoo Finance (`yfinance`)")