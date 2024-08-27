import loguru as logging
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import sys
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds
import pendulum
from typing import List

# Initialize logger
logger = logging.logger
logger.info("Starting dashboard")

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config.config import config
from src.utils.azure_utils import load_gold_data, load_predictions
from src.utils.logging_utils import logger


def create_demand_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create a line plot of daily electricity demand.

    Args:
        df (pd.DataFrame): DataFrame containing 'date' and 'daily_carga_mw' columns.

    Returns:
        go.Figure: Plotly figure object with the demand plot.
    """
    fig = px.line(df, x="date", y="daily_carga_mw", title="Demanda de Eletricidade Diária")
    fig.update_xaxes(title_text="Data")
    fig.update_yaxes(title_text="Demanda (MW)")
    return fig


def create_forecast_plot(df: pd.DataFrame, models: List[str], confidence_levels: List[int]) -> go.Figure:
    """
    Create a forecast plot with confidence intervals.

    Args:
        df (pd.DataFrame): DataFrame containing actual and forecasted values.
        models (List[str]): List of model names to include in the plot.
        confidence_levels (List[int]): List of confidence levels for intervals.

    Returns:
        go.Figure: Plotly figure object with the forecast plot.
    """
    fig = go.Figure()

    # Add actual values
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines+markers', name='Real', line=dict(color='black', width=2)))

    # Add predictions and confidence intervals for each model
    for model in models:
        fig.add_trace(go.Scatter(x=df['ds'], y=df[model], mode='lines', name=f'{model}', line=dict(width=2)))

        for conf in confidence_levels:
            lo_col = f'{model}-lo-{conf}'
            hi_col = f'{model}-hi-{conf}'

            fig.add_trace(go.Scatter(x=df['ds'], y=df[hi_col], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df['ds'], y=df[lo_col], mode='lines', fill='tonexty',
                                     fillcolor=f'rgba(0,100,80,{0.1 * conf/100})', line=dict(width=0),
                                     name=f'{model} {conf}% CI', showlegend=conf == confidence_levels[-1]))

    fig.update_layout(title="Previsões dos Modelos com Intervalos de Confiança",
                      xaxis_title="Data", yaxis_title="Demanda (MW)",
                      legend_title="Legenda", hovermode="x")
    return fig


def main() -> None:
    """
    Main function to render the Streamlit dashboard.
    """
    st.title("Demanda de Eletricidade - Dashboard")

    try:
        # Load data
        df_gold = load_gold_data().rename(columns={"ds": "date", "y": "daily_carga_mw"})
        df_predictions = load_predictions()

        # Historical Demand Section
        st.header("Demanda Histórica")
        fig_demand = create_demand_plot(df_gold)
        st.plotly_chart(fig_demand)

        # Forecast Section
        st.header("Previsão de Demanda")

        # Select models to display
        models = [col for col in df_predictions.columns if "Regressor" in col and '-' not in col]
        selected_models = st.multiselect("Modelos", models, default=models[2] if models else None)

        # Select confidence levels
        confidence_levels = sorted(set([int(col.split("-")[-1]) for col in df_predictions.columns if '-' in col]))
        selected_confidence_levels = st.multiselect("Níveis de Confiança", confidence_levels, default=[95])

        if selected_models and selected_confidence_levels:
            # Filter data to show only the last 30 days of historical data and the forecast
            cutoff_date = pendulum.now().subtract(days=30).start_of('day').naive()
            df_forecast = df_predictions[df_predictions['ds'] >= cutoff_date]

            fig_forecast = create_forecast_plot(df_forecast, selected_models, selected_confidence_levels)
            st.plotly_chart(fig_forecast)

            if st.checkbox("Mostrar Dados de Previsão"):
                st.dataframe(df_forecast)
        else:
            st.warning("Por favor, selecione pelo menos um modelo e um nível de confiança para visualizar as previsões.")

    except Exception as e:
        logger.error(f"Error in dashboard: {str(e)}")
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
