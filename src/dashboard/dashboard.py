import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv
from utils import get_gold_data, load_predictions, create_plotly_figure
import plotly.express as px
import pendulum

# Load environment variables
load_dotenv()

def main():
    st.title("Consumo de Energia")

    st.header("Conjunto Completo")
    all_data = get_gold_data()

    # Plot the data
    fig = px.line(all_data, x="ds", y="y", title="Consumo de Energia")
    st.plotly_chart(fig)

    st.header("Predições")

    predictions = load_predictions()

    # Select the models to compare
    models = [col for col in predictions.columns if "Regressor" in col and '-' not in col]
    selected_models = st.multiselect("Modelos", models)

    # Select the confidence levels
    confidence_levels = sorted(set(sorted([col.split("-")[2] for col in predictions.columns if '-' in col])))
    selected_confidence_levels = st.multiselect("Níveis de Confiança", confidence_levels)

    # Generate the plot
    # Get the date 30 days ago
    start_date = pendulum.now().subtract(days=30).start_of('day')

    # Filter the dataframe to show data from 30 days before the forecast start date
    df = all_data[all_data['ds'] >= start_date - pendulum.duration(days=30)].merge(
        predictions.drop(columns=['unique_id','y']), on=["ds"], how="left"
    )

    figure = create_plotly_figure(df, selected_models, selected_confidence_levels)
    st.plotly_chart(figure)

if __name__ == "__main__":
    main()
