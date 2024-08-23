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
    st.title("Demanda de Eletricidade")

    st.header("Conjunto Completo")
    all_data = get_gold_data()

    # Plot the data
    fig = px.line(all_data, x="ds", y="y", title="Demanda de Eletricidade")

    # Add names to the X and Y axis
    fig.update_xaxes(title_text="Data")
    fig.update_yaxes(title_text="Demanda (MW)")

    st.plotly_chart(fig)

    st.header("Previsão de Demanda")
    predictions = load_predictions()

    # Select the models to compare
    models = [col for col in predictions.columns if "Regressor" in col and '-' not in col]
    selected_models = st.multiselect("Modelos", models)

    # Select the confidence levels
    confidence_levels = sorted(set(sorted([col.split("-")[2] for col in predictions.columns if '-' in col])))
    selected_confidence_levels = st.multiselect("Níveis de Confiança", confidence_levels)

    # Generate the plot
    start_date = pendulum.now().start_of('day').naive()

    # Filter the dataframe to show data from 30 days before the forecast start date
    df = predictions[predictions['ds'] >= start_date - pendulum.duration(days=30)]

    figure = create_plotly_figure(df, selected_models, selected_confidence_levels)
    st.plotly_chart(figure)
    if st.checkbox("Mostrar Dados"):
        st.dataframe(df)
if __name__ == "__main__":
    main()
