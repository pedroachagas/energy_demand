import os
import streamlit as st
from dotenv import load_dotenv
from utils import get_blob_data

# Load environment variables
load_dotenv()

# Get the file name from the environment variable
FILE_NAME = os.getenv("AZURE_FILE_NAME")

def main():
    st.title("Consumo de Energia")

    try:
        df = get_blob_data(FILE_NAME)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

    if df is not None:
        # Plot the data
        st.subheader("Data Visualization")
        st.line_chart(
            data=df,
            x_label="Data",
            y_label="Consumo (MW)",
        )


if __name__ == "__main__":
    main()
