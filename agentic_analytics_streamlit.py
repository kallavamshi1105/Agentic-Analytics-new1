import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
st.set_page_config(page_title="Agentic Analytics", layout="wide")

# --- Helper: Read uploaded file ---
def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    try:
        # Handle CSV files
        if uploaded_file.name.lower().endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(uploaded_file, encoding="latin1")

        # Handle Excel files
        elif uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            try:
                return pd.read_excel(uploaded_file, engine="openpyxl")
            except ImportError:
                return pd.read_excel(uploaded_file, engine="xlrd")

        else:
            st.error("Unsupported file type. Please upload CSV or Excel.")
            return None

    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

# --- Helper: Generate AI insights ---
 
def generate_insights(df, query):
    try:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            return "OpenAI API key is missing. Please configure OPENAI_API_KEY."

        client = OpenAI(api_key=api_key)

        prompt = f"""
        You are a data analyst.
        Dataset columns: {list(df.columns)}

        User question: {query}

        Provide a clear and useful analytical answer.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI insight generation failed: {e}"


# --- Main App ---
def main():
    st.title("📊 Agentic Analytics — Data Visualization")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        df = read_uploaded_file(uploaded_file)
        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df.head())

            st.subheader("Data Summary")
            st.write(df.describe(include='all'))

            st.subheader("Ask a Question About Your Data")
            query = st.text_input("Enter your query")
            if query:
                insight = generate_insights(df, query)
                st.markdown(f"**AI Insight:** {insight}")

            st.subheader("Visualization")
            x_axis = st.selectbox("Select X-axis", df.columns)
            y_axis = st.selectbox("Select Y-axis", df.columns)
            chart_type = st.selectbox("Select Chart Type", ["line", "bar", "scatter", "box"])

            if st.button("Generate Chart"):
                if chart_type == "line":
                    fig = px.line(df, x=x_axis, y=y_axis)
                elif chart_type == "bar":
                    fig = px.bar(df, x=x_axis, y=y_axis)
                elif chart_type == "scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis)
                elif chart_type == "box":
                    fig = px.box(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    import os
    import sys
    import subprocess

    # Relaunch through Streamlit if not already running inside it
    if not any("streamlit" in arg for arg in sys.argv):
        script_path = os.path.abspath(__file__)
        subprocess.run(["streamlit", "run", script_path])
    else:
        main()
