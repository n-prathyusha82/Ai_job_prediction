import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="AI Job Salary Prediction",
    page_icon="💼",
    layout="wide"
)

# ==================================
# LOAD MODEL
# ==================================
try:
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ==================================
# HEADER
# ==================================
st.title("💼 AI Job Salary Prediction")

st.markdown(
    "Predict AI Job Salaries using Machine Learning"
)

# ==================================
# SIDEBAR INPUTS
# ==================================
st.sidebar.header("Job Information")

work_year = st.sidebar.number_input(
    "Work Year",
    min_value=2020,
    max_value=2030,
    value=2025
)

experience_level = st.sidebar.selectbox(
    "Experience Level",
    ["EN", "MI", "SE", "EX"]
)

employment_type = st.sidebar.selectbox(
    "Employment Type",
    ["FT", "PT", "CT", "FL"]
)

company_size = st.sidebar.selectbox(
    "Company Size",
    ["S", "M", "L"]
)

remote_ratio = st.sidebar.slider(
    "Remote Ratio",
    0,
    100,
    50
)

predict_btn = st.sidebar.button(
    "Predict Salary"
)

# ==================================
# PREDICTION
# ==================================
if predict_btn:

    try:

        # TEMPORARY MANUAL ENCODING
        exp_map = {
            "EN": 0,
            "MI": 1,
            "SE": 2,
            "EX": 3
        }

        emp_map = {
            "FT": 0,
            "PT": 1,
            "CT": 2,
            "FL": 3
        }

        company_map = {
            "S": 0,
            "M": 1,
            "L": 2
        }

        input_df = pd.DataFrame({

            "work_year": [work_year],

            "experience_level": [
                exp_map[experience_level]
            ],

            "employment_type": [
                emp_map[employment_type]
            ],

            "company_size": [
                company_map[company_size]
            ],

            "remote_ratio": [remote_ratio]

        })

        X = scaler.transform(input_df)

        prediction = model.predict(X)[0]

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Predicted Salary"
            )

            st.success(
                f"${prediction:,.2f}"
            )

        with col2:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=float(prediction),
                    title={
                        "text":
                        "Predicted Salary (USD)"
                    },
                    gauge={
                        "axis": {
                            "range":
                            [0, 500000]
                        }
                    }
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader(
            "Input Data"
        )

        st.dataframe(
            input_df
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

else:

    st.info(
        "Enter details and click Predict Salary."
    )