import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------
# Load Data
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("advertising.csv")
    return df

df = load_data()

# ------------------------
# Train Models
# ------------------------
@st.cache_resource
def train_models(df):
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}
        trained_models[name] = model

    results_df = pd.DataFrame(results).T
    return trained_models, results_df

trained_models, results_df = train_models(df)

# ------------------------
# Sidebar Navigation
# ------------------------
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 EDA", "📈 Model Comparison", "🤖 Prediction"])

# ------------------------
# EDA Page
# ------------------------
if page == "📊 EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Statistics")
    st.write(df.describe())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Pairplot
    st.subheader("Pairplot")
    st.pyplot(sns.pairplot(df))

    # Boxplots
    st.subheader("Outlier Analysis")
    fig, axs = plt.subplots(3, figsize=(5, 6))
    sns.boxplot(df['TV'], ax=axs[0])
    sns.boxplot(df['Radio'], ax=axs[1])
    sns.boxplot(df['Newspaper'], ax=axs[2])
    plt.tight_layout()
    st.pyplot(fig)

# ------------------------
# Model Comparison Page
# ------------------------
elif page == "📈 Model Comparison":
    st.title("Model Performance Comparison")
    st.dataframe(results_df.style.highlight_max(axis=0))

    st.write("✅ Best model is the one with highest **R²** and lowest **RMSE**.")

# ------------------------
# Prediction Page
# ------------------------
elif page == "🤖 Prediction":
    st.title("Sales Prediction Tool")
    st.write("Predict sales based on your advertising budget.")

    # Choose model
    model_choice = st.selectbox("Select Model", list(trained_models.keys()))
    model = trained_models[model_choice]

    # User input sliders
    tv = st.slider("TV Advertising Budget (in $1000s)", 0, 300, 150)
    radio = st.slider("Radio Advertising Budget (in $1000s)", 0, 50, 25)
    newspaper = st.slider("Newspaper Advertising Budget (in $1000s)", 0, 100, 20)

    input_data = pd.DataFrame({"TV": [tv], "Radio": [radio], "Newspaper": [newspaper]})

    if st.button("Predict Sales"):
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Sales using {model_choice}: **{prediction:.2f} units**")

    # Random Test Data
    st.subheader("🔎 Try Example Random Test Data")
    if st.button("Generate Test Data"):
        test_data = pd.DataFrame({
            "TV": np.random.randint(50, 300, size=5),
            "Radio": np.random.randint(10, 50, size=5),
            "Newspaper": np.random.randint(0, 100, size=5)
        })
        test_data["Predicted Sales"] = model.predict(test_data)
        st.dataframe(test_data)
