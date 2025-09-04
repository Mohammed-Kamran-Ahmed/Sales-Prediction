import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Sales Prediction App", layout="wide")

# ------------------------
# Load data + train model
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("advertising.csv")
    return df

@st.cache_resource
def train_model(df):
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

df = load_data()
model = train_model(df)

# ------------------------
# Sidebar Navigation
# ------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 EDA", "🤖 Prediction"])

# ------------------------
# EDA Page
# ------------------------
if page == "📊 EDA":
    st.title("Exploratory Data Analysis")
    st.write("Let's explore the Advertising dataset!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    st.write(df.describe())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Pairplot
    st.subheader("Pairplot")
    st.write("Scatter relationships between variables")
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
# Prediction Page
# ------------------------
elif page == "🤖 Prediction":
    st.title("Sales Prediction Tool")
    st.write("Predict sales based on your advertising budget for **TV, Radio, and Newspaper**.")

    # User input sliders
    tv = st.slider("TV Advertising Budget (in $1000s)", 0, 300, 150)
    radio = st.slider("Radio Advertising Budget (in $1000s)", 0, 50, 25)
    newspaper = st.slider("Newspaper Advertising Budget (in $1000s)", 0, 100, 20)

    input_data = pd.DataFrame({"TV": [tv], "Radio": [radio], "Newspaper": [newspaper]})

    if st.button("Predict Sales"):
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Sales: **{prediction:.2f} units**")

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
