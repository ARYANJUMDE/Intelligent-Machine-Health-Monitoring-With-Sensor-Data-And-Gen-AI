import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("my_model.pkl")
scaler = joblib.load("my_scaler.pkl")

st.title("⚙️ Predictive Maintenance App")

st.write("Upload your dataset (CSV) and get predictions for machine failures.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data (First 5 Rows)")
    st.write(data.head())

    drop_cols = ["Product ID", "Type", "Failure Type", "UDI", "Target"]
    for col in drop_cols:
        if col in data.columns:
            data = data.drop(columns=col)

    scaled_data = scaler.transform(data)

    predictions = model.predict(scaled_data)

    result = pd.DataFrame(predictions, columns=["Prediction"])
    result["Prediction"] = result["Prediction"].map({0: "No Failure", 1: "Failure"})

    output = pd.concat([data, result], axis=1)

    st.subheader("✅ Predictions")
    st.write(output.head(10))

    # 📊 --- Visualizations ---
    st.subheader("📊 Visualizations")

    # 1. Failure Distribution (Bar Chart)
    st.write("### Failure Distribution")
    fig1, ax1 = plt.subplots()
    output["Prediction"].value_counts().plot(kind="bar", ax=ax1, color=["green", "red"])
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # 2. Failure Distribution (Pie Chart)
    st.write("### Failure Distribution (Pie Chart)")
    fig2, ax2 = plt.subplots()
    output["Prediction"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", colors=["green", "red"], ax=ax2
    )
    ax2.set_ylabel("")
    st.pyplot(fig2)

    # 3. Heatmap of correlations
    st.write("### 🔥 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# 4. Scatter Plot (user selectable features)
    st.write("### 📍 Scatter Plot")

    numeric_cols = list(data.columns)
    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Select X-axis feature", numeric_cols, index=0)
        y_axis = st.selectbox("Select Y-axis feature", numeric_cols, index=1)

        fig4, ax4 = plt.subplots()
        sns.scatterplot(
            data=output,
            x=x_axis,
            y=y_axis,
            hue="Prediction",
            palette={"No Failure": "green", "Failure": "red"},
            ax=ax4,
            alpha=0.7
        )
        st.pyplot(fig4)
    else:
        st.warning("Not enough numeric columns for scatter plot.")
