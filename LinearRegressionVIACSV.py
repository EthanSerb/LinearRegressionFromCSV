import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


st.title("Linear Regression From CSV")

file = st.file_uploader("Upload your CSV file", type=["csv"])
if file is not None:
    data = pd.read_csv(file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    columns = data.columns.tolist()
    x_col = st.selectbox("Select the feature (X)", columns)
    y_col = st.selectbox("Select the target (Y)", columns)

    if st.button("Train Model"):
        X = data[[x_col]].values
        Y = data[y_col].values

        model = LinearRegression()
        model.fit(X, Y)

        st.write(f"Model Coefficient: {model.coef_[0]}")
        st.write(f"Model Intercept: {model.intercept_}")
        st.write(f"R^2 Score: {model.score(X, Y)}")


        plt.scatter(X, Y, color='blue', label='Data points')
        plt.plot(X, model.predict(X), color='red', label='Regression line')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title('Linear Regression')
        plt.legend()
        st.pyplot(plt)