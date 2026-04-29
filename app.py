import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

st.title("🔋 EV Battery A/B Testing Dashboard")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df['Energy_Wh'] = df['Voltage_V'] * df['Current_A'] * (df['Time_s'] / 3600)

    st.write(df.head())

    fig = plt.figure()
    plt.plot(df['Cycle'], df['Energy_Wh'])
    st.pyplot(fig)

    A = df[df['Temperature_C'] < 25]['Energy_Wh']
    B = df[df['Temperature_C'] >= 25]['Energy_Wh']

    stat, p = ttest_ind(A, B)
    st.write("P-value:", p)
