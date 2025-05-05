import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
from time import sleep
from glob import glob

# Load model class
class HeartDiseaseModel(torch.nn.Module):
    def __init__(self, input_size):
        super(HeartDiseaseModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Simulate client training (just load CSVs and pretend to train)
def simulate_client_training(client_id):
    df = pd.read_csv(f'datasets/client_{client_id}_data.csv')
    acc = np.random.uniform(80, 88)
    sleep(1)  # simulate training time
    return round(acc, 2)

# Simulate federated aggregation (placeholder)
def federated_averaging(acc_list):
    return round(np.mean(acc_list), 2)

# Page UI starts here
st.set_page_config(layout="wide")
st.title("🧠 Federated Learning Simulator – Heart Disease Detection")

if "round" not in st.session_state:
    st.session_state.round = 0
    st.session_state.history = []

st.write("### 🧪 Client Training Status")

cols = st.columns(3)
clients_ready = True

# Simulate client training
if st.button("▶️ Run Federated Round"):
    st.session_state.round += 1
    client_accs = []
    
    for i in range(3):
        with cols[i]:
            st.markdown(f"**Client {i+1}**")
            with st.spinner("Training..."):
                acc = simulate_client_training(i+1)
                client_accs.append(acc)
                st.success(f"Local Accuracy: {acc}%")

    global_acc = federated_averaging(client_accs)
    st.session_state.history.append(global_acc)

    st.success(f"✅ Global Model Accuracy after Round {st.session_state.round}: {global_acc}%")

# Chart area
st.write("---")
st.write("### 📈 Global Accuracy Over Rounds")

if st.session_state.history:
    fig, ax = plt.subplots()
    ax.plot(range(1, len(st.session_state.history)+1), st.session_state.history, marker='o')
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Global Model Accuracy")
    ax.grid(True)
    st.pyplot(fig)
else:
    st.info("Run at least one round to see the graph.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
