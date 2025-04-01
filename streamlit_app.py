import streamlit as st
import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the simulation function from main.py.
from main import simulate_federated_learning

st.set_page_config(page_title="Federated Learning Simulation", layout="wide")

# ------------------------------------------------
# Session State Setup
# ------------------------------------------------
if "global_model" not in st.session_state:
    st.session_state.global_model = None
if "simulation_params" not in st.session_state:
    st.session_state.simulation_params = {
        "NUM_CLIENTS": 5,           # integer
        "NUM_EPOCHS": 15,           # integer
        "BATCH_SIZE": 128,          # integer
        "NUM_GLOBAL_ROUNDS": 1,     # integer
        "GLOBAL_LR": 0.013,         # float
        "DP_learning_rate": 0.005,  # float
        "sensitivity": 0.5,         # float
        "l2_norm_clip": 2.25,       # float
        "noise_multiplier": 0.36,   # float
        "decay_schedule": "linear", # string
        "RESET_GLOBAL_MODEL": False # boolean
    }

# ------------------------------------------------
# Sidebar: Parameter Controls
# ------------------------------------------------
st.sidebar.header("Federated Learning Parameters")

st.session_state.simulation_params["NUM_CLIENTS"] = st.sidebar.number_input(
    "Number of Clients", min_value=1,
    value=st.session_state.simulation_params["NUM_CLIENTS"],
    step=1
)
st.session_state.simulation_params["NUM_EPOCHS"] = st.sidebar.number_input(
    "Local Epochs per Round", min_value=1,
    value=st.session_state.simulation_params["NUM_EPOCHS"],
    step=1
)
st.session_state.simulation_params["BATCH_SIZE"] = st.sidebar.number_input(
    "Batch Size", min_value=16,
    value=st.session_state.simulation_params["BATCH_SIZE"],
    step=1
)
st.session_state.simulation_params["NUM_GLOBAL_ROUNDS"] = st.sidebar.number_input(
    "Global Rounds", min_value=1,
    value=st.session_state.simulation_params["NUM_GLOBAL_ROUNDS"],
    step=1
)

st.session_state.simulation_params["GLOBAL_LR"] = st.sidebar.number_input(
    "Global Learning Rate", min_value=0.0001,
    value=st.session_state.simulation_params["GLOBAL_LR"],
    format="%.4f"
)
st.session_state.simulation_params["DP_learning_rate"] = st.sidebar.number_input(
    "DP Learning Rate", min_value=0.0001,
    value=st.session_state.simulation_params["DP_learning_rate"],
    format="%.4f"
)
st.session_state.simulation_params["sensitivity"] = st.sidebar.number_input(
    "Sensitivity", min_value=0.1,
    value=st.session_state.simulation_params["sensitivity"],
    format="%.4f"
)
st.session_state.simulation_params["l2_norm_clip"] = st.sidebar.number_input(
    "L2 Norm Clipping", min_value=0.1,
    value=st.session_state.simulation_params["l2_norm_clip"],
    format="%.4f"
)
st.session_state.simulation_params["noise_multiplier"] = st.sidebar.number_input(
    "Base Noise Multiplier", min_value=0.01,
    value=st.session_state.simulation_params["noise_multiplier"],
    format="%.4f"
)

st.session_state.simulation_params["decay_schedule"] = st.sidebar.selectbox(
    "Decay Schedule", options=["exponential", "linear"],
    index=0 if st.session_state.simulation_params["decay_schedule"] == "exponential" else 1
)
st.session_state.simulation_params["RESET_GLOBAL_MODEL"] = st.sidebar.checkbox(
    "Reset Global Model Each Round",
    value=st.session_state.simulation_params["RESET_GLOBAL_MODEL"]
)

if st.sidebar.button("Reset Simulation"):
    st.session_state.global_model = None
    st.success("Simulation state reset.")

# ------------------------------------------------
# Main UI Controls
# ------------------------------------------------
st.title("Federated Learning Simulation UI")
run_simulation = st.button("Run Federated Training")

if run_simulation:
    st.info("Starting federated training simulation. This may take a while...")
    global_model = simulate_federated_learning(st.session_state.simulation_params)
    st.session_state.global_model = global_model
    st.success("Federated training simulation completed.")

if st.session_state.global_model is not None:
    st.header("Global Model Evaluation")
    st.write("The global model has been updated over the federated rounds.")
    if os.path.exists("plots/accuracy_privacy.png"):
        st.image("plots/accuracy_privacy.png", caption="Accuracy vs. Privacy (ε)")
    if os.path.exists("plots/roc_curve.png"):
        st.image("plots/roc_curve.png", caption="ROC Curve")
    st.subheader("Global Model Summary")
    st.text(st.session_state.global_model.summary())

if st.button("Re-run Simulation"):
    st.session_state.global_model = None
    st.experimental_rerun()
