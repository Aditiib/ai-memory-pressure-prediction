import streamlit as st
import psutil

st.set_page_config(
    page_title="AI Memory Pressure Predictor",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Based Memory Pressure Predictor")

st.write(
    "Predicting memory pressure before swapping begins."
)

st.divider()

# Get real memory information
memory = psutil.virtual_memory()

ram_usage = memory.percent
available_ram = memory.available / (1024 ** 3)

# Display memory information
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("RAM Usage", f"{ram_usage:.1f}%")

with col2:
    st.metric("Available RAM", f"{available_ram:.2f} GB")

with col3:
    swap = psutil.swap_memory()
    st.metric("Swap Usage", f"{swap.percent:.1f}%")

with col4:
    st.metric("Memory PSI", "Not connected yet")

st.divider()

st.subheader("AI Prediction")

st.info(
    "🔄 AI prediction model will be connected after training."
)

st.write("Currently monitoring system memory.")

st.progress(0)

st.write("Model status: Not trained yet")