import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Landslide Risk System",
    page_icon="⛰️",
    layout="wide"
)

# Title
st.title("⛰️ Landslide Early Risk System")
st.write("A prototype dashboard for landslide risk assessment.")

# Sidebar
st.sidebar.header("Settings")

location = st.sidebar.selectbox(
    "Select Location",
    ["Location A", "Location B", "Location C", "Location D", "Location E"]
)

date = st.sidebar.date_input("Select Date")

threshold = st.sidebar.slider(
    "Risk Threshold",
    0,
    100,
    50
)

# Main section
st.header("Risk Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rainfall", "72 mm")

with col2:
    st.metric("Slope", "31°")

with col3:
    st.metric("Risk Level", "HIGH")

# Prediction button
if st.button("🔍 Predict Risk", use_container_width=True):
    st.success(f"Risk assessment completed for {location}.")

    st.subheader("Why is this location risky?")
    st.write(
        "The prototype indicates higher risk because of "
        "high rainfall and steep terrain."
    )

# Map placeholder
st.header("🗺️ Risk Map")

st.info(
    "Map will be integrated in a later stage. "
    "Currently this is a placeholder."
)