import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Landslide Risk System",
    page_icon="⛰️",
    layout="wide"
)

# Location data
locations = {
    "Location A": {"rainfall": 45, "slope": 18},
    "Location B": {"rainfall": 72, "slope": 31},
    "Location C": {"rainfall": 90, "slope": 38},
    "Location D": {"rainfall": 30, "slope": 12},
    "Location E": {"rainfall": 110, "slope": 42}
}

# Title
st.title("⛰️ Landslide Early Risk System")
st.write("A prototype dashboard for landslide risk assessment.")

# Sidebar
st.sidebar.header("⚙️ Settings")

location = st.sidebar.selectbox(
    "Select Location",
    list(locations.keys())
)

date = st.sidebar.date_input("Select Date")

threshold = st.sidebar.slider(
    "Risk Threshold",
    0,
    100,
    50
)

# Get selected location data
rainfall = locations[location]["rainfall"]
slope = locations[location]["slope"]

# Risk calculation
risk_score = (rainfall * 0.6) + (slope * 1.2)

if risk_score >= threshold:
    risk_level = "HIGH"
elif risk_score >= threshold * 0.6:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"

# Main section
st.header("Risk Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌧️ Rainfall", f"{rainfall} mm")

with col2:
    st.metric("⛰️ Slope", f"{slope}°")

with col3:
    st.metric("⚠️ Risk Level", risk_level)

# Risk score
st.progress(
    min(int(risk_score), 100),
    text=f"Risk Score: {int(risk_score)}/100"
)

# Prediction button
if st.button("🔍 Predict Risk", use_container_width=True):

    st.success(
        f"Risk assessment completed for {location}."
    )

    st.subheader("Why is this location risky?")

    if risk_level == "HIGH":
        st.write(
            "The location shows higher landslide risk "
            "because of high rainfall and steep terrain."
        )

    elif risk_level == "MEDIUM":
        st.write(
            "The location shows moderate landslide risk. "
            "Rainfall and terrain conditions should be monitored."
        )

    else:
        st.write(
            "The location currently shows lower landslide risk."
        )

# Risk Map
st.header("🗺️ Risk Map")

st.info(
    "Interactive geographic risk mapping will be integrated "
    "in the next stage."
)

# Project information
st.header("📊 System Information")

st.write(
    "This prototype combines rainfall, slope and a user-defined "
    "risk threshold to provide an initial landslide risk assessment."
)