import streamlit as st
import pandas as pd
import folium
import joblib

from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Landslide Risk System",
    page_icon="⛰️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⛰️ Landslide Early Risk System")
st.write(
    "Integrated platform for landslide data analysis, "
    "machine learning prediction and GIS risk mapping."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Navigation")

module = st.sidebar.radio(
    "Select Module",
    [
        "🏔️ Risk Prediction",
        "📁 CSV Data",
        "🤖 ML Model",
        "🗺️ GIS Risk Map",
        "ℹ️ System Information"
    ]
)


# ============================================================
# SAMPLE LOCATION DATA
# ============================================================

locations = {
    "Location A": {
        "rainfall": 45,
        "slope": 18,
        "latitude": 21.1702,
        "longitude": 72.8311
    },
    "Location B": {
        "rainfall": 72,
        "slope": 31,
        "latitude": 21.1800,
        "longitude": 72.8400
    },
    "Location C": {
        "rainfall": 90,
        "slope": 38,
        "latitude": 21.1600,
        "longitude": 72.8200
    },
    "Location D": {
        "rainfall": 30,
        "slope": 12,
        "latitude": 21.1900,
        "longitude": 72.8500
    },
    "Location E": {
        "rainfall": 110,
        "slope": 42,
        "latitude": 21.1500,
        "longitude": 72.8100
    }
}


# ============================================================
# MODULE 1 - RISK PREDICTION
# ============================================================

if module == "🏔️ Risk Prediction":

    st.header("🏔️ Landslide Risk Prediction")

    location = st.selectbox(
        "Select Location",
        list(locations.keys())
    )

    date = st.date_input("Select Date")

    threshold = st.slider(
        "Risk Threshold",
        0,
        100,
        50
    )

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

    st.subheader("Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🌧️ Rainfall",
            f"{rainfall} mm"
        )

    with col2:
        st.metric(
            "⛰️ Slope",
            f"{slope}°"
        )

    with col3:
        st.metric(
            "⚠️ Risk Level",
            risk_level
        )

    st.progress(
        min(int(risk_score), 100),
        text=f"Risk Score: {int(risk_score)}/100"
    )

    if st.button(
        "🔍 Predict Risk",
        use_container_width=True
    ):
        st.success(
            f"Risk assessment completed for {location}."
        )

    st.subheader("Why is this location risky?")

    if risk_level == "HIGH":
        st.warning(
            "The location shows higher landslide risk "
            "because of high rainfall and steep terrain."
        )

    elif risk_level == "MEDIUM":
        st.info(
            "The location shows moderate landslide risk. "
            "Rainfall and terrain conditions should be monitored."
        )

    else:
        st.success(
            "The location currently shows lower landslide risk."
        )


# ============================================================
# MODULE 2 - CSV DATA
# ============================================================

elif module == "📁 CSV Data":

    st.header("📁 Landslide CSV Data")

    st.write(
        "Upload a CSV dataset containing rainfall, terrain, "
        "location or landslide-related information."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.success(
                f"Dataset loaded successfully: "
                f"{df.shape[0]} rows × {df.shape[1]} columns"
            )

            st.subheader("📋 Dataset Preview")

            st.dataframe(
                df,
                use_container_width=True
            )

            st.subheader("📊 Dataset Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Rows",
                    df.shape[0]
                )

            with col2:
                st.metric(
                    "Columns",
                    df.shape[1]
                )

            with col3:
                st.metric(
                    "Missing Values",
                    int(df.isna().sum().sum())
                )

            st.subheader("🔎 Column Information")

            info_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": [
                    str(dtype)
                    for dtype in df.dtypes
                ],
                "Missing Values": [
                    int(df[col].isna().sum())
                    for col in df.columns
                ]
            })

            st.dataframe(
                info_df,
                use_container_width=True
            )

            st.session_state["dataset"] = df

        except Exception as e:

            st.error(
                f"Could not read CSV file: {e}"
            )

    else:

        st.info(
            "Upload a CSV file to start data analysis."
        )


# ============================================================
# MODULE 3 - MACHINE LEARNING
# ============================================================

elif module == "🤖 ML Model":

    st.header("🤖 Machine Learning Risk Prediction")

    st.write(
        "Train a Random Forest classification model "
        "using an uploaded landslide dataset."
    )

    if "dataset" not in st.session_state:

        st.warning(
            "First upload a CSV dataset from the "
            "'📁 CSV Data' module."
        )

    else:

        df = st.session_state["dataset"].copy()

        st.subheader("Dataset Available for ML")

        st.write(
            f"Dataset size: {df.shape[0]} rows × "
            f"{df.shape[1]} columns"
        )

        target_column = st.selectbox(
            "Select Target / Risk Column",
            df.columns
        )

        numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        if target_column in numeric_columns:

            numeric_features = [
                col for col in numeric_columns
                if col != target_column
            ]

        else:

            numeric_features = numeric_columns

        if len(numeric_features) == 0:

            st.error(
                "No numeric feature columns found."
            )

        else:

            selected_features = st.multiselect(
                "Select ML Features",
                numeric_features,
                default=numeric_features
            )

            if st.button(
                "🚀 Train ML Model",
                use_container_width=True
            ):

                if len(selected_features) == 0:

                    st.error(
                        "Select at least one feature."
                    )

                else:

                    model_df = df[
                        selected_features + [target_column]
                    ].dropna()

                    X = model_df[selected_features]
                    y = model_df[target_column]

                    if y.nunique() < 2:

                        st.error(
                            "Target column must contain "
                            "at least two classes."
                        )

                    else:

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42,
                            stratify=y
                        )

                        model = RandomForestClassifier(
                            n_estimators=100,
                            random_state=42
                        )

                        model.fit(
                            X_train,
                            y_train
                        )

                        predictions = model.predict(
                            X_test
                        )

                        accuracy = accuracy_score(
                            y_test,
                            predictions
                        )

                        st.success(
                            "ML model trained successfully!"
                        )

                        st.metric(
                            "Model Accuracy",
                            f"{accuracy * 100:.2f}%"
                        )

                        st.session_state["ml_model"] = model
                        st.session_state[
                            "ml_features"
                        ] = selected_features

                        model_bytes = joblib.dumps(model)

                        st.download_button(
                            "💾 Download Trained Model",
                            data=model_bytes,
                            file_name="landslide_model.pkl",
                            mime="application/octet-stream"
                        )


# ============================================================
# MODULE 4 - GIS MAP
# ============================================================

elif module == "🗺️ GIS Risk Map":

    st.header("🗺️ GIS Landslide Risk Map")

    st.write(
        "Interactive map showing geographic landslide "
        "risk locations."
    )

    # First try uploaded CSV data
    if "dataset" in st.session_state:

        df = st.session_state["dataset"].copy()

        lat_candidates = [
            col for col in df.columns
            if col.lower() in [
                "latitude",
                "lat"
            ]
        ]

        lon_candidates = [
            col for col in df.columns
            if col.lower() in [
                "longitude",
                "lon",
                "lng"
            ]
        ]

        if lat_candidates and lon_candidates:

            lat_col = lat_candidates[0]
            lon_col = lon_candidates[0]

            st.success(
                f"GIS coordinates detected: "
                f"{lat_col}, {lon_col}"
            )

            center_lat = df[lat_col].mean()
            center_lon = df[lon_col].mean()

            m = folium.Map(
                location=[
                    center_lat,
                    center_lon
                ],
                zoom_start=10
            )

            for _, row in df.iterrows():

                try:

                    lat = float(row[lat_col])
                    lon = float(row[lon_col])

                    popup_text = "<br>".join(
                        [
                            f"<b>{col}</b>: {row[col]}"
                            for col in df.columns
                            if pd.notna(row[col])
                        ]
                    )

                    folium.Marker(
                        location=[lat, lon],
                        popup=popup_text,
                        tooltip="Landslide Location"
                    ).add_to(m)

                except (ValueError, TypeError):
                    continue

            st_folium(
                m,
                width=1100,
                height=600
            )

        else:

            st.info(
                "CSV uploaded, but Latitude/Longitude "
                "columns were not detected."
            )

    else:

        # Default demonstration map
        m = folium.Map(
            location=[21.1702, 72.8311],
            zoom_start=11
        )

        for name, data in locations.items():

            folium.Marker(
                location=[
                    data["latitude"],
                    data["longitude"]
                ],
                popup=(
                    f"<b>{name}</b><br>"
                    f"Rainfall: {data['rainfall']} mm<br>"
                    f"Slope: {data['slope']}°"
                ),
                tooltip=name
            ).add_to(m)

        st_folium(
            m,
            width=1100,
            height=600
        )


# ============================================================
# MODULE 5 - SYSTEM INFORMATION
# ============================================================

elif module == "ℹ️ System Information":

    st.header("ℹ️ System Information")

    st.write(
        """
        ### ⛰️ Landslide Risk System

        This system is designed as an integrated platform
        for landslide risk assessment.

        **Main components:**

        📁 **CSV Data**
        - Upload real-world datasets
        - Preview data
        - Check missing values
        - View dataset statistics

        🤖 **Machine Learning**
        - Select features
        - Select target/risk column
        - Train Random Forest model
        - Evaluate model accuracy
        - Download trained model

        🗺️ **GIS**
        - Latitude and longitude based mapping
        - Interactive geographic visualization
        - Location-wise data inspection

        🧠 **Jupyter Notebook**

        The ML training workflow can also be maintained
        in a Jupyter Notebook for experimentation,
        preprocessing, visualization and model development.

        The trained model can then be connected to this
        Streamlit dashboard.

        ⚠️ This is currently a prototype and should not
        be treated as an operational landslide warning system.
        """
    )