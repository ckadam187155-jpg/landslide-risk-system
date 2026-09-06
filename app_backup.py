from io import BytesIO
from datetime import datetime

import folium
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from streamlit_folium import st_folium

st.set_page_config(page_title="Landslide Risk Intelligence", page_icon="⛰️", layout="wide")

LOCATIONS = {
    "Location A": {"rainfall": 45, "slope": 18, "latitude": 21.1702, "longitude": 72.8311},
    "Location B": {"rainfall": 72, "slope": 31, "latitude": 21.1800, "longitude": 72.8400},
    "Location C": {"rainfall": 90, "slope": 38, "latitude": 21.1600, "longitude": 72.8200},
    "Location D": {"rainfall": 30, "slope": 12, "latitude": 21.1900, "longitude": 72.8500},
    "Location E": {"rainfall": 110, "slope": 42, "latitude": 21.1500, "longitude": 72.8100},
}
RISK_COLORS = {"HIGH": "#c2413d", "MEDIUM": "#c47a18", "LOW": "#21845b"}


def apply_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root{--forest:#123c2b;--green:#21845b;--mint:#e7f1e9;--canvas:#f4f5ef;--ink:#18231e;--muted:#52625a;--line:#d7e0d7;--amber:#c47a18;--red:#c2413d}
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)} .stApp{background:var(--canvas)}
    [data-testid="stSidebar"]{background:var(--forest)} [data-testid="stSidebar"]>div:first-child{padding:1.5rem 1.15rem} [data-testid="stSidebar"] *{color:#f1f7f1} [data-testid="stSidebar"] hr{border-color:#ffffff33}
    [data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:8px;padding:.45rem .55rem;margin:.08rem 0} [data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#ffffff1a}
    h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif;color:var(--forest);letter-spacing:0} h1{font-size:2.2rem!important;line-height:1.1!important} .block-container{max-width:1560px;padding:1.8rem 3rem 2.5rem}
    .brand{padding-bottom:1.2rem;border-bottom:1px solid #ffffff33;margin-bottom:1.1rem}.brand-name{font-family:'Space Grotesk';font-size:1.15rem;font-weight:700;line-height:1.05;letter-spacing:.03em}.brand-name span{color:#9ed4a8}.brand-copy{color:#c3d9c8!important;font-size:.74rem;margin-top:.55rem;line-height:1.45}.side-status{border-top:1px solid #ffffff33;margin-top:1.2rem;padding-top:1rem;font-size:.78rem}.side-status div{display:flex;justify-content:space-between;padding:.25rem 0}.status-dot{color:#9ed4a8!important}.prototype{color:#b7ccb9!important;font-size:.7rem;margin-top:1.25rem}
    .page-header{display:flex;justify-content:space-between;gap:1.5rem;align-items:flex-start;margin-bottom:1.7rem;padding-bottom:1.15rem;border-bottom:1px solid var(--line)}.eyebrow{color:var(--green);font-size:.73rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;margin-bottom:.45rem}.page-header h1{margin:0 0 .45rem}.subtitle{color:var(--muted);font-size:1rem;margin:0}.online{color:var(--green);background:#e4f1e7;border:1px solid #b8d8c0;border-radius:999px;padding:.48rem .8rem;font-size:.78rem;font-weight:700;white-space:nowrap}
    .card{background:#fffefa;border:1px solid var(--line);border-radius:12px;padding:1.25rem 1.35rem;box-shadow:0 5px 18px #123c2b0f;margin-bottom:1.15rem}.soft-card{background:var(--mint);border:1px solid #cbdccc;border-radius:12px;padding:1.15rem 1.25rem;margin-bottom:1rem}.card-title{color:var(--forest);font-family:'Space Grotesk';font-size:1.08rem;font-weight:700;margin-bottom:.2rem}.card-caption{color:var(--muted);font-size:.84rem;line-height:1.45;margin-bottom:1rem}
    .kpi{background:#fffefa;border:1px solid var(--line);border-top:4px solid var(--green);border-radius:10px;padding:1rem 1.05rem;min-height:132px;box-shadow:0 4px 14px #123c2b0d}.kpi-label{color:var(--muted);font-size:.79rem;font-weight:600}.kpi-value{color:var(--forest);font-family:'Space Grotesk';font-size:1.65rem;font-weight:700;margin:.45rem 0 .2rem}.kpi-note{color:var(--muted);font-size:.76rem}.risk-high{color:var(--red)!important}.risk-medium{color:var(--amber)!important}.risk-low{color:var(--green)!important}.badge{display:inline-block;border-radius:999px;padding:.3rem .62rem;font-size:.72rem;font-weight:700;letter-spacing:.04em}.badge-high{background:#fae7e5;color:#a9322f}.badge-medium{background:#fff0d5;color:#955b0d}.badge-low{background:#dff0e4;color:#176640}.factor{border-left:4px solid var(--green);background:#f8faf6;border-radius:7px;padding:.8rem .9rem;min-height:96px}.factor strong{color:var(--forest);display:block;margin:.3rem 0}.factor small{color:var(--muted)}.callout{border-left:4px solid var(--green);background:#e7f1e9;padding:.9rem 1rem;border-radius:0 8px 8px 0;color:var(--forest)}
    .footer{border-top:1px solid var(--line);color:var(--muted);font-size:.76rem;margin-top:2rem;padding-top:1rem;display:flex;justify-content:space-between;gap:1rem} div[data-testid="stButton"]>button{background:var(--forest);color:white;border:1px solid var(--forest);border-radius:8px;font-weight:600} div[data-testid="stButton"]>button:hover{background:#236b4c;color:white;border-color:#236b4c} [data-testid="stMetric"]{background:#fffefa;border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem} [data-testid="stMetricLabel"]{color:var(--muted)} [data-testid="stMetricValue"]{color:var(--forest)}/* ===== TEXT VISIBILITY FIX ===== */

h1 {
    color: #c62828 !important;
    font-weight: 800 !important;
}

h2, h3, h4 {
    color: #111111 !important;
    font-weight: 700 !important;
}

p, span, label, li {
    color: #111111 !important;
}

.stMarkdown,
.stMarkdown p {
    color: #111111 !important;
}

[data-testid="stMetric"] {
    color: #111111 !important;
}

[data-testid="stWidgetLabel"] {
    color: #111111 !important;
}

.stButton button {
    color: white !important;
}/* ===== SIDEBAR TEXT FIX ===== */

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: white !important;
}

/* Sidebar selected item */
[data-testid="stSidebar"] button {
    color: white !important;
}

/* Keep status indicators readable */
[data-testid="stSidebar"] .status {
    color: white !important;
}/* ===== SIDEBAR TEXT FIX ===== */

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: white !important;
}

/* Sidebar selected item */
[data-testid="stSidebar"] button {
    color: white !important;
}

/* Keep status indicators readable */
[data-testid="stSidebar"] .status {
    color: white !important;
}s
    </style>
    """, unsafe_allow_html=True)


def header(title, description):
    st.markdown(f'<div class="page-header"><div><div class="eyebrow">Environmental intelligence platform</div><h1>{title}</h1><p class="subtitle">{description}</p></div><div class="online">● System Online</div></div>', unsafe_allow_html=True)


def title(text, caption=None):
    caption_html = f'<div class="card-caption">{caption}</div>' if caption else ""
    st.markdown(f'<div class="card-title">{text}</div>{caption_html}', unsafe_allow_html=True)


def assess(rainfall, slope, threshold):
    score = rainfall * 0.6 + slope * 1.2
    level = "HIGH" if score >= threshold else "MEDIUM" if score >= threshold * 0.6 else "LOW"
    return score, level


def badge(level):
    return f'<span class="badge badge-{level.lower()}">{level} RISK</span>'


def chart(figure, height=300):
    figure.update_layout(height=height, margin=dict(l=10, r=10, t=35, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="#18231e", size=12), title_font=dict(size=15, color="#123c2b"))
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def current():
    location = st.session_state.get("location", "Location C")
    threshold = st.session_state.get("threshold", 50)
    data = LOCATIONS[location]
    score, level = assess(data["rainfall"], data["slope"], threshold)
    return location, data, score, level, threshold


def map_view(data, center=(21.1702, 72.8311), zoom=11, height=510):
    fmap = folium.Map(location=list(center), zoom_start=zoom)
    for name, item in data.items():
        score, level = assess(item["rainfall"], item["slope"], 50)
        folium.CircleMarker(location=[item["latitude"], item["longitude"]], radius=8, color=RISK_COLORS[level], fill=True, fill_color=RISK_COLORS[level], fill_opacity=.8, popup=f"<b>{name}</b><br>Risk: {int(score)}/100<br>Rainfall: {item['rainfall']} mm<br>Slope: {item['slope']}°", tooltip=f"{name} · {level}").add_to(fmap)
    st_folium(fmap, width=None, height=height)


def overview():
    header("Landslide Risk Intelligence", "A concise view of current conditions, risk signals, and locations requiring attention.")
    location, data, score, level, _ = current()
    st.markdown(f'<div class="callout">Current assessment for <strong>{location}</strong>: {badge(level)} &nbsp; Monitor environmental conditions and follow official local guidance.</div>', unsafe_allow_html=True)
    st.markdown("### Current indicators")
    values = [("⚠️", "Overall Risk Level", level, "Current classification", level.lower()), ("◈", "Risk Score", f"{int(score)} / 100", "Rainfall and slope model", level.lower()), ("☔", "Rainfall", f"{data['rainfall']} mm", "Selected location input", ""), ("⌁", "Slope", f"{data['slope']}°", "Terrain steepness input", "")]
    for col, (icon, label, value, note, color) in zip(st.columns(4), values):
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-label">{icon} &nbsp; {label}</div><div class="kpi-value risk-{color}">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)
    st.markdown("### Risk signals")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title("Risk score gauge", "Calculated from the existing rainfall and slope formula.")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=min(score, 100), number={"font": {"color": RISK_COLORS[level], "size": 38}}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[level]}, "steps": [{"range": [0, 30], "color": "#dff0e4"}, {"range": [30, 50], "color": "#fff0d5"}, {"range": [50, 100], "color": "#fae7e5"}]}, title={"text": f"{level} RISK", "font": {"color": RISK_COLORS[level], "size": 16}}))
        chart(gauge, 265)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title("Rainfall vs risk score", "The five demonstration locations using the preserved calculation.")
        figure = go.Figure(go.Scatter(x=[v["rainfall"] for v in LOCATIONS.values()], y=[assess(v["rainfall"], v["slope"], 50)[0] for v in LOCATIONS.values()], mode="markers+text", text=list(LOCATIONS), textposition="top center", marker={"size": 11, "color": "#21845b"}, hovertemplate="Rainfall: %{x} mm<br>Risk: %{y:.0f}/100<extra></extra>"))
        figure.update_layout(xaxis_title="Rainfall (mm)", yaxis_title="Risk score", yaxis={"range": [0, 100]})
        chart(figure, 265)
        st.markdown('</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title("Slope vs risk score")
        figure = go.Figure(go.Scatter(x=[v["slope"] for v in LOCATIONS.values()], y=[assess(v["rainfall"], v["slope"], 50)[0] for v in LOCATIONS.values()], mode="markers+text", text=list(LOCATIONS), textposition="top center", marker={"size": 11, "color": "#c47a18"}))
        figure.update_layout(xaxis_title="Slope (°)", yaxis_title="Risk score", yaxis={"range": [0, 100]})
        chart(figure, 280)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title("Risk distribution")
        counts = {key: 0 for key in ("HIGH", "MEDIUM", "LOW")}
        for item in LOCATIONS.values():
            counts[assess(item["rainfall"], item["slope"], 50)[1]] += 1
        pie = go.Figure(go.Pie(labels=list(counts), values=list(counts.values()), hole=.62, marker_colors=[RISK_COLORS[key] for key in counts]))
        pie.update_layout(showlegend=True, legend=dict(orientation="h", y=-.05))
        chart(pie, 280)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("### Priority locations")
    rows = [{"Location": name, "Risk Score": int(assess(v["rainfall"], v["slope"], 50)[0]), "Risk Level": assess(v["rainfall"], v["slope"], 50)[1], "Rainfall": f"{v['rainfall']} mm"} for name, v in LOCATIONS.items()]
    st.dataframe(pd.DataFrame(sorted(rows, key=lambda row: row["Risk Score"], reverse=True)), hide_index=True, use_container_width=True)


def prediction():
    header("Risk Prediction", "Assess a demonstration location with the existing rainfall, slope, and threshold logic.")
    controls, result = st.columns([1, 1.5])
    with controls:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title("Assessment inputs", "These are the existing application inputs.")
        location = st.selectbox("Select location", list(LOCATIONS), index=list(LOCATIONS).index(st.session_state.get("location", "Location C")))
        date = st.date_input("Assessment date")
        threshold = st.slider("Risk threshold", 0, 100, st.session_state.get("threshold", 50))
        st.session_state.update({"location": location, "threshold": threshold})
        st.markdown('</div>', unsafe_allow_html=True)
    location, data, score, level, threshold = current()
    with result:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title("Prediction result", f"Assessment for {location} on {date.strftime('%d %b %Y')}")
        st.markdown(f'{badge(level)}<h2 class="risk-{level.lower()}">{level}</h2><p>Risk score: <strong>{int(score)} / 100</strong></p>', unsafe_allow_html=True)
        st.progress(min(int(score), 100))
        st.write("Higher rainfall and steep terrain are increasing the calculated risk." if level == "HIGH" else "Rainfall and terrain conditions should continue to be monitored." if level == "MEDIUM" else "The selected conditions currently indicate lower calculated risk.")
        if st.button("Run risk assessment", use_container_width=True):
            st.success(f"Risk assessment completed for {location}.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("### Key factors")
    factors = [("☔", "Rainfall", f"{data['rainfall']} mm", "Precipitation input"), ("⌁", "Slope", f"{data['slope']}°", "Terrain input"), ("◉", "Location", location, "Assessment area"), ("⊙", "Threshold", f"{threshold}/100", "Classification setting")]
    for col, (icon, label, value, note) in zip(st.columns(4), factors):
        with col:
            st.markdown(f'<div class="factor"><span>{icon}</span><strong>{label}</strong><small><b>{value}</b><br>{note}</small></div>', unsafe_allow_html=True)
    st.markdown("### Recommended actions")
    actions_col, map_col = st.columns([1, 1.5])
    with actions_col:
        st.markdown('<div class="soft-card"><strong>Decision-support recommendations</strong>', unsafe_allow_html=True)
        actions = ["Follow official disaster-management advisories.", "Avoid vulnerable areas during severe conditions.", "Continue monitoring rainfall and slope conditions."] if level == "HIGH" else ["Monitor rainfall and terrain conditions.", "Review local conditions regularly.", "Prepare emergency communication."]
        for action in actions:
            st.markdown(f"- {action}")
        st.caption("These are not official warnings.")
        st.markdown('</div>', unsafe_allow_html=True)
    with map_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title("Location context", "Existing demonstration locations.")
        map_view(LOCATIONS, (data["latitude"], data["longitude"]))
        st.markdown('</div>', unsafe_allow_html=True)


def csv_data():
    header("CSV Data", "Load, inspect, and prepare environmental data for analysis and model training.")
    st.markdown('<div class="soft-card"><strong>Workflow</strong> &nbsp; 1. Upload CSV &nbsp; → &nbsp; 2. Preview dataset &nbsp; → &nbsp; 3. Analyze data &nbsp; → &nbsp; 4. Visualize risk</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to start data analysis.")
        return
    try:
        df = pd.read_csv(uploaded)
        st.session_state["dataset"] = df
        st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        for col, label, value in zip(st.columns(3), ["Rows", "Columns", "Missing values"], [df.shape[0], df.shape[1], int(df.isna().sum().sum())]):
            with col:
                st.metric(label, value)
        with st.expander("Preview dataset", expanded=True):
            st.dataframe(df, use_container_width=True, hide_index=True)
        with st.expander("Column information"):
            st.dataframe(pd.DataFrame({"Column": df.columns, "Data type": [str(x) for x in df.dtypes], "Missing values": [int(df[x].isna().sum()) for x in df.columns]}), use_container_width=True)
        with st.expander("Basic statistics"):
            st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    except Exception as error:
        st.error(f"Could not read CSV file: {error}")


def ml_model():
    header("ML Model / Training", "Train and inspect the existing Random Forest classifier using uploaded data.")
    metric = st.session_state.get("ml_accuracy")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    title("Model status")
    st.markdown("**Random Forest Classifier**")
    st.markdown(f'<span class="badge badge-low">{"ACTIVE" if "ml_model" in st.session_state else "READY TO TRAIN"}</span>', unsafe_allow_html=True)
    st.caption("Metrics are shown only after a real training run.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    title("Performance", "No metrics are invented when a model has not been trained.")
    metric_values = [f"{metric * 100:.2f}%" if metric is not None else "Not available", "Not available", "Not available", "Not available"]
    for col, label, value in zip(st.columns(4), ["Accuracy", "Precision", "Recall", "F1 score"], metric_values):
        with col:
            st.metric(label, value)
    st.markdown('</div>', unsafe_allow_html=True)
    if "dataset" not in st.session_state:
        st.warning("Upload a CSV dataset from the CSV Data page before training.")
        return
    df = st.session_state["dataset"].copy()
    target = st.selectbox("Select target / risk column", df.columns)
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    features = [col for col in numeric if col != target] if target in numeric else numeric
    if not features:
        st.error("No numeric feature columns found.")
        return
    selected = st.multiselect("Select ML features", features, default=features)
    if st.button("Train Random Forest model", use_container_width=True):
        if not selected:
            st.error("Select at least one feature.")
            return
        model_df = df[selected + [target]].dropna()
        X, y = model_df[selected], model_df[target]
        if y.nunique() < 2:
            st.error("Target column must contain at least two classes.")
            return
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
        except ValueError as error:
            st.error(f"Unable to create a stratified test split: {error}")
            return
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.session_state.update({"ml_model": model, "ml_features": selected, "ml_accuracy": accuracy})
        st.success("ML model trained successfully.")
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        st.download_button("Download trained model", data=buffer.getvalue(), file_name="landslide_model.pkl", mime="application/octet-stream")
    if "ml_model" in st.session_state:
        figure = go.Figure(go.Bar(x=st.session_state["ml_model"].feature_importances_, y=st.session_state["ml_features"], orientation="h", marker_color="#21845b"))
        figure.update_layout(title="Feature importance", xaxis_title="Importance", yaxis_title="Feature")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        chart(figure, 330)
        st.markdown('</div>', unsafe_allow_html=True)


def gis_map():
    header("GIS Risk Map", "Inspect risk locations and preserve coordinate-aware visualization from the existing workflow.")
    if "dataset" in st.session_state:
        df = st.session_state["dataset"].copy()
        lat = next((c for c in df.columns if c.lower() in ["latitude", "lat"]), None)
        lon = next((c for c in df.columns if c.lower() in ["longitude", "lon", "lng"]), None)
        if lat and lon:
            st.success(f"GIS coordinates detected: {lat}, {lon}")
            fmap = folium.Map(location=[df[lat].mean(), df[lon].mean()], zoom_start=10)
            for _, row in df.iterrows():
                try:
                    folium.Marker(location=[float(row[lat]), float(row[lon])], popup="<br>".join(f"<b>{col}</b>: {row[col]}" for col in df.columns if pd.notna(row[col])), tooltip="Landslide location").add_to(fmap)
                except (TypeError, ValueError):
                    continue
            st_folium(fmap, width=None, height=600)
        else:
            st.info("CSV uploaded, but Latitude/Longitude columns were not detected. Showing demonstration locations.")
            map_view(LOCATIONS)
    else:
        map_view(LOCATIONS)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    title("Risk legend")
    for col, level in zip(st.columns(3), ["HIGH", "MEDIUM", "LOW"]):
        with col:
            st.markdown(f'{badge(level)}<br><small>Risk locations classified {level.lower()}</small>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def ai_intelligence():
    header("AI Risk Intelligence", "Transparent, model-based explanations built from current environmental inputs.")
    location, data, score, level, threshold = current()
    st.markdown(f'<div class="soft-card"><div class="eyebrow">Current assessment</div><h2>{location} is classified as {level} risk</h2><p>This decision-support view uses the available rainfall and slope conditions. It does not claim an external AI service.</p></div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title("Key contributing factors")
        st.markdown(f"- Rainfall input: **{data['rainfall']} mm**\n- Slope input: **{data['slope']}°**\n- Calculated score: **{int(score)} / 100**\n- Classification threshold: **{threshold}**")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title("Why is the risk at this level?")
        st.write("The score combines rainfall with terrain steepness. Higher rainfall increases saturation pressure, while steeper terrain increases the potential for slope instability. Review field conditions alongside this prototype assessment.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    title("Recommended actions", "Decision-support recommendations only")
    for action in ["Monitor rainfall conditions.", "Avoid unnecessary travel in unstable areas.", "Stay alert to local warnings.", "Follow local authority guidance.", "Prepare emergency supplies."]:
        st.markdown(f"- {action}")
    st.caption("This is not an official emergency warning system.")
    st.markdown('</div>', unsafe_allow_html=True)


def system_info():
    header("System Information", "Technical context, data sources, and operating status for this prototype.")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title("Model information")
        st.markdown(f"**Model type:** Random Forest Classifier  \n**Training status:** {'Completed in this session' if 'ml_model' in st.session_state else 'Not trained'}  \n**Model status:** {'Active' if 'ml_model' in st.session_state else 'Ready'}")
        title("Input factors")
        st.markdown("Rainfall · Slope · Soil, Lithology, Vegetation, and Land Use when supplied in CSV")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        title("System status")
        st.markdown(f"**Application:** Online  \n**Model:** {'Active' if 'ml_model' in st.session_state else 'Ready'}  \n**Data:** {'Loaded' if 'dataset' in st.session_state else 'Demonstration data loaded'}  \n**Version:** 1.0 prototype  \n**Last updated:** {datetime.now().strftime('%d %b %Y')}")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout"><strong>Prototype for educational/hackathon use.</strong><br>This system is not an official emergency warning system. Follow disaster-management authorities and field professionals for real-world decisions.</div>', unsafe_allow_html=True)


apply_style()
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-name">LANDSLIDE RISK<br><span>INTELLIGENCE</span></div><div class="brand-copy">AI-powered landslide monitoring and risk assessment</div></div>', unsafe_allow_html=True)
    module = st.radio("Navigation", ["Overview", "Risk Prediction", "GIS Risk Map", "CSV Data", "ML Model / Training", "AI Risk Intelligence", "System Information"], label_visibility="collapsed")
    st.markdown('<div class="side-status"><div><span>System status</span><span class="status-dot">● Online</span></div><div><span>Model status</span><span class="status-dot">● Active</span></div><div><span>Data status</span><span class="status-dot">● Loaded</span></div></div><div class="prototype">AI/ML Decision Support Prototype</div>', unsafe_allow_html=True)

if module == "Overview":
    overview()
elif module == "Risk Prediction":
    prediction()
elif module == "GIS Risk Map":
    gis_map()
elif module == "CSV Data":
    csv_data()
elif module == "ML Model / Training":
    ml_model()
elif module == "AI Risk Intelligence":
    ai_intelligence()
else:
    system_info()

st.markdown('<div class="footer"><span>Landslide Risk Intelligence · AI/ML Decision Support Prototype</span><span>Educational prototype · Not an official emergency warning system</span></div>', unsafe_allow_html=True)
