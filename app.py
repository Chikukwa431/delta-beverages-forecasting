import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import os

st.set_page_config(
    page_title="Delta Beverages | Castle Lager Demand Forecasting",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_bg_base64():
    img = Image.new("RGB", (1600, 900), color=(15, 25, 45))
    draw = ImageDraw.Draw(img)
    for i in range(900):
        r = int(15 + (i / 900) * 20)
        g = int(25 + (i / 900) * 18)
        b = int(45 + (i / 900) * 10)
        draw.line([(0, i), (1600, i)], fill=(r, g, b))
    draw.ellipse([1100, -150, 1750, 500], fill=(212, 160, 23))
    draw.ellipse([1180, -70, 1670, 420], fill=(24, 36, 58))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

bg_b64 = create_bg_base64()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login_page():
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
    }}
    .login-card {{
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid #D4A017;
        border-radius: 14px;
        padding: 36px;
        max-width: 440px;
        margin: 80px auto 20px auto;
        box-shadow: 0 12px 32px rgba(0,0,0,0.6);
        color: #F8FAFC;
        text-align: center;
    }}
    .login-title {{
        color: #FACC15;
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 4px;
    }}
    .login-sub {{
        color: #CBD5E1;
        font-size: 14px;
        margin-bottom: 20px;
    }}
    </style>
    <div class="login-card">
        <div class="login-title">DELTA BEVERAGES</div>
        <div class="login-sub">Castle Lager Econometric & ML Demand Forecasting Portal</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("Sign In to Dashboard", use_container_width=True)
            if submit:
                if username == "TKapfuma" and password == "Kapfuma123":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please check your username and password.")

@st.cache_data
def load_or_generate_data(uploaded_file=None):
    default_path = r"C:\Users\hp\Desktop\Castle_Lager_Records_Delta_Beverages.xlsx"
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, sheet_name="Daily_Demand_Data_5000")
        except Exception:
            df = pd.read_excel(uploaded_file)
    elif os.path.exists(default_path):
        try:
            df = pd.read_excel(default_path, sheet_name="Daily_Demand_Data_5000")
        except Exception:
            df = pd.read_excel(default_path)

    if df is None:
        np.random.seed(42)
        n = 5000
        dates = pd.date_range(end="2026-10-31", periods=n, freq="D")
        trend = np.linspace(1800, 2650, n)
        seasonal = 320 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        weekly = np.where(dates.dayofweek >= 4, 240, -60)
        inflation = np.clip(np.cumsum(np.random.normal(0.02, 0.35, n)) + 14.5, 4.0, 48.0)
        temp = 24 + 5 * np.sin(2 * np.pi * (dates.dayofyear - 260) / 365.25) + np.random.normal(0, 1.8, n)
        promo = np.random.binomial(1, 0.14, n)
        demand = trend + seasonal + weekly - 6.5 * inflation + 18 * (temp - 22) + 210 * promo + np.random.normal(0, 95, n)
        df = pd.DataFrame({
            "Date": dates,
            "Demand_HL": np.round(np.maximum(demand, 450), 2),
            "Inflation_Rate": np.round(inflation, 2),
            "Temperature_C": np.round(temp, 1),
            "Promo_Flag": promo
        })

    date_col = [c for c in df.columns if "date" in c.lower()][0] if any("date" in c.lower() for c in df.columns) else df.columns[0]
    target_col = [c for c in df.columns if "demand" in c.lower() or "sales" in c.lower() or "hl" in c.lower()][0] if any(k in c.lower() for c in df.columns for k in ["demand", "sales", "hl"]) else df.columns[1]
    
    df = df.rename(columns={date_col: "Date", target_col: "Demand_HL"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "Inflation_Rate" not in df.columns:
        df["Inflation_Rate"] = 15.0
    if "Temperature_C" not in df.columns:
        df["Temperature_C"] = 24.5
    if "Promo_Flag" not in df.columns:
        df["Promo_Flag"] = 0

    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f"Lag_{lag}"] = df["Demand_HL"].shift(lag)
    df["Rolling_Mean_7"] = df["Demand_HL"].shift(1).rolling(7).mean()
    df["Rolling_Std_7"] = df["Demand_HL"].shift(1).rolling(7).std()
    df["Rolling_Mean_30"] = df["Demand_HL"].shift(1).rolling(30).mean()
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 4).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_resource
def train_models(df):
    feature_cols = [
        "Inflation_Rate", "Temperature_C", "Promo_Flag",
        "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Lag_14", "Lag_21", "Lag_30",
        "Rolling_Mean_7", "Rolling_Std_7", "Rolling_Mean_30",
        "DayOfWeek", "Month", "IsWeekend"
    ]
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()

    X_train, y_train = train_df[feature_cols], train_df["Demand_HL"]
    X_test, y_test = test_df[feature_cols], test_df["Demand_HL"]

    rf = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.06, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)
    pred_xgb = xgb_model.predict(X_test)

    lgb_model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.06, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    pred_lgb = lgb_model.predict(X_test)

    try:
        sarimax_mod = sm.tsa.SARIMAX(
            train_df["Demand_HL"].iloc[-600:],
            exog=train_df[["Inflation_Rate", "Temperature_C", "Promo_Flag"]].iloc[-600:],
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        pred_sarimax = sarimax_mod.forecast(
            steps=len(test_df),
            exog=test_df[["Inflation_Rate", "Temperature_C", "Promo_Flag"]]
        ).values
    except Exception:
        pred_sarimax = 0.55 * test_df["Lag_1"].values + 0.45 * test_df["Rolling_Mean_7"].values

    results = {}
    for name, preds in [("SARIMAX", pred_sarimax), ("Random Forest", pred_rf), ("LightGBM", pred_lgb), ("XGBoost", pred_xgb)]:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        r2 = r2_score(y_test, preds)
        results[name] = {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R2": r2, "Preds": preds}

    test_df["Pred_SARIMAX"] = pred_sarimax
    test_df["Pred_RF"] = pred_rf
    test_df["Pred_LGBM"] = pred_lgb
    test_df["Pred_XGB"] = pred_xgb

    return results, test_df, xgb_model, lgb_model, rf, feature_cols

def dashboard():
    st.sidebar.markdown("### 🍺 Delta Beverages")
    st.sidebar.caption("Castle Lager Forecasting Suite")
    uploaded_file = st.sidebar.file_uploader("Upload Excel Dataset (.xlsx)", type=["xlsx"])
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    df = load_or_generate_data(uploaded_file)
    results, test_df, xgb_model, lgb_model, rf_model, feature_cols = train_models(df)

    best_model = min(results.keys(), key=lambda k: results[k]["RMSE"])

    st.title("Delta Beverages: Castle Lager Demand Forecasting & Econometric Analysis")
    st.caption("Comparative Multi-Model Pipeline: SARIMAX vs. Random Forest, LightGBM & XGBoost")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Daily Demand", f"{df['Demand_HL'].mean():,.1f} HL")
    c2.metric("Best Performing Model", best_model)
    c3.metric(f"{best_model} RMSE", f"{results[best_model]['RMSE']:.2f} HL")
    c4.metric(f"{best_model} R² Score", f"{results[best_model]['R2']:.4f}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Executive Overview",
        "2. Model Benchmark",
        "3. Feature Drivers",
        "4. What-If Simulator",
        "5. Methodology & Export"
    ])

    with tab1:
        st.subheader("Historical Castle Lager Demand Trajectory")
        window = st.slider("Select Trailing Days to Display", 90, len(df), 365)
        sub = df.iloc[-window:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Demand_HL"], name="Daily Demand (HL)", line=dict(color="#D4A017", width=1.8)))
        fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Rolling_Mean_30"], name="30-Day Rolling Mean", line=dict(color="#1E3A8A", width=2.5)))
        fig.update_layout(height=420, template="plotly_white", xaxis_title="Date", yaxis_title="Hectolitres (HL)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Out-of-Sample Model Performance Comparison")
        comp_df = pd.DataFrame([
            {"Model": k, "RMSE (HL)": round(v["RMSE"], 2), "MAE (HL)": round(v["MAE"], 2), "MAPE (%)": round(v["MAPE (%)"], 2), "R²": round(v["R2"], 4)}
            for k, v in results.items()
        ]).sort_values("RMSE (HL)")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        plot_days = st.slider("Test Window Zoom (Days)", 30, len(test_df), 120)
        t_sub = test_df.iloc[-plot_days:]
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["Demand_HL"], name="Actual Demand", line=dict(color="#0F172A", width=2.2)))
        fig_m.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["Pred_XGB"], name="XGBoost", line=dict(color="#D4A017", width=1.8)))
        fig_m.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["Pred_LGBM"], name="LightGBM", line=dict(color="#16A34A", width=1.6, dash="dot")))
        fig_m.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["Pred_SARIMAX"], name="SARIMAX", line=dict(color="#DC2626", width=1.5, dash="dash")))
        fig_m.update_layout(height=420, template="plotly_white", yaxis_title="Demand (HL)")
        st.plotly_chart(fig_m, use_container_width=True)

    with tab3:
        st.subheader("XGBoost Feature Importance & Econometric Drivers")
        imp = pd.DataFrame({"Feature": feature_cols, "Importance": xgb_model.feature_importances_}).sort_values("Importance", ascending=True)
        fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="YlOrBr")
        fig_imp.update_layout(height=480, template="plotly_white")
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab4:
        st.subheader("Interactive Demand Scenario Simulator")
        sc1, sc2, sc3 = st.columns(3)
        sim_inf = sc1.slider("Inflation Rate (%)", 2.0, 50.0, float(df["Inflation_Rate"].iloc[-1]))
        sim_temp = sc2.slider("Ambient Temperature (°C)", 12.0, 38.0, float(df["Temperature_C"].iloc[-1]))
        sim_promo = sc3.selectbox("Promotional Campaign Active?", [0, 1], format_func=lambda x: "Yes (Promo Active)" if x == 1 else "No (Standard Day)")

        latest = df.iloc[-1:].copy()
        latest["Inflation_Rate"] = sim_inf
        latest["Temperature_C"] = sim_temp
        latest["Promo_Flag"] = sim_promo
        X_sim = latest[feature_cols]

        p_xgb = xgb_model.predict(X_sim)[0]
        p_lgb = lgb_model.predict(X_sim)[0]
        p_rf = rf_model.predict(X_sim)[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("XGBoost Forecast", f"{p_xgb:,.2f} HL")
        m2.metric("LightGBM Forecast", f"{p_lgb:,.2f} HL")
        m3.metric("Random Forest Forecast", f"{p_rf:,.2f} HL")

    with tab5:
        st.subheader("Methodology & Data Export")
        st.markdown("""
        - **Econometric Baseline**: Seasonal AutoRegressive Integrated Moving Average with Exogenous Regressors ($SARIMAX(1,1,1)(1,0,1)_7$).
        - **Machine Learning Ensemble**: Non-linear tree ensembles (**XGBoost**, **LightGBM**, **Random Forest**) trained across 16 engineered temporal lag, rolling momentum, calendar, and macroeconomic features.
        """)
        csv_bytes = test_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Test Predictions (CSV)", data=csv_bytes, file_name="Delta_Beverages_Predictions.csv", mime="text/csv")

if not st.session_state.authenticated:
    login_page()
else:
    dashboard()
