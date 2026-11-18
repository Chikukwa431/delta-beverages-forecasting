@echo off
title Delta Beverages - Castle Lager Forecasting Suite Launcher
echo ============================================================
echo   DELTA BEVERAGES CASTLE LAGER DEMAND FORECASTING SUITE
echo   Setting up Python environment and launching dashboard...
echo ============================================================
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
pause
