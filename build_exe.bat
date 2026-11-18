@echo off
title Build Standalone Windows Executable
call venv\Scripts\activate
pyinstaller --onefile --clean --name "Delta_Beverages_Forecasting" app.py
echo Build complete! Check the 'dist' folder for Delta_Beverages_Forecasting.exe
pause
