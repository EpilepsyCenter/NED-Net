@echo off
title NED-Net (native window)
cd /d "%~dp0"
echo ============================================================
echo   NED-Net  ^|  native desktop app (WebView2)
echo   The app opens in its own window in a few seconds.
echo   Close the app window (or this console) to stop the server.
echo ============================================================
echo.
".venv\Scripts\python.exe" -m eeg_seizure_analyzer.dash_app.desktop
echo.
echo Server stopped.
pause
