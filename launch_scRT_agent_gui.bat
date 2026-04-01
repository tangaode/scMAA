@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%\.venv\Scripts\pythonw.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%\.venv\Scripts\pythonw.exe"
) else if exist "%SCRIPT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

for %%F in ("%SCRIPT_DIR%OPENAI.env" "%SCRIPT_DIR%deepseek.env" "%SCRIPT_DIR%.env") do (
    if exist "%%~F" (
        for /f "usebackq tokens=* delims=" %%L in ("%%~F") do (
            set "LINE=%%L"
            if not "!LINE!"=="" (
                echo !LINE! | findstr /b /c:"#">nul
                if errorlevel 1 (
                    for /f "tokens=1,* delims==" %%A in ("!LINE!") do set "%%A=%%B"
                )
            )
        )
    )
)

cd /d "%SCRIPT_DIR%"
start "" "%PYTHON_EXE%" "%SCRIPT_DIR%run_scrt_gui.pyw"
