@echo off
REM Simple batch file to run the Streamlit app
REM Tries to use an existing hermes-agent virtual environment if present,
REM otherwise creates a temporary venv in the shareable_version folder.

set SHAREABLE_DIR=%~dp0
set VENV_PATH=%USERPROFILE%\AppData\Local\hermes\hermes-agent\venv

REM Check if the hermes-agent venv exists
if exist "%VENV_PATH%\Scripts\python.exe" (
    echo Using virtual environment at %VENV_PATH%
    "%VENV_PATH%\Scripts\python.exe" -m streamlit run "%SHAREABLE_DIR%app.py"
    goto :eof
)

REM If not, create a temporary venv in the shareable_version folder
echo Hermes-agent venv not found. Creating a temporary virtual environment...
cd /d "%SHAREABLE_DIR%"
if not exist "venv" (
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
pause