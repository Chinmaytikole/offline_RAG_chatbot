@echo off
echo Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment. Make sure Python is installed.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo Installation complete. Run "python run.py" to start the app.
pause
