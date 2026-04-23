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
echo Downloading Whisper base model (Systran/faster-whisper-base, ~145 MB)...
python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-base', local_dir='whisper_model', local_dir_use_symlinks=False)"
if errorlevel 1 (
    echo Failed to download Whisper model. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo Installation complete. Run run.bat to start the app.
pause
