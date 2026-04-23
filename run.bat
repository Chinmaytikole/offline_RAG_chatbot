@echo off
if not exist .venv\Scripts\activate.bat (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting app...
uvicorn app:app --reload --host 127.0.0.1 --port 8000
pause
