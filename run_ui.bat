@echo off
cd /d "%~dp0"
python -m streamlit run src\email_assistant\ui\app.py
