#!/usr/bin/env sh
cd "$(dirname "$0")"
python -m streamlit run src/email_assistant/ui/app.py
