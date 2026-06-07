@echo off
REM ProteinBase QC Studio - setup + run
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
pip install -r requirements.txt
if not exist data\measurements.db ( python ingest.py )
streamlit run app.py
