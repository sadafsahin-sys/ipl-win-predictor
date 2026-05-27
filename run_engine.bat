@echo off
cd /d "C:\Users\sadaf\ipl-win-predictor"
call ipl_env\Scripts\activate
python -m streamlit run app.py
pause