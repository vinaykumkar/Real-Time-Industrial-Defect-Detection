@echo off
cd /d D:\NEU
call .venv\Scripts\activate.bat
python scripts\run_camera.py --source 0 --backend auto
