rem ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ AUTO API ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 
@echo OFF

rem : Thay đổi đường dẫn thành conda trên máy tính cá nhân
set CONDA_PATH=C:\Users\DesertFoxee\anaconda3

rem : Tên môi trường cần activate ở đây là room_api
set ENV_NAME=room_api

rem : Activate môi trường
call %CONDA_PATH%\condabin\activate.bat %ENV_NAME%

rem : Chạy file api.py path là vị trí của file auto_api.bat
python tools/api.py

rem ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ AUTO API ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑