test&cls
@echo ON
@echo "@@@@@@@   AutoSpider    @@@@@@@@@@"                                                                             
@echo "      (@@@@@@@@@@@@@@@@@@@)       "       
@echo "@@@@@@(...................)@@@@@@@"       



@echo OFF 
rem ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ AUTO API ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 


rem : Thay đổi đường dẫn thành conda trên máy tính cá nhân
set CONDA_PATH=C:\Users\DesertFoxee\anaconda3

rem : Tên môi trường cần activate ở đây là room_api
set ENV_NAME=room_api

rem : Activate môi trường
call %CONDA_PATH%\condabin\activate.bat %ENV_NAME%

rem : Chạy file spider.py path là vị trí của file auto_spider.bat
python tools/spider.py

rem ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ AUTO API ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑