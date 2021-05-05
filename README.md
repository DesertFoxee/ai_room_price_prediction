# Ứng dụng kỹ thuật học sâu trong dự báo giá phòng cho thuê tại Hà Nội
### 1. Điều kiện 
	- Cài đặt Anaconda(4.10.1) trên Window
	
### 2. Tiến hành chạy API trên Window

⮚ Tiến hành cài môi trường Project
```console
	conda env create -f room_api_env.yml
```
Sử dụng tham số <b>_-p_</b> chỉ vị trí: ``` conda env create -f room_api_env.yml -p C:/Users/user/anaconda3/envs ```

⮚ Active môi trường room_api 
```console
	conda activate room_api
```
⮚ Chạy file api.py 
```console
	python api.py
```