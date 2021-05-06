# Ứng dụng kỹ thuật học sâu trong dự báo giá phòng cho thuê tại Hà Nội
### 1. Điều kiện 
	- Cài đặt Anaconda(4.10.1) trên Window
	
### 2. Tiến hành chạy API trên Window

⮚ Tiến hành cài môi trường Project
```path_project``` 	: Đường dẫn của project <br>
```path_env``` 		: Đường dẫn của môi trường, mặc định : ```C:/Users/user/anaconda3/envs ```

```console
	conda env create -f <path_project>room_api_env.yml 
```
Sử dụng tham số ```-p``` chỉ vị trí: ``` conda env create -f <path_project>room_api_env.yml -p <path_env>/room_api```

#### C1: Chạy trực tiếp file ```api.py```
⮚ Active môi trường room_api 
```console
	conda activate room_api
```
⮚ Chạy api trực tiếp api.py trên cmd anaconda 
```console
	python api.py
```
#### C2: Chạy thông qua file ```auto_api.bat```
Sau khi tiến hành cài môi trường cho project <br>
Sửa biến ```CONDA_PATH``` trong ```auto_api.bat``` => Đường dẫn anaconda trên máy.
```shell
	CONDA_PATH=C:\Users\DesertFoxee\anaconda3 -> CONDA_PATH=path_anaconda
```