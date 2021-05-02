from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
import train
import common.config as cf
import common.utils as utl
import keras
import numpy as np
import re
from enum import Enum

Server_PORT = 5000
url_root = '/api/models/'

mlp_model     = None
knn_model     = None
ranf_model    = None
mlinear_model = None

TF = 0 # Loại TrueFalse
RA = 1 # Loại Khoảng giá trị
LI = 2 # Loại liệt kê

# Cấu hình api hiện thị giá trị unicode
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

params = [
         (cf.col_nam      ,2020    ,RA,[2000,2200]                ,'Giá trị năm từ 2000 trở lên'),
         (cf.col_thang    ,0       ,RA,[1,12]                     ,'Giá trị tháng [1->12]'),
         (cf.col_dientich ,0.0     ,RA,[1,200]                    ,'Giá trị nằm trong khoảng [1->200]m^2'),
         (cf.col_vido     ,0.0     ,RA,[20.0,21.2]                ,'Vi độ không thuộc nội thành Hà Nội'),
         (cf.col_kinhdo   ,0.0     ,RA,[105.0,106.0]              ,'Kinh độ không thuộc nội thành Hà Nội'),
         (cf.col_loai     ,'Nhacap',LI,["Nhacap","Nhatang",'Ccmn'],'Giá trị hợp lệ: [Nhà cấp/Nhà tầng/Ccmn]'),
         (cf.col_drmd     ,0.0     ,RA,[1, 50]                    ,'Giá trị nằm trong khoảng [1->50]m'),
         (cf.col_kcdc     ,0.0     ,RA,[1, 2000]                  ,'Giá trị nằm trong khoảng [1->2000]m'),
         (cf.col_loaiwc   ,'KKK'   ,LI,["KKK", "Khepkin"]         ,'Giá trị hợp lệ : [KKK/Khép kín]'),
         (cf.col_giuongtu ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
         (cf.col_banghe   ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
         (cf.col_nonglanh ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
         (cf.col_dieuhoa  ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
         (cf.col_tulanh   ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
         (cf.col_maygiat  ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
         (cf.col_tivi     ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
         (cf.col_bep      ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
         (cf.col_gacxep   ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
         (cf.col_thangmay ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
         (cf.col_bancong  ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
         (cf.col_chodexe  ,0       ,TF,[]                         ,'Giá trị hợp lệ : [Có/Không]'),
]

test_param ={
  "bancong": 0,
  "banghe": 0,
  "bep": 0,
  "chodexe": 1,
  "dientich": 20.0,
  "dieuhoa": 0,
  "drmd": 2.2,
  "gacxep": 1,
  "giuongtu": 0,
  "kcdc": 169.0,
  "kinhdo": 105.850008,
  "vido": 20.972612,
  "loai": "Nhacap",
  "loaiwc": "Khepkin",
  "maygiat": 0,
  "nam": 2019,
  "nonglanh": 0,
  "thang": 4,
  "thangmay": 0,
  "tivi": 0,
  "tulanh": 0,
}


def month_encoder(month):
    month_enc = {}
    value_sin = np.sin(2 * np.pi * month/12)
    value_cos = np.cos(2 * np.pi * month/12)

    month_enc.update({cf.col_thang+'_sin': value_sin})
    month_enc.update({cf.col_thang+'_cos': value_cos})
    return month_enc


# Chuẩn hóa dữ liệu là một đối tượng tham số cho dự báo
# Return : Object param sau chuẩn hóa
def standardize_room_param(obj_param):
    obj_param_standard = {}
    for param in params:
        param_name, param_value = param[0], obj_param[param[0]]
        # Xử lý riêng trường hợp của cột tháng
        if param_name == cf.col_thang:
            value_month = month_encoder(param_value)
            obj_param_standard.update(value_month)
        else:
            # Sử dụng standard tham số của obj nếu tồn tại file
            enc = utl.load_encoder(cf.path_folder_encoder+param_name+'_enc.pkl')
            if enc is not None:
                value = enc.transform([[param_value]])
                value = value[0, 0]
            else:
                value = param_value
            obj_param_standard.update({param_name: value})
    return obj_param_standard


# Kiểm tra, xác thực các trường của room đúng
# Return : json các trường lỗi
def validate_room_param(room_param):
    error = {}
    for param in params:
        req_value = room_param[param[0]]
        field_name, field_type = param[0], param[2]
        field_cmp, field_msg = param[3], param[4]
        had_error = False
        if field_type == RA:
            if not (field_cmp[0] <= req_value <= field_cmp[1]):
                had_error = True
        elif field_type == LI:
            if not any(req_value in s for s in field_cmp):
                had_error = True
        elif field_type == TF:
            if req_value != 0 and req_value != 1:
                had_error = True
        if had_error:
            error.update({field_name: field_msg})
    return error


# Lấy tham số của phòng từ request
# Return : Danh sách tham số
def get_room_param_from_request(req):
    list_param = {}
    for param in params:
        param_name, param_def = param[0], param[1]
        value_param = req.args.get(param_name, default=param_def, type=type(param_def))
        list_param.update({param_name: value_param})
    return list_param


def predict_room_price_from_model(model, room_param, path_model):
    err = validate_room_param(room_param)
    if err:
        status = {'success': False, 'err': err}
    else:
        obj_param = standardize_room_param(room_param)
        model_param_1D = np.array(list(obj_param.values()))
        model_param = model_param_1D.reshape(1, -1)
        price = -1
        # Load models nếu lần đầu chưa load được
        if model is None:
            model = utl.load_model(path_model)

        if model is not None:
            try:
                price = model.predict(model_param)
            except:
                print("[Error] : Prediction failed !!")
                price = -1
        if price <= 0:
            status = {'success': False, 'predict': -1}
        else:
            status = {'success': True, 'predict': price[0]}
    return status


# Param : http://127.0.0.1:5000/api/model/knn?thang=4&nam=2019&vido=20.972612&kinhdo=105.850008&loai=Nhacap&loaiwc=Khepkin&dientich=20.0&drmd=2.2&kcdc=169.0&chodexe=1&gacxep=1
# @app.route(url_root+"knn", methods=['GET', 'POST'])
# def KNN_regression():
#     req_param = get_param_from_request(request)
#     res = get_predict_from_model(knn_model, req_param, cf.cf_model_knn['path'])
#     return jsonify(res)
#
#
# @app.route(url_root+"rand", methods=['GET', 'POST'])
# def random_forest_regression():
#     req_param = get_param_from_request(request)
#     res = get_predict_from_model(ranf_model, req_param, cf.cf_model_randf['path'])
#     return jsonify(res)
#
#
# @app.route(url_root+"mlp", methods=['GET', 'POST'])
# def MLP_regression():
#     req_param = get_param_from_request(request)
#     res = get_predict_from_model(mlp_model, req_param, cf.cf_model_mlp['path'])
#     return jsonify(res)
#
#

# Param : http://127.0.0.1:5000/api/models/linear?thang=4&nam=2019&vido=20.972612&kinhdo=105.850008&loai=Nhacap&loaiwc=Khepkin&dientich=20.0&drmd=2.2&kcdc=169.0&chodexe=1&gacxep=1
@app.route(url_root+"linear", methods=['GET', 'POST'])
def multiple_linear_regression():
    room_param = get_room_param_from_request(request)
    res = predict_room_price_from_model(mlinear_model, room_param, cf.cf_model_mlinear['path'])
    return jsonify(res)


if __name__ == '__main__':
    # Có thể cấu hình cổng của server : app.run(debug=True,port=12345)
    app.run(debug=True, port=Server_PORT)
    # ML = utl.load_model(cf.cf_model_mlinear['path'])
    # obj_param = standardize_obj_param(test_param)
    # print(obj_param)
    # print(type(obj_param.values()))
    # obj_param = standardize_obj_param(test_param)
    # print(obj_param)
    # enc = utl.load_encoder(cf.path_folder_encoder + 'kinhdo' + '_enc.pkl')
    # value = enc.transform([[105]])
    # print(value)

    # abc = check_param(test_param)
    # print(abc)