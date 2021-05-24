from flask import Flask, request, jsonify
import common.config as cf
import common.utils as utl
import numpy as np
import geocoder as geo
from flask_cors import CORS


SERVER_PORT = 5000
URL_ROOT = '/api/models/'

root_path = utl.get_root_path()

# Cấu hình api hiện thị giá trị unicode
app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False


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
  "kinhdo": 105.815203,
  "vido": 20.993913,
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


def district_encoder(district_name):
    enc = utl.get_encoder(cf.col_quan)
    if enc is None:
        return None
    labels = enc.get_feature_names([cf.col_quan])
    if district_name is None:
        district_enc = {labels[i]: 0 for i in range(len(labels))}
        return district_enc
    else:
        values = enc.transform([[district_name]]).toarray()[0]
        district_enc = {labels[i]: values[i] for i in range(len(labels))}
        return district_enc


# Chuẩn hóa dữ liệu là một đối tượng tham số cho dự báo
# Return : Object param sau chuẩn hóa
def standardize_room_param(obj_param):
    obj_param_standard = {}
    for param in cf.api_params:
        param_name, param_value = param[0], None
        if param_name in obj_param:
            param_value = obj_param[param[0]]

        # Xử lý riêng trường hợp của cột tháng
        if param_name == cf.col_thang:
            value_month = month_encoder(param_value)
            obj_param_standard.update(value_month)
        # Xử lý riêng trường hợp của quận
        elif param_name == cf.col_quan:
            district_name = geo.get_district_coordinates(obj_param[cf.col_vido], obj_param[cf.col_kinhdo])
            if district_name is None:
                print("[X] District_name is None")
            value_district = district_encoder(district_name)
            obj_param_standard.update(value_district)
        else:
            # Sử dụng standard tham số của obj nếu tồn tại file
            enc = utl.get_encoder(param_name)
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
    for param in cf.api_params:
        req_value = room_param[param[0]]
        field_name, field_type = param[0], param[2]
        field_cmp, field_msg   = param[3], param[5]
        had_error = False
        if field_type == cf.RA:
            if not (field_cmp[0] <= req_value <= field_cmp[1]):
                had_error = True
        elif field_type == cf.LI:
            if not any(req_value in s for s in field_cmp):
                had_error = True
        elif field_type == cf.TF:
            if req_value != 0 and req_value != 1:
                had_error = True
        elif field_type == cf.NO:  # Không xử lý và bỏ qua nó
            continue
        if had_error:
            error.update({field_name: field_msg})
    return error


# Chuyển đổi số thập phân sang string format currency
def round_currency_up(number):
    interger = 4
    if number >= 1000000:
        interger = 5
    cur = number / pow(10, interger)
    cur = round(cur) * pow(10, interger)
    str_cur = "{:10,}".format(cur)
    str_cur = str_cur.replace(',', '.')
    return str_cur


# Lấy tham số của phòng từ request
# Return : Danh sách tham số
def get_room_param_from_request(req):
    list_param = {}
    for param in cf.api_params:
        param_name, param_def = param[0], param[1]
        value_param = req.args.get(param_name, default=param_def, type=type(param_def))
        list_param.update({param_name: value_param})
    return list_param


def predict_room_price_from_model(conf_model, room_param):
    err = validate_room_param(room_param)
    if err:
        status = {'success': False, 'err': err}
    else:
        obj_param      = standardize_room_param(room_param)
        model_param_1D = np.array(list(obj_param.values()))
        model_param    = model_param_1D.reshape(1, -1)
        price = -1

        # Load models nếu lần đầu chưa load được
        if conf_model['reload']:
            conf_model['model'] = utl.load_model(root_path + conf_model['path'])
            conf_model['reload'] = False

        if conf_model['model'] is not None:
            try:
                price = conf_model['model'].predict(model_param)
            except Exception as e:
                conf_model['reload'] = True
                print("[Error] : Prediction failed !!")
        else:
            conf_model['reload'] = True

        # Kiểm tra giá hợp lệ không
        if price <= 0:
            status = {'success': False, 'predict': -1}
        else:
            price = price.flatten()
            price_room = np.float64(price[0])
            str_price  = round_currency_up(price_room)
            status = {'success': True, 'predict': str_price}
    return status


# Param : http://127.0.0.1:5000/api/models/knn?thang=4&nam=2019&vido=20.972612&kinhdo=105.850008&loai=Nhacap&loaiwc=Khepkin&dientich=20.0&drmd=2.2&kcdc=169.0&chodexe=1&gacxep=1
@app.route(URL_ROOT+"knn", methods=['GET', 'POST'])
def KNN_regression():
    room_param = get_room_param_from_request(request)
    res = predict_room_price_from_model(cf.cf_model_knn, room_param)
    return jsonify(res)


# Param : http://127.0.0.1:5000/api/models/rand?thang=4&nam=2019&vido=20.972612&kinhdo=105.850008&loai=Nhacap&loaiwc=Khepkin&dientich=20.0&drmd=2.2&kcdc=169.0&chodexe=1&gacxep=1
@app.route(URL_ROOT+"rand", methods=['GET', 'POST'])
def random_forest_regression():
    room_param = get_room_param_from_request(request)
    res = predict_room_price_from_model(cf.cf_model_randf, room_param)
    return jsonify(res)


# Param : http://127.0.0.1:5000/api/models/mlp?thang=4&nam=2019&vido=20.972612&kinhdo=105.850008&loai=Nhacap&loaiwc=Khepkin&dientich=20.0&drmd=2.2&kcdc=169.0&chodexe=1&gacxep=1
@app.route(URL_ROOT+"mlp", methods=['GET', 'POST'])
def MLP_regression():
    room_param = get_room_param_from_request(request)
    res = predict_room_price_from_model(cf.cf_model_mlp, room_param)
    return jsonify(res)


# Param : http://127.0.0.1:5000/api/models/linear?thang=4&nam=2019&vido=20.972612&kinhdo=105.850008&loai=Nhacap&loaiwc=Khepkin&dientich=20.0&drmd=2.2&kcdc=169.0&chodexe=1&gacxep=1
@app.route(URL_ROOT+"linear", methods=['GET', 'POST'])
def multiple_linear_regression():
    room_param = get_room_param_from_request(request)
    res = predict_room_price_from_model(cf.cf_model_mlinear, room_param)
    return jsonify(res)


if __name__ == '__main__':
    # Có thể cấu hình cổng của server : app.run(debug=True,port=12345)
    utl.load_all_encoder()
    app.run(debug=True, port=SERVER_PORT)

    # Phần này để test :
    # ML = utl.load_model(cf.cf_model_mlinear['path'])
    # obj_param = standardize_obj_param(test_param)
    # print(obj_param)
    # print(type(obj_param.values()))
    # obj_param = standardize_obj_param(test_param)
    # print(obj_param)
    # enc = utl.load_encoder(cf.path_folder_encoder + 'kinhdo' + '_enc.pkl')
    # value = enc.transform([[105]])
    # print(value)
