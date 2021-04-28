from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf

Server_PORT = 5000
url_root = '/api/model/'

path_model_mlp      = "mlp_room_prediction.pkl"
path_model_knn      = "knn_room_prediction.pkl"
path_model_randf    = "randf_room_prediction.pkl"
path_model_mlinear  = "mlinear_room_prediction.pkl"

# Cấu hình api
app = Flask(__name__)

params = [
         ('giaphong' ,0       ),
         ('dientich' ,0       ),
         ('vido'     ,0       ),
         ('kinhdo'   ,0       ),
         ('loai'     ,'Nhacap'),
         ('drmd'     ,0       ),
         ('kcdc'     ,0       ),
         ('loaiwc'   ,'KKK'   ),
         ('giuongtu' ,'Khong' ),
         ('banghe'   ,'Khong' ),
         ('nonglanh' ,'Khong' ),
         ('dieuhoa'  ,'Khong' ),
         ('tulanh'   ,'Khong' ),
         ('maygiat'  ,'Khong' ),
         ('tivi'     ,'Khong' ),
         ('bep'      ,'Khong' ),
         ('gacxep'   ,'Khong' ),
         ('thangmay' ,'Khong' ),
         ('bancong'  ,'Khong' ),
         ('chodexe'  ,'Khong' ),
]


def get_param_from_request(req):
    list_param = {}
    for param in params:
        value_param = req.args.get(param[0], default=param[1], type=type(param[1]))
        list_param.update({param[0]: value_param})
    return list_param


@app.route(url_root+"mlp", methods=['GET'])
def mlp():
    return "Multilayer Perceptron APIs!"


@app.route(url_root+"knn", methods=['GET', 'POST'])
def knn():
    req_param = get_param_from_request(request)
    return jsonify(req_param)


@app.route(url_root+"randomforest", methods=['GET'])
def randomf():
    return "Random Forest Regression APIs!"


@app.route(url_root+"mlinear", methods=['GET'])
def mlinear():
    return "Multiple Linear Regression APIs!"


# Load model từ file
def load_model(model_file_name):
    model = tf.keras.models.load_model(model_file_name)


if __name__ == '__main__':
    # Có thể cấu hình cổng của server : app.run(debug=True,port=12345)
    app.run(debug=True, port=Server_PORT)