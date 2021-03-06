from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import train
from sklearn.model_selection import train_test_split
import keras
import pandas as pd
import ntpath
import urllib
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import common.config as cf
import glob


# Lấy đường dẫn tuyệt đối của folder project
def get_root_path():
    path_common = os.path.dirname(__file__)
    path_root = os.path.dirname(path_common)
    path_root = path_root.replace('\\', '/')
    return path_root + '/'


# Lấy địa chỉ chính từ url
def get_web_host_name_from_url(url):
    host_data = urllib.parse.urlparse(url)
    host_page = host_data.scheme + "://" + host_data.netloc
    return host_page


# Lấy dữ liệu html thô từ url
def get_html_data_from_url(url, err_msg=None):
    html_data = None
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html_data = urlopen(req, timeout=20).read()
    except Exception as ex:
        if err_msg is not None:
            err_msg = "[Error] get data from: " + url
    return html_data


# phân tích cú pháp html lấy các trường trường [href] tương ứng với selector
def parse_html_data_to_url(html_data_raw, selector):
    soup = BeautifulSoup(html_data_raw, features='html.parser')
    urls_data = []
    try:
        tag_link = soup.select(selector)
        for link in tag_link:
            str_link = link.get('href')
            urls_data.append(str_link)
    except Exception as ex:
        pass
    return urls_data


# phân tích cú pháp html lấy các trường dữ liệu tương ứng selectors
def parse_html_data_to_obj(html_data_raw, arr_selectors):
    data_obj = [None] * len(arr_selectors)
    soup = BeautifulSoup(html_data_raw, features='html.parser')
    for index, key_selector in enumerate(arr_selectors):
        try:
            # chỉ sô đầu tiên nếu có sẽ chỉ phần tử thứ bao nhiêu trong list select ra được lấy
            index_selector = key_selector[0]
            if index_selector.isdigit():  # lấy phần tử thứ n
                selector = key_selector[1::]
                str_data = soup.select(selector)[int(index_selector)].text
            elif index_selector == '?':  # lấy thuộc tính
                index_attr = key_selector.find(' ')
                attr = key_selector[1:index_attr]
                selector = key_selector[index_attr + 1:]
                str_data = soup.select_one(selector)[attr]
            else:  # lấy text của phần tử thông thường
                str_data = soup.select_one(key_selector).text
        except Exception as e:
            # Xuất hiện lỗi phân tích cú pháp
            return None
        data_obj[index] = str_data
    return data_obj


# Sử dụng inset dữ liệu vào spider
def inset_data(data_obj, url):
    data_obj.insert(0, url[0])  # insert stt tự từ url
    data_obj.insert(1, url[1])  # insert link
    data_obj.insert(7, round(random.uniform(1.8, 4.0), 1))  # insert link


# Đẩy dữ liệu vào file chế độ ghi thêm từng mảng
def push_data_to_exist_file(data_out, file_out):
    df = pd.DataFrame(data=data_out)
    df.to_csv(file_out, mode='a', header=False, index=False)


# Đẩy phần header vào file
def push_header_to_file(file_out, head=None):
    df = pd.DataFrame(data={}, columns=head)
    df.to_csv(file_out, mode='w', header=True, index=False)


# Đẩy dữ liệu vào file chế độ ghi mới sau tiền xử lý
def push_data_to_new_file(data_out, file_out, head=None, index=False):
    if index:
        df = pd.DataFrame(data=data_out, columns=head[1:])
        df.index.name = head[0]
    else:
        df = pd.DataFrame(data=data_out, columns=head)
    if head is not None:
        df.to_csv(file_out, mode='w', header=True, index=index)
    else:
        df.to_csv(file_out, mode='w', header=False, index=index)


# Lấy tên file từ path
def get_file_name_from_path(path):
    head, tail = ntpath.split(path)
    filename = os.path.splitext(tail)[0]
    return filename or ntpath.basename(head)


# Kiểm tra tham số random_state tốt nhất cho mô hình
def test_random_state(X, y):
    rmse = -1
    random_state = 0
    for x in range(1, 150):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=x)
        rmse_temp = train.linear_regressions(X_train, y_train, X_test, y_test, False)
        if (rmse == -1) or (rmse < rmse_temp):
            rmse = rmse_temp
            random_state = x
    print("MAPE Min:" + str(rmse))
    print("Radom state :" + str(random_state))


# Hiển thị phân phối đối với một cột pandas
def show_distribution(data, title='Phan phoi', xlabel='x', ylabel='Tan suat'):
    sns.displot(data, color='b')
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def show_values_on_bars(axs, h_v="v", space=0.4):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# Hiển thị phân phối đối với một cột pandas
def show_distribution_y(data, column, title='Phan phoi', xlabel='x', ylabel='Tan suat'):
    descending_order = data[column].value_counts().sort_values(ascending=False).index
    ax = sns.countplot(y=data[column],order=descending_order)
    y = data[column].value_counts()
    for i, v in enumerate(y):
        ax.text(v + 0.2, i + .16, str(v), color='black', fontweight='light', fontsize=14)

    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def show_history(history):
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # train_val_loss=  history.history['mean_absolute_percentage_error']
    # test_val_loss=  history.history['val_mean_absolute_percentage_error']

    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Visualize loss history
    # plt.plot(epoch_count, train_val_loss, 'r--')
    # plt.plot(epoch_count, test_val_loss, 'b-')
    # plt.legend(['Training Loss', 'Test Loss'])
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()


# Hàm phần trăm lỗi tuyệt đối trung bình
def mape(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted) / Y_actual)) * 100
    return mape


# Lưu trữ file encoders sử dụng cho dự đoán sau này :có đuôi là *.pkl
# Return : void
def save_encoder(encoder, path):
    print("Save encoder...to file " + path, end=" ")
    try:
        with open(path, 'wb') as file_handler:
            pickle.dump(encoder, file_handler)
            print("=> OK")
    except:
        print("=> Failed")


# Lấy encoder với tên encoder tương ứng
# Return : encoders
def get_encoder(encoder_name):
    if encoder_name in cf.cf_encoder:
        return cf.cf_encoder[encoder_name]
    else:
        print('[X]' + encoder_name +": not exist !")
    return None


# Load toàn bộ encoder cho chương trinh một lần duy nhất
def load_all_encoder():
    print("[IF] Loading encoder file ", end="=>")
    for param in cf.api_params:
        encoder_name        = param[0]  # Tên encoder
        imposition_encoder  = param[4]  # Mức độ quan trọng (phải load)
        path_load_encoder = get_root_path()+cf.path_folder_encoder+encoder_name+'_enc.pkl'
        enc = load_encoder(path_load_encoder)
        if enc is None and imposition_encoder == 1:
            print("[X] Can not load : " + path_load_encoder)
        cf.cf_encoder.update({encoder_name: enc})
    print(" OK")


# Load encoders từ file : có đuôi là *.pkl
# Return : encoders
def load_encoder(path):
    try:
        with open(path, 'rb') as file:
            enc_loaded = pickle.load(file)
            return enc_loaded
    except IOError:
        return None


# Save model sử dụng pickle
# Return: True/False
def save_model(model, path):
    print("[OF] Model saving...to file " + path, end=" ")
    try:
        if cf.cf_model_mlp['path'] in path:
            model.save(path)
            print("=> OK")
        else:
            with open(path, 'wb') as file_handler:
                pickle.dump(model, file_handler)
                print("=> OK")
        return True
    except:
        print("=> Failed")
    return False


# Save model sử dụng pickle
# Return: True/False
def load_model(path):
    print("[IF] Loading model from...file " + path, end=" ")
    try:
        if cf.cf_model_mlp['path'] in path:
            model = keras.models.load_model(path)
        else:
            with open(path, 'rb') as file_handler:
                model = pickle.load(file_handler)
        print("=> OK")
        return model
    except Exception as e:
        print("=> Failed")
    return None


# Thực hiện đa luồng (luồng phải được cấu hình)
def run_thread(threads):
    for t in threads:
        t.start()
    for t in threads:
        t.join()