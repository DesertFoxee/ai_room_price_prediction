import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras.initializers as ker_init
import matplotlib.pyplot as plt
import common.config as cf
import common.utils as utl
import seaborn as sns
import model as models

path_data_raw = 'data_train/data_phongtro123_data.csv'
path_data_train = 'roomdata.csv'
path_data_train2 = 'roomdata2.csv'

random_state = 42

#⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣ Biểu diễn ⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣
# Biểu đồ giá trị thực tế và giá dự báo dạng đường
def show_actual_and_predict(y_test, y_pred):
    actual_list = y_test.tolist()
    predict_list = y_pred.tolist()
    index_list = list(range(1, len(actual_list)+1))

    plt.plot(index_list, actual_list, label='Thực tế', linewidth=2)
    plt.plot(index_list, predict_list, label='Dự báo', linewidth=2)

    plt.xlabel('Mẫu thử nghiệm')
    plt.ylabel('Giá phòng')
    plt.legend(loc='upper left')
    plt.xticks(index_list)
    plt.title('Giá phòng thực tế và dự báo')
    plt.show()


# Hiển thị thông tin đánh giá mô hình dự trên y actual và y prediction
def print_test_infor(y_test, y_pred):
    print('MAE     :', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE     :', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE    :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('MAPE    :', metrics.mean_absolute_percentage_error(y_test, y_pred))
    print('VarScore:', metrics.explained_variance_score(y_test, y_pred))


# Biểu đồ thể hiện biên độ giao động giá thực tế và giá dự đoán
def show_residual_actual_and_predict(y_test, y_pred):
    diff = y_test - y_pred
    index_list = list(range(1, len(y_test) + 1))
    plt.plot(index_list, diff, label='Số dư', linewidth=2)
    plt.title('Sự chênh lệnh giá phòng và dự đoán')  # Plot heading
    plt.xlabel('Index', fontsize=12)  # X-label
    plt.ylabel('Thực tế- Dự đoán', fontsize=12)  # Y-label
    plt.show()


# Biểu đồ sự chênh lệnh giá và tần suất
def show_residual_and_frequency(y_test, y_pred):
    diff = y_test - y_pred
    sns.displot(x=diff, kde=True)
    plt.title("Biểu đồ tần suất và chênh lệch")
    plt.xlabel("Chênh lệch")
    plt.ylabel("Tần suất")
    plt.show()
#⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡ Biểu diễn ⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡


#⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣ Model ⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣
# Mô hình mạng MLP (Multilayer Perceptron) - Deep learning
def multiple_layer_perceptron_regression(X_train, y_train, X_test, y_test, show_infor=True, save_model=False):
    neural_number     = X_train.shape[1]
    input_size        = X_train.shape[1]
    hidden_layer_size = 3
    MLP = models.get_mlp_model(input_size, hidden_layer_size, neural_number, ker_init.he_normal())
    history = MLP.fit(X_train, y_train, batch_size=32, epochs=600)

    y_pred = MLP.predict(X_test)
    y_pred = y_pred.flatten()

    if save_model:
        utl.save_model(MLP, cf.cf_model_mlp['path'])

    if show_infor:
        show_actual_and_predict(y_test, y_pred)
        show_residual_actual_and_predict(y_test, y_pred)
        # utl.show_history(history)
        show_residual_and_frequency(y_test, y_pred)
        print_test_infor(y_test, y_pred)


# Mô hình Multiple Linear Regression
def linear_regressions(X_train, y_train, X_test, y_test, show_infor=True, save_model=False):
    ML = models.get_linear_model()

    ML.fit(X_train, y_train)
    y_pred = ML.predict(X_test)

    var_score = metrics.explained_variance_score(y_test, y_pred)

    if save_model:
        utl.save_model(ML, cf.cf_model_mlinear['path'])

    if show_infor:
        show_actual_and_predict(y_test, y_pred)
        show_residual_actual_and_predict(y_test, y_pred)
        # utl.show_history(history)
        show_residual_and_frequency(y_test, y_pred)
        print_test_infor(y_test, y_pred)
    return var_score


# Mô hình k-Nearest Neighbors Regression
def knn_regressions(X_train, y_train, X_test, y_test, show_infor=True, save_model=False):
    # KNN = KNeighborsRegressor(n_neighbors=5, weights='distance')
    KNN = models.get_knn_model(5)

    KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)

    var_score = metrics.explained_variance_score(y_test, y_pred)

    if save_model:
        utl.save_model(KNN, cf.cf_model_knn['path'])

    if show_infor:
        show_actual_and_predict(y_test, y_pred)
        show_residual_actual_and_predict(y_test, y_pred)
        show_residual_and_frequency(y_test, y_pred)
        print_test_infor(y_test, y_pred)
    return var_score


# Mô hình Random Forest Regression
def random_forest_regressions(X_train, y_train, X_test, y_test, show_infor=True, save_model=False):
    RF = models.get_random_model(random_state)

    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)

    var_score = metrics.explained_variance_score(y_test, y_pred)

    if save_model:
        utl.save_model(RF, cf.cf_model_randf['path'])

    if show_infor:
        show_actual_and_predict(y_test, y_pred)
        show_residual_actual_and_predict(y_test, y_pred)
        show_residual_and_frequency(y_test, y_pred)
        print_test_infor(y_test, y_pred)
    return var_score
#⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡ Model ⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡


#⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣ Chuẩn hóa ⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣⭣
# Encoder cột tháng có tính chất chu kỳ
def MonthEncoder(df):
    df.insert(df.columns.get_loc(cf.col_thang), cf.col_thang+'_sin', np.sin(2 * np.pi * df[cf.col_thang]/12))
    df.insert(df.columns.get_loc(cf.col_thang), cf.col_thang+'_cos', np.cos(2 * np.pi * df[cf.col_thang]/12))
    df.drop([cf.col_thang], axis='columns', inplace=True)


def preprocessing_data(df, save=False):
    # "giuongtu","banghe","nonglanh","dieuhoa","tulanh","maygiat","tivi","bep","gacxep","thangmay","bancong","chodexe"
    col_cate_hot   = ['quan']                                         # không thứ tự không ảnh hưởng trọng số
    col_cate_ori   = [['loai',  ['Nhacap','Nhatang','Ccmn']],
                      ['loaiwc',['KKK','Khepkin'          ]]]         # có thứ tự : cold warm, hot
    col_cate_lab   = []                                               # dùng cho cate không có thứ tự
    col_standard   = ["dientich","vido","kinhdo","drmd","kcdc"]
    col_normal     = ["nam"]

    # Chuẩn hóa trường tháng có tính chất chu kỳ
    MonthEncoder(df)

    # categories label: Dành cho danh mục tính liệt kê vẫn ảnh có ảnh hưởng thứ tự
    for col_name in col_cate_lab:
        enc = preprocessing.LabelEncoder()
        df[col_name] = enc.fit_transform(df[[col_name]])

    # categories label : Danh cho danh mục không ảnh tính độc lập
    for col_name in col_cate_hot:
        hot = preprocessing.OneHotEncoder()
        oe_results = hot.fit_transform(df[[col_name]]).toarray()
        ohe_df = pd.DataFrame(oe_results, columns=hot.get_feature_names([col_name]))
        for col_new in ohe_df:
            df.insert(df.columns.get_loc(col_name), col_new, ohe_df[col_new].values)
        df.drop([col_name], axis='columns', inplace=True)
        if save:
            utl.save_encoder(hot, cf.path_folder_encoder + col_name + '_enc.pkl')

    # categories ori : Dành cho danh sách có tính mức độ cấp độ
    # df['loai'] = df['loai'].map({'Nhacap':1,'Nhatang': 2,'Ccmn': 3})
    # df['loaiwc'] = df['loaiwc'].map({'KKK':1,'Khepkin': 2})
    for col_name in col_cate_ori:
        ori = preprocessing.OrdinalEncoder(categories=[col_name[1]])
        df[col_name[0]] = ori.fit_transform(df[[col_name[0]]])
        if save:
            utl.save_encoder(ori, cf.path_folder_encoder + col_name[0] + '_enc.pkl')

    # standardize : Thường dành cho các trường phân phối chuẩn
    for col_name in col_standard:
        stan = preprocessing.StandardScaler()
        df[col_name] = stan.fit_transform(df[[col_name]])
        if save:
            utl.save_encoder(stan, cf.path_folder_encoder + col_name + '_enc.pkl')

    # normalize :  Dành cho nhưng trường phân phối không chuẩn
    for col_name in col_normal:
        norm = preprocessing.MinMaxScaler()
        df[col_name] = norm.fit_transform(df[[col_name]])
        if save:
            utl.save_encoder(norm, cf.path_folder_encoder + col_name + '_enc.pkl')
#⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡ Chuẩn hóa ⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡⭡


def main():
    # df = pd.read_csv(path_data_train)
    #
    # # utl.show_distribution(df["giaphong"], "Phân phối giá phòng", "Giá phòng")
    # preprocessing_data(df, save=False)
    # X = df.drop(['giaphong'], axis=1).values
    # y = df['giaphong'].values
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
    # utl.test_random_state(X,y)

    # Dùng train và lưu lưu
    # linear_regressions(X_train, y_train, X_test, y_test, show_infor=True, save_model=True)
    # knn_regressions(X_train, y_train, X_test, y_test, show_infor=True, save_model=True)
    # random_forest_regressions(X_train, y_train, X_test, y_test, show_infor=True, save_model=True)
    # multiple_layer_perceptron_regression(X_train, y_train, X_test, y_test, show_infor=True, save_model=True)

    # Dùng để train và không lưu
    df = pd.read_csv(path_data_train2)

    preprocessing_data(df, save=False)
    X = df.drop(['giaphong'], axis=1).values
    y = df['giaphong'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

    # linear_regressions(X_train, y_train, X_test, y_test, show_infor=False, save_model=True)
    # knn_regressions(X_train, y_train, X_test, y_test, show_infor=False, save_model=True)
    # random_forest_regressions(X_train, y_train, X_test, y_test, show_infor=False,save_model=True)
    # multiple_layer_perceptron_regression(X_train, y_train, X_test, y_test, show_infor=False, save_model=True)

    # linear_regressions(X_train, y_train, X_test, y_test, show_infor=True, save_model=False)
    # knn_regressions(X_train, y_train, X_test, y_test, show_infor=True, save_model=False)
    # random_forest_regressions(X_train, y_train, X_test, y_test, show_infor=True, save_model=False)
    multiple_layer_perceptron_regression(X_train, y_train, X_test, y_test, show_infor=True, save_model=False)


# Hàm main
if __name__ == "__main__":
    main()
