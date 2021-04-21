import numpy as np
import pandas as pd
from sklearn import metrics
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import common.utils as utl
import seaborn as sns

path_data_raw = 'data_pre/data_train/data_phongtro123_data.csv'
path_data_out = 'housepricedata.csv'
path_data_out_01 = 'housepricedata01.csv'
path_mlp_model = 'prediction_room_model_mlp.h5'
path_linear_model = 'prediction_room_model_linear.h5'

random_state = 110
save_model = False

def preprocessing_raw_data(df):
    df['loaiwc'] = df['loaiwc'].str.lower()
    df['loaiwc'] = df['loaiwc'].astype('category')
    df['loaiwc'] = df['loaiwc'].cat.codes

    df['loai'] = df['loai'].str.lower()
    df['loai'] = df['loai'].astype('category')
    df['loai'] = df['loai'].cat.codes


# Biểu đồ thể hiện biên độ giao động giá thực tế và giá dự đoán
def show_diag_err_amp_fluct(y_test, y_pred):
    c = [i for i in range(len(y_pred))]
    fig = plt.figure()
    plt.plot(c, y_test - y_pred, color="blue", linewidth=2.5, linestyle="-")
    fig.suptitle('Error Terms', fontsize=20)  # Plot heading
    plt.xlabel('Index', fontsize=18 )  # X-label
    plt.ylabel('ytest-ypred', fontsize=16)  # Y-label
    plt.show()


# Biểu đồ sự chênh lệnh giá và tần suất
def show_diag_freq_residuals(y_test, y_pred):
    sns.distplot(y_test - y_pred)
    plt.title("Biểu đồ tần suất và chênh lệch")
    plt.xlabel("Chênh lệch")
    plt.ylabel("Tần suất")
    plt.show()


def show_residuals(y_test, y_pred):
    plt.scatter(y_pred, y_test - y_pred)
    plt.title("Predicted vs residuals")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()


# Mô hình mạng MLP (Multilayer Perceptron) - Deep learning
def mlp(X_train, y_train, X_test, y_test, btest_infor=True, factor=1):
    neural_number = X_train.shape[1] * factor
    model = Sequential([
        Dense(neural_number, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(neural_number, activation='relu'),
        Dense(neural_number, activation='relu'),
        Dense(neural_number, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x=X_train, y=y_train,
              validation_data=(X_test, y_test),
              batch_size=50, epochs=800)

    y_pred = model.predict(X_test)

    if save_model:
        model.save(path_mlp_model)

    if btest_infor:
        utl.show_history(history)
        show_diag_freq_residuals(y_test, y_pred)
        print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
        print('MSE:', metrics.mean_squared_error(y_test, y_pred))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('VarScore:', metrics.explained_variance_score(y_test, y_pred))


# Mô hình multiple linear regression
def linear_regressions(X_train, y_train, X_test, y_test, btest_infor=True):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    if save_model:
        regressor.save(path_linear_model)

    if btest_infor:
        show_diag_freq_residuals(y_test, y_pred)
        print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
        print('MSE:', metrics.mean_squared_error(y_test, y_pred))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('VarScore:', metrics.explained_variance_score(y_test, y_pred))
    return rmse


def main():
    df = pd.read_csv(path_data_out_01)

    preprocessing_raw_data(df)

    X = df.drop(['giaphong'], axis=1).values
    y = df['giaphong'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

    s_scaler = StandardScaler()
    X_train = s_scaler.fit_transform(X_train.astype(np.float))
    X_test = s_scaler.fit_transform(X_test.astype(np.float))

    # linear_regressions(X_train, y_train, X_test, y_test)
    mlp(X_train, y_train, X_test, y_test)


# Hàm main
if __name__ == "__main__":
    main()

