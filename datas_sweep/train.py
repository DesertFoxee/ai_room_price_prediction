import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn import metrics

path_data_raw = 'data_pre/data_train/data_phongtro123_data.csv'
path_data_out = 'housepricedata.csv'

# Xử lý các cột trong dữ liệu thô sang file housepricedata.csv
def convert_data_row():
    df = pd.read_csv(path_data_raw)
    df.drop(['stt','chitiet','thoigian','diachi'], axis=1, inplace=True)
    df.to_csv(path_data_out, index=False,header=True)


def preprocessing_raw_data(df):
    df['loaiwc'] = df['loaiwc'].str.lower()
    df['loaiwc'] = df['loaiwc'].astype('category')
    df['loaiwc'] = df['loaiwc'].cat.codes

    df['loai'] = df['loai'].str.lower()
    df['loai'] = df['loai'].astype('category')
    df['loai'] = df['loai'].cat.codes


def deep_learn(X_train, y_train, X_test, y_test):
    neural_number = 9
    model = Sequential([
        Dense(neural_number, activation='relu'),
        Dense(neural_number, activation='relu'),
        Dense(neural_number, activation='relu'),
        Dense(neural_number, activation='relu'),
        # Dense(neural_number, activation='relu'),
        Dense(1),
    ])

    model.compile(
        optimizer='Adam',
        loss='mean_squared_error',
    )

    model.fit(x=X_train, y=y_train,
              validation_data=(X_test, y_test),
              batch_size=20, epochs=600)

    y_pred = model.predict(X_test)

    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('VarScore:',metrics.explained_variance_score(y_test,y_pred))


def  linear_regressions(X_train, y_train, X_test, y_test):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('VarScore:',metrics.explained_variance_score(y_test,y_pred))


def main():
    df = pd.read_csv(path_data_out)

    preprocessing_raw_data(df)


    X = df.drop(['giaphong'],axis=1).values
    y = df['giaphong'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.fit_transform(X_test)

    linear_regressions(X_train, y_train, X_test, y_test)
    deep_learn(X_train, y_train, X_test, y_test)


# Hàm main
if __name__ == "__main__":
    main()

