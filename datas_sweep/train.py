import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


path_data_raw = 'data_pre/data_train/data_phongtro123_data.csv'
path_data_out = 'housepricedata.csv'

# Xử lý các cột trong dữ liệu thô sang file housepricedata.csv
def convert_data_row():
    df = pd.read_csv(path_data_raw)
    df.drop(['stt','chitiet','thoigian','diachi'], axis=1, inplace=True)
    df.to_csv(path_data_out, index=False,header=True)


def stand_data(df):
    df['loaiwc'] = df['loaiwc'].str.lower()
    df['loaiwc'] = df['loaiwc'].astype('category')
    df['loaiwc'] = df['loaiwc'].cat.codes

    df['loai'] = df['loai'].str.lower()
    df['loai'] = df['loai'].astype('category')
    df['loai'] = df['loai'].cat.codes


def main():
    df = pd.read_csv(path_data_out)
    stand_data(df)

    dataset = df.values
    # Lấy ra cột giá
    col_price = dataset[:, 0]
    col_features = dataset[:, 1:9]
    Y = col_price
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(col_features)

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

    model = Sequential([
        Dense(32, activation='relu', input_shape=(10,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    hist = model.fit(X_train, Y_train,
                     batch_size=32, epochs=100,
                     validation_data=(X_val, Y_val))
    print(model.evaluate(X_test, Y_test)[1])


# Hàm main
if __name__ == "__main__":
    main()

