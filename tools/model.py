from keras.layers import Dense
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


# Mô hình k-Nearest Neighbors Regression
def get_knn_model(n_neighbors):
    KNN = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    return KNN


# Mô hình Multiple Linear Regression
def get_linear_model():
    ML = LinearRegression()
    return ML


# Mô hình Random Forest Regression
def get_random_model(factor_random):
    RF = RandomForestRegressor(n_estimators=1000, random_state=factor_random)
    return RF


# Mô hình mạng MLP (Multilayer Perceptron) - Deep learning
def get_mlp_model(input_size, layer_hidden_size, neural_hidden_size):
    MLP = Sequential()
    for ilay in range(layer_hidden_size):
        if ilay ==0:  # Input layer
            MLP.add(Dense(neural_hidden_size, activation='relu', input_shape=(input_size,)))
        else:         # Hidden layer
            MLP.add(Dense(neural_hidden_size, activation='relu'))
    # Output layer
    MLP.add(Dense(1))
    MLP.compile(optimizer='adam', loss='mean_squared_error')
    return MLP







