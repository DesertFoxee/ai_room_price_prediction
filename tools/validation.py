from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
import numpy  as np
import pandas as pd
import train  as train
import model  as models
import seaborn as sns
import matplotlib.pyplot as plt

path_data_train2 = 'roomdata2.csv'


# Biểu đồ mức độ quan trọng của thuộc tính sử dụng random forest
def show_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + 'Mức độ quan trọng của thuộc tính')
    plt.xlabel('Độ quan trọng thuộc tính')
    plt.ylabel('Tên thuộc tính')
    plt.show()


# Hàm biểu đồ mức độ quan trọng của thuộc tính sử dụng Ramdom forest
def feature_importain(X_train, y_train):
    RF = models.get_random_model(train.random_state)
    RF.fit(X_train.values, y_train.values)
    show_feature_importance(RF.feature_importances_,X_train.columns,"Random forest ")


# Hàm thâm định chất lượng model sử dụng kfold
# Return : MAPE và độ lệch chuẩn
def k_fold_cross_validation(X, y, k, random_state, model):
    cv = KFold(n_splits=k, random_state=random_state, shuffle=True)
    # neg_mean_absolute_percentage_error
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_percentage_error', cv=cv)
    return abs(np.mean(scores)),np.std(scores)


# Hàm thâm định chất lượng model mlp sử dụng kfold
# Return : MAPE và độ lệch chuẩn
def k_fold_cross_validation_mlp(X, y, k, random_state, model):
    kfold = KFold(n_splits=k,random_state=random_state, shuffle=True)
    mape_list = list()
    for train_ids, val_ids in kfold.split(X, y):
        model.fit(X[train_ids], y[train_ids], batch_size=32, epochs=600, verbose=0)
        y_pred = model.predict(X[val_ids])
        scores = mean_absolute_percentage_error(y_pred, y[val_ids])
        mape_list.append(scores)
    return abs(np.mean(mape_list)), np.std(mape_list)


def mainex():
    df = pd.read_csv(path_data_train2)
    train.preprocessing_data(df)
    X = df.drop(['giaphong'], axis=1)
    y = df['giaphong']

    feature_importain(X,y)


def main():
    df = pd.read_csv(path_data_train2)

    train.preprocessing_data(df)
    X = df.drop(['giaphong'], axis=1).values
    y = df['giaphong'].values

    type_model = 4

    if type_model ==1:
        model = models.get_linear_model()
    elif type_model ==2:
        model = models.get_knn_model(5)
    elif type_model ==3:
        model = models.get_random_model(train.random_state)
    elif type_model == 4:
        model = models.get_mlp_model(X.shape[1], 3, 2)
    # score, std_score = k_fold_cross_validation(X, y, 10, train.random_state, model)
    score, std_score = k_fold_cross_validation_mlp(X, y,10, train.random_state, model)
    print("MAPE :%0.2f +/- %0.2f" %(score, std_score))


# Hàm main
if __name__ == "__main__":
    main()
    # mainex()