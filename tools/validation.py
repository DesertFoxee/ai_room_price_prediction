from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
import pandas as pd
import train as train
import  model as models

path_data_train2 = 'roomdata2.csv'


def k_fold_cross_validation(X, y, k, random_state, model):
    cv = KFold(n_splits=k, random_state=random_state, shuffle=True)
    scores = cross_val_score(model, X, y, scoring='r2', cv=cv)

    return mean(scores), scores.min(), scores.max()


def main():
    df = pd.read_csv(path_data_train2)

    train.preprocessing_data(df)
    X = df.drop(['giaphong'], axis=1).values
    y = df['giaphong'].values

    type_model = 1


    if type_model ==1:
        model = models.get_linear_model()
    elif type_model ==2:
        model = models.get_knn_model(5)
    elif type_model ==3:
        model = models.get_random_model(train.random_state)
    elif type_model == 4:
        model = models.get_mlp_model()
    score ,min_score, max_score=  k_fold_cross_validation(X, y, 10 , train.random_state, model)

    print("Mean score :" + str(score))
    print("Min score :" + str(min_score))
    print("Max score :" +str(max_score))


# Hàm main
if __name__ == "__main__":
    main()
