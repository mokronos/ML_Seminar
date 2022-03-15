import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, explained_variance_score


def feat_sel(X, y):
    feat_count = X.shape[1]
    opt_feat_count = 0
    high_score = 0

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()

    for i in range(feat_count):

        rfe = RFE(model, n_features_to_select=i+1)
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe, y_train)
        score = model.score(X_test_rfe, y_test)

        if score>high_score:
            high_score = score

            opt_feat_count = i+1

    print("opt number of features: {}".format(opt_feat_count))
    print("score with {} features: {}".format(opt_feat_count, high_score))

    rfe = RFE(model, n_features_to_select=opt_feat_count)
    X_rfe = rfe.fit_transform(X,y)

    model.fit(X_rfe,y)
    print(rfe.support_)
    print(rfe.ranking_)

    cols = list(X.columns)
    temp = pd.Series(rfe.support_, index = cols)
    selected_features_rfe = temp[temp==True].index
    print(selected_features_rfe)

    X1 = X[selected_features_rfe]

    X_train, X_test, y_train, y_test = train_test_split(X1,y, test_size = 0.33)
    ml = LinearRegression()
    ml.fit(X_train, y_train)

    y_pred = ml.predict(X_test)




    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def lin_reg_eval():

        r2 = r2_score(y_test, y_pred)
        mse = np.sqrt(mean_squared_error(y_test,y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        variance = explained_variance_score(y_test, y_pred)
        mean_percentage = mean_absolute_percentage_error(y_test, y_pred)

        print("Auswertung der linearen Regression: ")
        print("_________________________________________________")    
        print("Number of Features: {}".format(opt_feat_count))
        print("R2 Score: {}".format(r2))
        print("Mean Squared Error: {}".format(mse))
        print("Mean Absolute Error: {}".format(mae))
        print("Maximum Error: {}".format(max_err))
        print("Variance Score: {}".format(variance))
        print("Mean absolute percentage error: {}".format(mean_percentage))

    lin_reg_eval()

    return opt_feat_count
    
features = pd.read_pickle("features.pkl")
pd.set_option('display.max_columns', None)

y = features["diameter"]
X = features.drop("diameter",  axis=1)

feat_sel(X,y)
