import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


PATH = "C:\\Users\\Zee\\Desktop\\School\\BCIT\\COMP 4254 Advance Data\\Data\\amtrack_survey_clean.csv"
df = pd.read_csv(PATH)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print('\n',"Show table")
print(df.head())
print('\n', "Table stats")
print(df.describe())

# bin feature
df['AgeBin'] = pd.cut(x=df['Age'], bins=[0, 24, 40, 65, 100])
df = pd.get_dummies(df, columns=['AgeBin'])

# set X and y
X = df.copy()
del X['Satisfied']

y = df['Satisfied']

# chi test to get significant values
test = SelectKBest(score_func=chi2, k='all')
chiScores = test.fit(X, y)
np.set_printoptions(precision=5)

print("\nPredictor variables: " + str(X))
print("\nPredictor Chi-Square Scores: " + str(chiScores.scores_))

# 15, 17, 19 cols are not significant in the chi square
del X['m_Delayed arrival']
del X['m_Trip Distance']
del X['Trip Type_2']

# ************************************************
# feature selection of model
X = X.copy()
X = X[['AgeBin_(40, 65]', 'AgeBin_(65, 100]', 'Gender_Male',
       'Booking experience', 'Boarding experience', 'Quality Food',
       'Online experience', 'Wifi',
       'Membership_non-member',  'Seat Type_premium' ]]

print("\nFeature Select")
print(X.head())
print(X.describe())

# ***********************************************

# scale X
sc_x = MinMaxScaler()
X = sc_x.fit_transform(X)

# setup Kfold
# create empty list for stats
count = 0
kfold = KFold(n_splits=10, shuffle=True)
accuracyList = []
precisionList = []
recallList = []
f1List = []

print("\nMetrics and Matrix for each fold:")
print("*****************************")
for train_index, test_index in kfold.split(X):
    # X_train = X.loc[X.index.isin(train_index)]
    # X_test  = X.loc[X.index.isin(test_index)]
    # y_train = y.loc[y.index.isin(train_index)]
    # y_test  = y.loc[y.index.isin(test_index)]

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # logistic model set
    logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',
                                       random_state=0)
    #fit model
    logisticModel.fit(X_train, y_train)
    y_pred = logisticModel.predict(X_test)
    y_prob = logisticModel.predict_proba(X_test)

    # confusion matrix
    cm = pd.crosstab(y_test,y_pred,
                     rownames=['Actual'],
                     colnames=['Predicted'])

    # calculate accuracy, precision, recall, f1
    # put them in respective lists
    accuracy  = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall    = metrics.recall_score(y_test, y_pred)
    f1        = metrics.f1_score(y_test, y_pred)

    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)

    # print the metrics and matrix
    count += 1
    print("Kfold = " + str(count))
    print("\nAccuracy : ", round(accuracy, 5))
    print("Precision: ", round(precision, 5))
    print("Recall   : ", round(recall, 5))
    print("F1       : ", round(f1, 5))
    print("\nConfusion Matrix")
    print(cm)
    print("*****************************")


accuracyAvg = str(round(np.mean(accuracyList), 5))
accuracyStd = str(round(np.std(accuracyList), 5))
precisionAvg = str(round(np.mean(precisionList), 5))
precisionStd = str(round(np.std(precisionList), 5))
recallAvg = str(round(np.mean(recallList), 5))
recallStd = str(round(np.std(recallList), 5))
f1Avg = str(round(np.mean(f1List), 5))
f1Std = str(round(np.std(f1List), 5))

dfStats = pd.DataFrame({"Accuracy":  [accuracyAvg, accuracyStd],
                        "Precision": [precisionAvg, precisionStd],
                        "Recall":[recallAvg, recallStd],
                        "F1": [f1Avg, f1Std]})
dfStats.index = ['Average', 'Std ']
dfStats = dfStats.T
print(dfStats)


