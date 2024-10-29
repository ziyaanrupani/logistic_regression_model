import pandas as pd
import numpy as np

# import csv to dataframe and display max rows
PATH = "C:\\Users\\Zee\\Desktop\\School\\BCIT\\COMP 4254 Advance Data\\Data\\amtrack_survey.csv"
df = pd.read_csv(PATH)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# print rows and investigate missing values
# max rows = 90915
print('\n',"Show table")
print(df.head())
print('\n', "Table stats")
print(df.describe())

# checking null for trip type
dfTest = df['Trip Type']
sumNull = dfTest.isnull().sum()
print('\n', "Null Value for Trip Type " + str(sumNull))

# checking unique for trip type
uniqueTripType = dfTest.unique()
print(uniqueTripType)

# checking null for Gender
dfTest = df['Gender']
sumNull = dfTest.isnull().sum()
print('\n', "Null Value for Gender " + str(sumNull))

# checking unique for Gender
uniqueTripType = dfTest.unique()
print(uniqueTripType)

# checking null for Seat type
dfTest = df['Seat Type']
sumNull = dfTest.isnull().sum()
print('\n', "Null Value for Seat Type " + str(sumNull))

# checking unique for Seat type
uniqueTripType = dfTest.unique()
print(uniqueTripType)

# checking null for Membership
dfTest = df['Membership']
sumNull = dfTest.isnull().sum()
print('\n', "Null Value for Membership " + str(sumNull))

# checking unique for Membership
uniqueTripType = dfTest.unique()
print(uniqueTripType)

# impute function
def imputeNullValues(colName, df):
    # Create two new column names based on original column name.
    indicatorColName = 'm_' + colName  # Tracks whether imputed.
    imputedColName = 'imp_' + colName  # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = df[colName].median()

    # Populate new columns with data.
    imputedColumn = []
    indictorColumn = []

    for i in range(len(df)):
        isImputed = False
        # mi_OriginalName column stores imputed & original data.
        if (np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])
            # mi_OriginalName column tracks if is imputed (1) or not (0).
        if (isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

            # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName] = imputedColumn
    del df[colName]  # Drop column with null values.
    return df

# impute into the dataframe
df = imputeNullValues('Delayed arrival', df)
df = imputeNullValues('Trip Distance', df)

print('\n', "Table stats with impute")
print(df.describe())

# Dummy variables
# convert strings to values in order to look at stats
df = pd.get_dummies(df, columns=['Trip Type', 'Gender', 'Seat Type',
                                 'Membership'])
print('\n', "View table with dummy")
print(df.head())
print('\n', "Table stats with dummy")
print(df.describe())

# convert impute dataframe to csv
df.to_csv('C:\\Users\\Zee\\Desktop\\School\\BCIT\\COMP 4254 Advance Data\\Data\\amtrack_survey_clean.csv', index=False)






