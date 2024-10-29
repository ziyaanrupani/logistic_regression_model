import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = "C:\\Users\\Zee\\Desktop\\School\\BCIT\\COMP 4254 Advance Data\\Data\\amtrack_survey_clean.csv"
df = pd.read_csv(PATH)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print('\n',"Show table:")
print(df.head())
print('\n', "Table stats:")
print(df.describe())

df2 = df.groupby(['Satisfied']).mean()
df2 = df2.T
print("\nTransposed dataframe split between satisfied and unsatisfied customers:\n")
print(df2)

# Boarding experience, Booking experience, Quality Food, Online experience
notSatisfied = [df2[0]['Boarding experience'],
                df2[0]['Booking experience'],
                df2[0]['Quality Food'],
                df2[0]['Online experience'],
                df2[0]['Wifi']]

yesSatisfied = [df2[1]['Boarding experience'],
                df2[1]['Booking experience'],
                df2[1]['Quality Food'],
                df2[1]['Online experience'],
                df2[1]['Wifi']]

# plot
x = np.arange(5)
width = 0.4

plt.bar(x-0.2, notSatisfied, width, color='orange')
plt.bar(x+0.2, yesSatisfied, width, color='green')
plt.xticks(x, ['Boarding Exp', 'Booking Exp', 'Quality Food', 'Online Exp',
               'Wifi'])
plt.xlabel("Features")
plt.ylabel("Score")
plt.legend(["Not Satisfied", "Satisfied"])
plt.show()
