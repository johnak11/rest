import pandas as pd
import pickle
from category_encoders import *
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

train_df = pd.read_csv("50_Startups.csv")

# print(train_df.head(4))

enc = OrdinalEncoder().fit(train_df)

df_train_encoded = enc.transform(train_df)

# print(df_train_encoded.head(5))

corremat = df_train_encoded.corr()
top_corr_features = corremat.index[(corremat['Profit'])> .1]
# plt.figure(figsize=(10,10))
# g = sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap='viridis',linewidths=.5)
# print(g)

X = df_train_encoded.iloc[:, :-1].values

y = df_train_encoded.iloc[:, 4].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)


from sklearn.metrics import r2_score
# print(r2_score(y_test,y_pred))


# Comparing the actual with predicted one

df =pd.DataFrame(data=y_test,columns=['y_test'])
df['y_pred'] = y_pred
# print(df)

# print(regressor.predict([[165349.20,	136897.80,	471784.10,1]]))


pickle.dump(regressor,open('startup_prediction.pkl','wb'))