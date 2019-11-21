# import pandas as pd
# import numpy as np
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier

# dataset = pd.read_csv("train.csv")

# for label in dataset.columns:
#     dataset[label].replace('', np.nan, inplace=True) # null cells become nan
#     dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])
    
# X = dataset.drop(['Attribute23'],axis=1)
# Y = dataset['Attribute23']
# #model = DecisionTreeClassifier(criterion='entropy',max_depth=1)
# #AdaBoost = AdaBoostClassifier(base_estimator= model,n_estimators=400,learning_rate=1)
# AdaBoost = AdaBoostClassifier(n_estimators=400,learning_rate=1,algorithm='SAMME')
# AdaBoost.fit(X,Y)
# prediction = AdaBoost.score(X,Y)
# print('The accuracy is: ',prediction*100,'%')


import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.preprocessing import StandardScaler # used for feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split# used for splitting training and testing data
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cluster import DBSCAN
from sklearn.utils import resample
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import csv
from scipy import stats
np.random.seed(1)


# read the train and test data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 方法二：直接丟棄有空值的data
train_data = train_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
print(train_data)

# split out the result column and delete it from train data
y_train = train_data["Attribute23"]
train_data.drop(labels="Attribute23", axis=1, inplace=True)
# categorial data("YES", "NO") to (1, 0)
labelencoder_Y = LabelEncoder()
y_train = labelencoder_Y.fit_transform(y_train)
print(y_train)

# train and test all together
full_data = train_data.append(test_data)
print("train data size = ", len(train_data))
print("full data size = ", len(full_data))
print(full_data)

# delete irrelevant column
drop_columns = ["Attribute1"]
full_data.drop(labels=drop_columns, axis=1, inplace=True)
print(full_data)

# categorial(text) data convert to number that model can use
textColumn = ["Attribute8", "Attribute10", "Attribute11", "Attribute22"]
full_data = pd.get_dummies(full_data)

# 方法一：null cell fill in mean
# full_data.fillna(value=full_data.mean(), inplace=True)

# finish preprocessing, split data
X_train = full_data.iloc[0:6884, :]
X_test = full_data.iloc[6884:, :]
print(X_train)
# X_train = full_data.values[0:6884]
# X_test = full_data.values[6884:]


# normalize
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(data=X_train)
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(data=X_test)
print(X_train)
print(X_test)

# feature selection(高維度 --> 低維度)
#apply SelectKBest class to extract top 10 best features
# bestfeatures = SelectKBest(score_func=chi2, k=23)
# fit = bestfeatures.fit(X_train,y_train)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X_train.columns)
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nsmallest(56,'Score').iloc[:, 0].values)  #print 60 lowest features
# print(featureScores.nlargest(10,'Score').iloc[:, 0].values) #print 6 best features
# irrelevant = featureScores.nsmallest(56,'Score').iloc[:, 0].values
# X_train.drop(labels=irrelevant, axis=1, inplace=True)
# X_test.drop(labels=irrelevant, axis=1, inplace=True)

# outlier detection and removal
# clf = IsolationForest( behaviour = 'new', max_samples=80, random_state = 1, contamination= 'auto')
# preds = clf.fit_predict(X_train)
# print(preds)
# z = stats.zscore(X_train)
# X_train[(np.abs(z) < 3).all(axis=1)]
# y_train[(np.abs(z) < 3).all(axis=1)]

# imbalancing data

# 方法1：很吃low dimension
# sm = SMOTE(random_state=27, ratio=1.0)
# X_train, y_train = sm.fit_sample(X_train, y_train)
# X_train = pd.DataFrame(data=X_train)
# print(X_train)

# 方法組：生成Yes Data或刪除No Data
# concatenate our training data back together
y_train = pd.DataFrame(data=y_train, columns=["Ans"])
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
not_fraud = X[X.Ans==0]
fraud = X[X.Ans==1]

# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

not_fraud_downsampled = resample(not_fraud,
                                replace = False, # sample without replacement
                                n_samples = len(fraud), # match minority n
                                random_state = 27) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([not_fraud_downsampled, fraud])

X_train = downsampled.iloc[:, :-1]
y_train = downsampled.iloc[:, -1]

print(X_train)
print(y_train)

# 方法1：AdaBoost(0.87 -> 0.71732) (n_estimators = 200 -> 0.72948) (ne = 200, dimension = 10, without outlier remove --> 0.75987)
# AdaBoost = AdaBoostClassifier(n_estimators=200,learning_rate=1.0,algorithm='SAMME.R')
# AdaBoost.fit(X_train, y_train)
# prediction = AdaBoost.predict(X_test)
# score = AdaBoost.score(X_train, y_train)
# 方法2：GradientBoost(0.89 -> 0.70516)
# gb_clf2 = GradientBoostingClassifier(n_estimators=800, learning_rate=0.5, max_features=3, max_depth=2, random_state=0)
# gb_clf2.fit(X_train, y_train)
# prediction = gb_clf2.predict(X_test)
# score = gb_clf2.score(X_train, y_train)

# 方法3：隨機森林(1.0 -> 0.73252 : n_e = 75(without outlier detection) , n_e lower no enhance)
clf = RandomForestClassifier(n_estimators=250,max_features="auto",criterion="entropy", bootstrap=True)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
score = clf.score(X_train, y_train)
print(prediction)
print(score)

# 方法4：voting
# clf1 = LogisticRegression(random_state=1)
# clf2 = RandomForestClassifier(n_estimators=200, max_features=3, random_state=1)
# clf3 = AdaBoostClassifier(n_estimators=400,learning_rate=1.0,algorithm='SAMME.R')
# clf4 = GradientBoostingClassifier(n_estimators=400, learning_rate=0.5, max_features=4, max_depth=2, random_state=0)
# clf5 = GaussianNB()

# eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf5)], voting='hard')  # 无权重投票
# # eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('ada', clf3), ('gb', clf4), ('gnb', clf5)],voting='soft', weights=[1,2,3,2,1]) # 权重投票

# eclf.fit(X_train, y_train)
# prediction = eclf.predict(X_test)
# score = eclf.score(X_train, y_train)






# 開啟輸出的 CSV 檔案
with open('output4.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    # 寫入一列資料
    writer.writerow(['id', 'ans'])
    # 寫入另外幾列資料
    for i in range(len(prediction)):
        writer.writerow([float(i), prediction[i]])

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# state = 12  
# test_size = 0.30  
  
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=state)

# gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
# gb_clf2.fit(X_train, y_train)
# predictions = gb_clf2.predict(X_val)

# print("Confusion Matrix:")
# print(confusion_matrix(y_val, predictions))

# print("Classification Report")
# print(classification_report(y_val, predictions))

# print(predictions)