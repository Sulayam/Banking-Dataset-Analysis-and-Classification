## import libraries & data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

train = pd.read_csv('Train-Set.csv')
test = pd.read_csv('Test-Set.csv')
test_ids = test['id']

train.isna().sum()

sns.heatmap(train.isna())

sns.heatmap(test.isna())

for i, j in enumerate(train['day']):
    if j in ['apr', 'aug', 'may', 'nov' , 'sep' , 'jul', 'mar', 'dec', 'jun', 'oct']:
        train['month'][i] = j


for i, j in enumerate(test['day']):
    if j in ['apr', 'aug', 'may', 'nov' , 'sep' , 'jul', 'mar', 'dec', 'jun', 'oct']:
        test['month'][i] = j


train.drop(['balance', 'Unnamed: 0', 'id', 'day'], axis=1, inplace=True)
test.drop(['balance', 'Unnamed: 0', 'id', 'day'], axis=1, inplace=True)

## Encoding

numerical_columns = train.select_dtypes(include=['number']).columns.tolist()
categorical_columns = train.select_dtypes(include=['object']).columns.tolist()
numerical_columns_test = test.select_dtypes(include=['number']).columns.tolist()
categorical_columns_test = test.select_dtypes(include=['object']).columns.tolist()

# one hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encodings = encoder.fit_transform(train[categorical_columns_test])
encodings_df = pd.DataFrame(encodings, columns=encoder.get_feature_names_out())

train_data_encoded = pd.concat([train, encodings_df], axis=1 )
train_data_encoded  = train_data_encoded.drop(categorical_columns, axis=1)

encoding_test = encoder.transform(test[categorical_columns_test])
encoding_test = pd.DataFrame(encoding_test, columns=encoder.get_feature_names_out())
test_data_encoded = pd.concat([test, encoding_test], axis=1 )
test_data_encoded = test_data_encoded.drop(categorical_columns_test, axis=1)

train_data_encoded['Target'] = train['Target']
# train_data_encoded = pd.get_dummies(train, columns=categorical_columns_test, drop_first=True)
# test_data_encoded = pd.get_dummies(test, columns=categorical_columns_test, drop_first=True)


train_data_encoded.shape , test_data_encoded.shape

from sklearn.model_selection import train_test_split, GridSearchCV,  RandomizedSearchCV
X = train_data_encoded.drop(['Target'],axis=1)
y = train_data_encoded['Target'].map({'no':0, 'yes':1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=50)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# UnderSampling

from imblearn.under_sampling import RandomUnderSampler
# Apply RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2,random_state=50)

model = LogisticRegression(solver='liblinear', C=1.0, random_state=0)  # Common parameters
model.fit(X_train, y_train)
pred_logistic = model.predict(X_test)

print('Logistic Regression')
print("precision_score", precision_score(y_test, pred_logistic, pos_label=1))
print("recall_score", recall_score(y_test, pred_logistic , pos_label=1))
print("accuracy_score", accuracy_score(y_test, pred_logistic ))
print("F1 score:", f1_score(y_test, pred_logistic , pos_label=1))

from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
pred_svm = clf.predict(X_test)

print('svm')
print("precision_score", precision_score(y_test, pred_svm, pos_label=1))
print("recall_score", recall_score(y_test, pred_svm , pos_label=1))
print("accuracy_score", accuracy_score(y_test, pred_svm ))
print("F1 score:", f1_score(y_test, pred_svm , pos_label=1))

from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)

pred_hgbc = clf.predict(X_test)

print('HGBC')
print("precision_score", precision_score(y_test, pred_hgbc, pos_label=1))
print("recall_score", recall_score(y_test, pred_hgbc , pos_label=1))
print("accuracy_score", accuracy_score(y_test, pred_hgbc ))
print("F1 score:", f1_score(y_test, pred_hgbc , pos_label=1))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=50, random_state=0).fit(X_train, y_train)

pred_rf = clf.predict(X_test)

print('Random Forest')
print("precision_score", precision_score(y_test, pred_rf, pos_label=1))
print("recall_score", recall_score(y_test, pred_rf , pos_label=1))
print("accuracy_score", accuracy_score(y_test, pred_rf ))
print("F1 score:", f1_score(y_test, pred_rf , pos_label=1))

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
base_clf = DecisionTreeClassifier()
bagging_clf = BaggingClassifier(base_estimator=base_clf, n_estimators=150, random_state=42)
bagging_clf.fit(X_train, y_train)
y_pred = bagging_clf.predict(X_test)

print('Decision Tree')
print("precision_score", precision_score(y_test, y_pred, pos_label=1))
print("recall_score", recall_score(y_test, y_pred , pos_label=1))
print("accuracy_score", accuracy_score(y_test, y_pred ))
print("F1 score:", f1_score(y_test, y_pred , pos_label=1))

!pip install xgboost

import xgboost as xgb
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',  # Specify multiclass classification
    'max_depth': 50,                # Maximum depth of a tree
    'eta': 0.02,                    # Learning rate
    'seed': 42                     # Random seed
}
num_rounds = 50
bst = xgb.train(params, dtrain, num_rounds)

y_pred = bst.predict(dtest)
y_preds=[1 if x>=0.5 else 0 for x in y_pred]

print('xgboost')
print("precision_score", precision_score(y_test, y_preds, pos_label=1))
print("recall_score", recall_score(y_test, y_preds , pos_label=1))
print("accuracy_score", accuracy_score(y_test, y_preds ))
print("F1 score:", f1_score(y_test, y_preds , pos_label=1))

#Save Model
predictions = clf.predict(test_data_encoded)
y_preds=[1 if x>=0.5 else 0 for x in predictions]

results=pd.DataFrame({'id': test_ids, 'Target':y_preds})
results['Target'] = results['Target']
results.to_csv('sub1_undersample_xgboost_withMonth.csv',index=False)

# EDA of undersampling

X = train.drop(['Target'],axis=1)
y = train['Target'].map({'no':0, 'yes':1})

from imblearn.under_sampling import RandomUnderSampler
# Apply RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2,random_state=50)

# countplots for the categorical columns

categorical_columns = ['job', 'marital', 'education', 'default',  'housing', 'loan', 'contact', 'poutcome']


x_cat = 2
y_cat = 4

fig, ax = plt.subplots(x_cat, y_cat, figsize=(25, 10))

colors_cat = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
           '#e377c2', '#7f7f7f']

for i in range(x_cat):
    for j in range(y_cat):
        column = categorical_columns[i*y_cat + j]
        sns.countplot(x=column, data=train, ax=ax[i, j], color=colors_cat[i*y_cat + j])
        ax[i, j].set_title(f'Distribution of {column}')
        ax[i, j].set_xlabel('')
        ax[i, j].set_ylabel('Count')

plt.show()
