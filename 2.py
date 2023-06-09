import pandas as pd
import numpy as np
import matplotlib.pyplot

fraud = pd.read_csv("E:\Assignment-15\\Fraud_check.csv")

fraud["income"]="<=30000"
fraud.loc[fraud["Taxable.Income"]>=30000,"income"]="Good"
fraud.loc[fraud["Taxable.Income"]<=30000,"income"]="Risky"

fraud.drop(["Taxable.Income"],axis=1,inplace=True)

fraud.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in fraud.columns:
    if fraud[column_name].dtype == object:
        fraud[column_name] = le.fit_transform(fraud[column_name])
    else:
        pass
  
features = fraud.iloc[:,0:5]
labels = fraud.iloc[:,5]

colnames = list(fraud.columns)
predictors = colnames[0:5]
target = colnames[5]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)

from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)

model.estimators_
model.classes_
model.n_features_
model.n_classes_

model.n_outputs_

model.oob_score_
prediction = model.predict(x_train)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
np.mean(prediction == y_train)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)
pred_test = model.predict(x_test)
acc_test =accuracy_score(y_test,pred_test)
