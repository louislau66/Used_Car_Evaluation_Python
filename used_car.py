import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import interactive
interactive(False)

""" 
Logistic Regression Project
   CAR                      car acceptability
   . PRICE                  overall price
   . . buying               buying price
   . . maint                price of the maintenance
   . TECH                   technical characteristics
   . . COMFORT              comfort
   . . . doors              number of doors
   . . . persons            capacity in terms of persons to carry
   . . . lug_boot           the size of luggage boot
   . . safety               estimated safety of the car

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import interactive
interactive(False)

car=pd.read_csv(r"C:\Users\Louis\Documents\Metro college\ML project\car_ver2.csv",header=None)
car.columns=['buying','maint','doors','persons','lug_boot','safety','acceptability']
#sns.set_palette('GnBu_d')
sns.set_style('whitegrid')

#explore dataset
car.info()
car.describe()
#unique values of each attributes
cols=car.columns
for i in cols:
    print(i,car[i].unique())
#convert target variable to binary    
#car.acceptability=car.acceptability.map({'unacc':0, 'acc':1, 'vgood':1, 'good':1})
#target variable distribution
sns.countplot(x='acceptability',data=car)
plt.show()

#Correlations
car1=pd.get_dummies(car,columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
cor=car1.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.Reds,fmt='.2f')
plt.xticks(rotation=20)
plt.show()

#explore relationship between predictors and target variable
def crossfig(var):
    pd.crosstab(car[var],car['acceptability']).plot(kind='bar')
    plt.title('Acceptability by '+var)
    plt.xlabel(var)
    plt.ylabel('Acceptability')
    plt.show()
    
##crossfig('buying')
##crossfig('maint')
##crossfig('doors')
##crossfig('lug_boot')

crossfig('safety')
crossfig('persons')

# Get X & y
X=car1.copy()
X.drop('acceptability',axis=1,inplace=True)
y=car1['acceptability']



#splitting test and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

#import classifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#train model
model.fit(X_train, y_train)

#predict on test set
y_pred = model.predict(X_test)
y_pred_probs = model.predict_proba(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report


accuracy_score(y_test,y_pred)

confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))


precision_score(y_test,y_pred,average='micro')
precision_score(y_test,y_pred,average='macro')

recall_score(y_test,y_pred,average='micro')

#K-fold cross validation
from sklearn.model_selection import cross_val_score
clf = LogisticRegression()
cross_val_score(clf,X,y,cv=4).mean()

#Decision Tree 
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))

confusion_matrix(y_test,predictions)
accuracy_score(y_test,predictions)

ctf = DecisionTreeClassifier()

cross_val_score(ctf,X,y,cv=4).mean()
Out[30]: 0.7754542818195894


