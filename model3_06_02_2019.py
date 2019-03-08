# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 18:42:31 2018

@author: Rudrajit Chanda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split



bank=pd.read_csv(r"C:\Users\Rudra\Documents\Python Scripts\1stLevel\bank_market.csv")


######### Common Functions #########


from sklearn.preprocessing import LabelEncoder
#def Data_Conversion(input_data_set):
#    labelencoder = LabelEncoder()
#    input_data_set['job']=labelencoder.fit_transform(input_data_set['job'])
#    input_data_set['marital']=labelencoder.fit_transform(input_data_set['marital'])
#    input_data_set['education']=labelencoder.fit_transform(input_data_set['education'])
#    input_data_set['default']=labelencoder.fit_transform(input_data_set['default'])
#    input_data_set['housing']=labelencoder.fit_transform(input_data_set['housing'])
#    input_data_set['loan']=labelencoder.fit_transform(input_data_set['loan'])
#    input_data_set['month']=input_data_set['month'].replace(['may',	'jun',	'jul',	'aug',	'oct',	'nov',	'dec',	'jan',	'feb',	'mar',	'apr',	'sep',],[5,6,7,8,10,11,12,1,2,3,4,9])
#    input_data_set['poutcome']=labelencoder.fit_transform(input_data_set['poutcome'])
#    input_data_set['contact']=labelencoder.fit_transform(input_data_set['contact'])
#    input_data_set['y']=labelencoder.fit_transform(input_data_set['y'])
#    return input_data_set

def Data_Conversion(input_data_set):
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    
    input_data_set['job'] = input_data_set['job'].map({'unknown': 0, 'housemaid': 1, 'entrepreneur': 2, 'self-employed':3,
                  'unemployed':4,'student':5, 'services':6,'retired':7, 'admin.':8,'blue-collar':9,'technician':10, 'management':11} ).astype(int)

    input_data_set['marital'] = input_data_set['marital'].map({'divorced':0,'single':1,'married':2}).astype(int)
 
    input_data_set['education'] = input_data_set['education'].map({'unknown':0,'primary':1,'tertiary':2,'secondary':3}).astype(int)
     
    input_data_set['default']=labelencoder.fit_transform(input_data_set['default'])
    input_data_set['housing']=labelencoder.fit_transform(input_data_set['housing'])
    input_data_set['loan']=labelencoder.fit_transform(input_data_set['loan'])
    input_data_set['month']=input_data_set['month'].replace(['may',	'jun',	'jul',	'aug',	'oct',	'nov',	'dec',	'jan',	'feb',	'mar',	'apr',	'sep',],[5,6,7,8,10,11,12,1,2,3,4,9])
    input_data_set['poutcome']=labelencoder.fit_transform(input_data_set['poutcome'])
    input_data_set['contact']=labelencoder.fit_transform(input_data_set['contact'])
    input_data_set['y']=labelencoder.fit_transform(input_data_set['y'])
     
    return input_data_set

def ROC_AUC_check(predicted_data,y_data):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_data, predicted_data)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate(Sensitivity)')
    plt.xlabel('False Positive Rate(Specificity)')
    plt.show()
    #Area under Curve-AUC
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(roc_auc)
    return roc_auc
    

def vif_cal(input_data, dependent_col):
    import statsmodels.formula.api as sm
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    print(xvar_names)
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)

def confusion_matrix_and_accuracy(predicted_data,y_data):  
    from sklearn.metrics import confusion_matrix      
    cm1 = confusion_matrix(y_data,predicted_data)
    print(cm1)
    
    total1=sum(sum(cm1))
    print(total1)
    
    accuracy=(cm1[0,0]+cm1[1,1])/total1
    print("Accuracy=",accuracy*100)
    
    sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Sensitivity : ', sensitivity*100 )
    
    specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('Specificity : ', specificity*100)
    
    return accuracy*100,sensitivity*100,specificity*100

def ResultListCreation(model_type,acc,aoc,test_acc,features,result,model):
            Result=pd.DataFrame([[model_type,acc[0],acc[1],acc[2],aoc,test_acc[0],test_acc[1],test_acc[2],features,model]],columns=['Model Type','Accuracy','Sensitivity','Specificity','AOC','Test_acc','Test_sens','Test_spec','Features','Model'])
            return result.append(Result)
        
def DecisionTreeView(clf,data):
        from IPython.display import Image
        from sklearn.externals.six import StringIO
        import pydotplus
        
        dot_data = StringIO()
        tree.export_graphviz(clf,
                             out_file = dot_data,
                             feature_names = data.columns,
                             filled=True, rounded=True,
                             impurity=False)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        Image(graph.create_png())     

def GraphFrameSize(dim= [8.0, 6.0],v=''):
    if v.upper()=='NEW':       
        plt.rcParams["figure.figsize"] = dim        
    else:
        plt.rcParams["figure.figsize"] = dim
         
#########**********##########

#### Data Analysis ####

#jet=plt.get_cmap('jet')

print('############Data Analysis#############')

bank['y']=LabelEncoder().fit_transform(bank['y'])

for a in bank.columns:
    print('----',a,'----')
    print(bank[a].describe())
    print('')

grp0=bank.groupby('education')
s=grp0['y'].agg(['sum','count'])

plt.bar(np.unique(bank['education']),s['sum'][0:], label="Y", color='g')
plt.legend()
plt.xlabel('education')
plt.ylabel('Y')
plt.title('education vs Y')
plt.show()

plt.bar(np.unique(bank['education']),s['count'][0:], label="Y", color='g')
plt.legend()
plt.xlabel('education')
plt.ylabel('count of customer')
plt.title('education vs count of customer')
plt.show()

s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)
#No outlier based on Jeducation


GraphFrameSize([19,7],'new')
grp2=bank.groupby('job')
s=grp2.agg(['sum','count'])['y']

plt.bar(np.unique(bank['job']),s['sum'][0:], label="Y", color='g')
plt.legend()
plt.xlabel('Job')
plt.ylabel('Y')
plt.title('Job vs Y')
plt.show()

plt.bar(np.unique(bank['job']),s['count'][0:], label="Y", color='g')
plt.legend()
plt.xlabel('Job')
plt.ylabel('count')
plt.title('Job vs count')
plt.show()

s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)
#No outlier based on Job


GraphFrameSize()
grp2=bank.groupby('marital')
s=grp2['y'].agg(['sum','count'])

plt.bar(np.unique(bank['marital']),s['sum'][0:], label="Y", color='g')
plt.legend()
plt.xlabel('marital')
plt.ylabel('Y')
plt.title('marital vs Y')
plt.show()

plt.bar(np.unique(bank['marital']),s['count'][0:], label="Y", color='g')
plt.legend()
plt.xlabel('marital')
plt.ylabel('count')
plt.title('marital vs count')
plt.show()

s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)
#No outlier based on marital status


grp2=bank.groupby('default')
s=grp2['y'].agg(['sum','count'])

plt.bar(np.unique(bank['default']),s['sum'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('default')
plt.ylabel('Y')
plt.title('default vs Y')
plt.show()

plt.bar(np.unique(bank['default']),s['count'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('default')
plt.ylabel('count')
plt.title('default vs count')
plt.show()

s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)


grp2=bank.groupby('loan')
s=grp2['y'].agg(['sum','count'])

plt.bar(np.unique(bank['loan']),s['sum'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('loan')
plt.ylabel('Y')
plt.title('loan vs Y')
plt.show()

plt.bar(np.unique(bank['loan']),s['count'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('loan')
plt.ylabel('count')
plt.title('loan vs count')
plt.show()

s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)



grp2=bank.groupby('housing')
s=grp2['y'].agg(['sum','count'])

plt.bar(np.unique(bank['housing']),s['sum'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('housing')
plt.ylabel('Y')
plt.title('housing vs Y')
plt.show()

plt.bar(np.unique(bank['housing']),s['count'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('housing')
plt.ylabel('count')
plt.title('housing vs count')
plt.show()

s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)



grp2=bank.groupby('contact')
s=grp2['y'].agg(['sum','count'])

plt.bar(np.unique(bank['contact']),s['sum'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('contact')
plt.ylabel('Y')
plt.title('contact vs Y')
plt.show()

plt.bar(np.unique(bank['contact']),s['count'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('contact')
plt.ylabel('count')
plt.title('contact vs count')
plt.show()

s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)



grp=bank.groupby('month')
s=grp['y'].agg(['sum','count'])
l=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
s['sum'][l]

plt.bar(l,s['sum'][l], label="Y", color='g')
plt.legend()
plt.xlabel('Month')
plt.ylabel('Y')
plt.title('Month vs Y')
plt.show()

plt.bar(l,s['count'][l], label="Y", color='g')
plt.legend()
plt.xlabel('Month')
plt.ylabel('count')
plt.title('Month vs count')
plt.show()

s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)



grp3=bank.groupby('day')
s=grp3['y'].agg(['sum','count'])

plt.bar(np.unique(bank['day']),s['sum'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('day')
plt.ylabel('Y')
plt.title('day vs Y')
plt.show()

plt.bar(np.unique(bank['day']),s['count'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('day')
plt.ylabel('count')
plt.title('day vs count')
plt.show()

s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)



grp3=bank.groupby('poutcome')
s=grp3['y'].agg(['sum','count'])

plt.bar(np.unique(bank['poutcome']),s['sum'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('poutcome')
plt.ylabel('Y')
plt.title('poutcome vs Y')
plt.show()

plt.bar(np.unique(bank['poutcome']),s['count'][0:], label="Y", color='g')
plt.legend(s)
plt.xlabel('poutcome')
plt.ylabel('Y')
plt.title('poutcome vs count')
plt.show()

s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)



print('Y vs duration')
s=bank.groupby(pd.cut(bank["duration"], np.arange(0,5500, 500))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)
s=bank.groupby(pd.cut(bank["duration"], np.arange(0,500, 50))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)

print('Y vs Age')
s=bank.groupby(pd.cut(bank["age"], np.arange(-1,100, 10))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)
s=bank.groupby(pd.cut(bank["age"], np.arange(9,20, 1))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)
s=bank.groupby(pd.cut(bank["age"], np.arange(79,90, 1))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)


print('Y vs balance')
s=bank.groupby(pd.cut(bank["balance"], np.arange(-7999,140000, 20000))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)


print('Y vs campaign')
s=bank.groupby(pd.cut(bank["campaign"], np.arange(0,65, 5))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)


print('Y vs pdays')
s=bank.groupby(pd.cut(bank["pdays"], np.arange(-2,900, 100))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)
s=bank.groupby(pd.cut(bank["pdays"], np.arange(-2,900, 30))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)
s=bank.groupby(pd.cut(bank["pdays"], np.arange(-2,29, 1))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)

print('Y vs previous')
s=bank.groupby(pd.cut(bank["previous"], np.arange(-1,300, 30))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)
s=bank.groupby(pd.cut(bank["previous"], np.arange(-1,29, 1))).agg(['sum','count'])['y']
s['percentage of turn up']=(s['sum']/s['count'])*100
print(s)


#Age vs campaign
plt.scatter(bank['age'], bank['campaign'], marker='o');
plt.xlabel('age')
plt.ylabel('campaign')
plt.title('age vs campaign')
plt.show()
#duration vs campaign
plt.scatter(bank['duration'], bank['campaign'], marker='o');
plt.xlabel('duration')
plt.ylabel('campaign')
plt.title('duration vs campaign')
plt.show()
#education vs campaign
plt.scatter(bank['education'], bank['campaign'], marker='o');
plt.xlabel('education')
plt.ylabel('campaign')
plt.title('education vs campaign')
plt.show()
#pdays vs campaign
plt.scatter(bank['pdays'], bank['campaign'], marker='o');
plt.xlabel('pdays')
plt.ylabel('campaign')
plt.title('pdays vs campaign')
plt.show()
#pdays vs campaign
plt.scatter(bank['previous'], bank['campaign'], marker='o');
plt.xlabel('previous')
plt.ylabel('campaign')
plt.title('previous vs campaign')
plt.show()
#poutcome vs campaign
plt.scatter(bank['poutcome'], bank['campaign'], marker='o');
plt.xlabel('poutcome')
plt.ylabel('campaign')
plt.title('poutcome vs campaign')
plt.show()
#poutcome vs campaign
plt.scatter(bank['balance'], bank['campaign'], marker='o');
plt.xlabel('balance')
plt.ylabel('campaign')
plt.title('balance vs campaign')
plt.show()

#balance vs duration
plt.scatter(bank['balance'], bank['duration'], marker='o');
plt.xlabel('balance')
plt.ylabel('duration')
plt.title('balance vs duration')
plt.show()
#pdays vs duration
plt.scatter(bank['pdays'], bank['duration'], marker='o');
plt.xlabel('pdays')
plt.ylabel('duration')
plt.title('pdays vs duration')
plt.show()
#previous vs duration
plt.scatter(bank['previous'], bank['duration'], marker='o');
plt.xlabel('previous')
plt.ylabel('duration')
plt.title('previous vs duration')
plt.show()
#age vs duration
plt.scatter(bank['age'], bank['duration'], marker='o');
plt.xlabel('age')
plt.ylabel('duration')
plt.title('age vs duration')
plt.show()

#balance vs age
plt.scatter(bank['age'], bank['balance'], marker='o');
plt.xlabel('age')
plt.ylabel('balance')
plt.title('age vs balance')
plt.show()

#previous vs age
plt.scatter(bank['age'], bank['previous'], marker='o');
plt.xlabel('age')
plt.ylabel('previous')
plt.title('age vs previous')
plt.show()
#pdays vs age
plt.scatter(bank['age'], bank['pdays'], marker='o');
plt.xlabel('age')
plt.ylabel('pdays')
plt.title('age vs pdays')
plt.show()


#Cust_num vs balance
plt.scatter(bank['Cust_num'], bank['balance'], marker='o');
plt.xlabel('Cust_num')
plt.ylabel('balance')
plt.title('Cust_num vs balance')
plt.show()
#Cust_num vs age
plt.scatter(bank['Cust_num'], bank['age'], marker='o');
plt.xlabel('Cust_num')
plt.ylabel('age')
plt.title('Cust_num vs age')
plt.show()

GraphFrameSize([19,7],'new')
plt.scatter(bank['job'], bank['balance'], marker='o');
plt.xlabel('job')
plt.ylabel('balance')
plt.title('job vs balance')
plt.show()


GraphFrameSize()
plt.scatter(bank['education'], bank['balance'], marker='o');
plt.xlabel('education')
plt.ylabel('balance')
plt.title('education vs balance')
plt.show()

GraphFrameSize()
plt.scatter(bank['marital'], bank['balance'], marker='o');
plt.xlabel('marital')
plt.ylabel('balance')
plt.title('marital vs balance')
plt.show()


bank['age'].quantile([0,0.25,0.50,0.75,1])

#Previous 275 may be an outlier
#duration 4918 may be an outlier
#age above 70 may be outlier; bank[bank['age']>70].shape
#balance above 25000 may be outlier; bank[bank['balance']>25000].shape
#Pdays avove 400 may be outlier
#bank=bank[bank['balance']<25000]
#s=bank[bank['balance']<25000] #valid
#s=s[s['age']<=70]
#s=s[s['pdays']<400]
#s=s[s['campaign']<15]
#s[s['previous']>=50]
#s.shape
#s=s[s['duration']!=0]
#s=s[s['duration']!=4918]

#bank[bank['campaign']>=15].shape


print('############****Data Analysis End***#############')
#### ************** ####

bank_market=pd.read_csv(r"C:\Users\Rudra\Documents\Python Scripts\1stLevel\bank_market.csv")
bank_market=bank_market.drop(['Cust_num'],axis=1)
bank_market= Data_Conversion(bank_market)
      
############ Variable significance finding###############
print('############****VIF Calculation***#############')
#Calculating VIF values using that function
vif_cal(input_data=bank_market, dependent_col="y")
vif_cal(input_data=bank_market.drop(['poutcome'],axis=1), dependent_col="y")
#bank_market=bank_market.drop(['poutcome'],axis=1)
print('############****VIF Calculation End***#############')
#########*******************************################      
      
      
print('############Train, Test, Hold Split#############')
#bank_market=pd.read_csv(r"C:\Users\Rudra\Documents\Python Scripts\1stLevel\bank_market_cre.csv")

bank_market_train, bank_market_hold, y_bank_market_train,y_bank_market_hold = train_test_split(bank_market,bank_market['y'], test_size=0.05,random_state=2)


bank_market_train=bank_market_train[bank_market_train['duration']!=0]
bank_market_train=bank_market_train[bank_market_train['duration']!=4918]

bank_market_train=bank_market_train[bank_market_train['campaign']<=15]
bank_market_train=bank_market_train[bank_market_train['previous']<=59]




bank_market_train, bank_market_test, y_bank_market_train,y_bank_market_test = train_test_split(bank_market_train.drop(['y'], axis=1),bank_market_train['y'], test_size=0.2,random_state=2)
print('############****Train, Test, Hold Split End***#############')



#######Over Sampling########
#from sklearn.metrics import recall_score
#from imblearn.over_sampling import SMOTE
#
#col=bank_market_train.columns
#
#sm = SMOTE(random_state=2, ratio = 1)
#bank_market_train, y_bank_market_train = sm.fit_sample(bank_market_train, y_bank_market_train)
#
#bank_market_train = pd.DataFrame(bank_market_train, columns = col)
#y_bank_market_train = pd.DataFrame(y_bank_market_train, columns = ['y'])

#z=bank_market_train;
#z['y']=y_bank_market_train
#from pandas import ExcelWriter
#writer = ExcelWriter(r'C:\Users\Rudra\Documents\Python Scripts\1stLevel\compare.xlsx')
#Result1.to_excel(writer,'Sheet1')
#writer.save()

#######***********#########


#Result1=pd.DataFrame()


######### MODEL BULIDING ################################
print('############****Model Building Starts***#############')
         ####### Logistic Regression #######
print('############****Logistic Regression***#############')         
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()

logistic.fit(bank_market_train,y_bank_market_train)

predict1=logistic.predict(bank_market_train)
predict1

acc=confusion_matrix_and_accuracy(predict1,y_bank_market_train)
#-------ROC & AUC------#
aoc=ROC_AUC_check(predict1,y_bank_market_train)        
#------*********------#

#----------Testing-------$
predict2=logistic.predict(bank_market_test)
predict2

test_aoc=confusion_matrix_and_accuracy(predict2,y_bank_market_test)

Result1=ResultListCreation('Logistic',acc,aoc,test_aoc,bank_market_train.columns,Result1,logistic)

from sklearn import cross_validation
bootstrap=cross_validation.ShuffleSplit(n=len(bank_market_train), 
                                    n_iter=10, 
                                    random_state=0)
BS_score = cross_validation.cross_val_score(logistic,bank_market_train, y_bank_market_train,cv=bootstrap)
print(BS_score)

print('BS MEan',BS_score.mean())

print('############****Logistic End***#############')
###### ******************* ########

######  Decision Tree ########
print('############****Decision Tree***#############')
from sklearn import tree        

#clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
#max_features=None, max_leaf_nodes=61,
#min_impurity_decrease=0.0, min_impurity_split=None,
#min_samples_leaf=10, min_samples_split=2,
#min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#splitter='best')

clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', #max_depth=10,
max_features=None, max_leaf_nodes=20,
min_impurity_decrease=0.0, min_impurity_split=None,
#min_samples_leaf=10, min_samples_split=2,
min_weight_fraction_leaf=0.0, presort=False, random_state=None,
splitter='best')

clf.fit(bank_market_train,y_bank_market_train)

predict_tree_trn = clf.predict(bank_market_train)

acc=confusion_matrix_and_accuracy(predict_tree_trn,y_bank_market_train)

#-------ROC & AUC------#
aoc=ROC_AUC_check(predict_tree_trn,y_bank_market_train)    
#------*********------#

#--------Testing-------#
predict2 = clf.predict(bank_market_test)
predict2

test_aoc=confusion_matrix_and_accuracy(predict2,y_bank_market_test)

Result1=ResultListCreation('Decision Tree',acc,aoc,test_aoc,bank_market_train.columns,Result1,clf)

from sklearn import cross_validation
bootstrap=cross_validation.ShuffleSplit(n=len(bank_market_train), 
                                    n_iter=10, 
                                    random_state=0)
BS_score = cross_validation.cross_val_score(clf,bank_market_train, y_bank_market_train,cv=bootstrap)
print(BS_score)

print('BS MEan',BS_score.mean())

from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file = dot_data,
                     feature_names = bank_market_train.columns,
                     filled=True, rounded=True,
                     impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())     
#--------*******-------#

###### ************** ########



##### Random Forest #####
#        RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=9, max_features='auto', max_leaf_nodes=100,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=31,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)

#        RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=11, max_features='auto', max_leaf_nodes=120,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=28,
#            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)

#        RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=10, max_features='auto', max_leaf_nodes=120,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=34,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)
print('############****Decision Tree End***#############')

      
      
print('############****Random Forest***#############')
from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
max_depth=8, max_features='auto', max_leaf_nodes=65,
min_impurity_decrease=0.0, min_impurity_split=None,
#min_samples_leaf=1, 
min_samples_split=34,
min_weight_fraction_leaf=0.0, 
n_estimators=60, n_jobs=1,
oob_score=False, random_state=None, verbose=0,
warm_start=False
)


forest.fit(bank_market_train,y_bank_market_train)
forest_predict=forest.predict(bank_market_train)

acc=confusion_matrix_and_accuracy(forest_predict,y_bank_market_train)

#-------ROC & AUC------#
aoc=ROC_AUC_check(forest_predict,y_bank_market_train)    
#------*********------#

#--------Testing-------#
predict2 = forest.predict(bank_market_test)
predict2
test_aoc=confusion_matrix_and_accuracy(predict2,y_bank_market_test)

Result1=ResultListCreation('Random Forest',acc,aoc,test_aoc,bank_market_train.columns,Result1,forest)




from sklearn import cross_validation
bootstrap=cross_validation.ShuffleSplit(n=len(bank_market_train), 
                                    n_iter=10, 
                                    random_state=0)
BS_score = cross_validation.cross_val_score(forest,bank_market_train, y_bank_market_train,cv=bootstrap)
print(BS_score)

print('BS MEan',BS_score.mean())
print('############****Random Forest End***#############')
         
##### ************* #####

###### GBM #####
print('############****GBM***#############')
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
boost=GradientBoostingClassifier(criterion='friedman_mse', init=None,
  learning_rate=0.1, loss='deviance', max_depth=5,
  max_features=None, max_leaf_nodes=None,
  min_impurity_decrease=0.0, min_impurity_split=None,
  min_samples_leaf=31, min_samples_split=12,
  min_weight_fraction_leaf=0.0, n_estimators=65,
  presort='auto', random_state=None, subsample=1.0, verbose=1,
  warm_start=False)

boost.fit(bank_market_train,y_bank_market_train)
boost_predict=boost.predict(bank_market_train)

acc=confusion_matrix_and_accuracy(boost_predict,y_bank_market_train)

#-------ROC & AUC------#
aoc=ROC_AUC_check(boost_predict,y_bank_market_train)    
#------*********------#

#--------Testing-------#
predict2 = boost.predict(bank_market_test)
predict2
test_aoc=confusion_matrix_and_accuracy(predict2,y_bank_market_test)

Result1=ResultListCreation('GBM',acc,aoc,test_aoc,bank_market_train.columns,Result1,boost)


from sklearn import cross_validation
bootstrap=cross_validation.ShuffleSplit(n=len(bank_market_train), 
                                    n_iter=10, 
                                    random_state=0)
BS_score = cross_validation.cross_val_score(boost,bank_market_train, y_bank_market_train,cv=bootstrap)
print(BS_score)

print('BS MEan',BS_score.mean())
print('############****GBM End***#############')

print('############****Model Building End***#############')                  
                       
######### Test With Hold Out Data #########
print('############****Test With hold Out Data***#############')
#bank_market_real=bank_market=pd.read_csv(r"C:\Users\Rudra\Documents\Python Scripts\1stLevel\bank_market_real.csv")
#bank_market_real= Data_Conversion(bank_market_real)
#bank_market_real_y= bank_market_real['y']
#bank_market_real=bank_market_real.drop(['y'],axis=1)


bank_market_real=bank_market_hold.drop(['y'],axis=1)
bank_market_real_y=y_bank_market_hold
#bank_market_real=bank_market_real[0:2]
#bank_market_real_y=bank_market_real_y[0:2]
 ####Logistic regression####
print('Actual Test - Logistic Regression')
prdeict_log=logistic.predict(bank_market_real)
confusion_matrix_and_accuracy(prdeict_log,bank_market_real_y)
####*******************####

###Tree###
print('\nActual Test - Decision Tree')
predict_re=clf.predict(bank_market_real)
confusion_matrix_and_accuracy(predict_re,bank_market_real_y)
###***###

#### Random Forest ####
print('\nActual Test - Random Forest')
predict_rand=forest.predict(bank_market_real)
confusion_matrix_and_accuracy(predict_rand,bank_market_real_y)
####***************####

####SVM####
print('\nActual Test - GBM')
boost_gbm=boost.predict(bank_market_real)
confusion_matrix_and_accuracy(boost_gbm,bank_market_real_y)
print('############****Test With Hold Out Data End***#############')#
print('############****THANK YOU***#############')
####*******************####
######## ************** #########
