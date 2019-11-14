import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import(LogisticRegressionCV,RidgeCV,LassoCV,LarsCV,LassoLarsIC,BayesianRidge)
from sklearn.feature_selection import RFE,f_regression,SelectFromModel,SelectKBest,RFECV,GenericUnivariateSelect,f_classif,SelectFdr, SelectFwe
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,AdaBoostRegressor
from minepy import MINE
from scipy import stats

# load data
rawdata=pd.read_csv("C:/ckme136/default of credit card clients.csv",skiprows=1)
rawdata.columns.tolist()
rawdata.columns=['ID','LMT_B','SEX','EDU','MARRG','AGE','PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','B_AMT1','B_AMT2','B_AMT3','B_AMT4','B_AMT5','B_AMT6',\
                  'P_AMT1','P_AMT2','P_AMT3','P_AMT4','P_AMT5','P_AMT6','DEFAULT']

for col in ['SEX','EDU','MARRG','PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','DEFAULT']:
    rawdata[col]=rawdata[col].astype('category')    

traindata=rawdata.drop("ID",1)
catcol=['SEX','EDU','MARRG','PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
numcol=['LMT_B','AGE','B_AMT1','B_AMT2','B_AMT3','B_AMT4','B_AMT5','B_AMT6','P_AMT1','P_AMT2','P_AMT3','P_AMT4','P_AMT5','P_AMT6']

x=traindata.drop("DEFAULT",1)
y=traindata["DEFAULT"]


## scale data
x[numcol]=pd.DataFrame(preprocessing.scale(x[numcol]),columns=numcol)

                    
##feature selection average ranking_

names=x.columns

ranks = {}
def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))
    
 

lgrCV=LogisticRegressionCV(random_state=0, cv=5)
lgrCV.fit(x, y)
ranks["Log_CV"] = rank_to_dict(np.abs(lgrCV.coef_).flatten(), x.columns)

ridgeCV =RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
ridgeCV.fit(x, y)
ranks["RidgeCV"] = rank_to_dict(np.abs(ridgeCV.coef_), x.columns)
 

lassoCV=LassoCV(cv=5, random_state=0)
lassoCV.fit(x,y)
ranks["LassoCV"] = rank_to_dict(np.abs(lassoCV.coef_), x.columns)

larsCV=LarsCV(cv=5)
larsCV.fit(x,y)
ranks["LarsCV"]=rank_to_dict(np.abs(larsCV.coef_), x.columns)

lassolarsIC=LassoLarsIC(criterion='bic')
lassolarsIC.fit(x,y)
ranks["LassoLarsIC"]=rank_to_dict(np.abs(lassolarsIC.coef_), x.columns)

bayesridge=BayesianRidge()
bayesridge.fit(x,y)
ranks["BayesRidge"]=rank_to_dict(np.abs(bayesridge.coef_), x.columns)


rfe = RFE(LogisticRegression(),n_features_to_select=5)
rfe.fit(x,y)
ranks["RFE"] = rank_to_dict(rfe.ranking_, names, order=-1)

kbest=SelectKBest(f_classif,k=20)
kbest.fit(x,y)
ranks["KBEST"]=rank_to_dict(np.abs(kbest.scores_), x.columns)
 

ab = AdaBoostRegressor(random_state=0)
ab.fit(x,y)
ranks["ET"]= rank_to_dict(ab.feature_importances_, x.columns)


et = ExtraTreesRegressor(random_state=0)
et.fit(x, y)
ranks["ET"]= rank_to_dict(et.feature_importances_, x.columns)

rf = RandomForestRegressor(random_state=0)
rf.fit(x,y)
ranks["RF"] =rank_to_dict(rf.feature_importances_, x.columns)

gb=GradientBoostingRegressor(random_state=0)
gb.fit(x,y)
ranks["GB"]=rank_to_dict(gb.feature_importances_, x.columns)
 
rfe = RFE(LogisticRegression(),n_features_to_select=5)
rfe.fit(x,y)
ranks["RFE"] = rank_to_dict(rfe.ranking_, names, order=-1)


r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())

ranks["Mean"] = r
methods.append("Mean")
 
print ("\t%s" % "\t".join(methods))
for name in names:
    print ("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods]))))  


