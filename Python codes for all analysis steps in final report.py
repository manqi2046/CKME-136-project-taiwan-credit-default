#!/usr/bin/env python
# coding: utf-8

# In[385]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as ss
import pylab as pl
import math


# In[205]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.cluster import KMeans,AffinityPropagation,MeanShift,SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV, cross_val_score, KFold
from scipy.stats import chi,mode
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm 
from sklearn.svm import LinearSVC,LinearSVR,SVC
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.linear_model import(LogisticRegressionCV,RidgeCV,LassoCV,LarsCV,LassoLarsIC,BayesianRidge,ElasticNetCV,OrthogonalMatchingPursuit,HuberRegressor,ARDRegression,PassiveAggressiveClassifier)
from sklearn.feature_selection import RFE,SelectKBest,f_classif,SelectFdr
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.metrics import accuracy_score


# In[207]:


# 1.Preprocessing data (Descriptive data analysis and data cleaning)
##1.1 Load data and data summury
rawdata=pd.read_csv("/Users/jingjingsun/Desktop/default of credit card clients.csv",skiprows=1)
rawdata.columns.tolist()
rawdata.columns=['ID','LMT_B','SEX','EDU','MARRG','AGE','PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','B_AMT1','B_AMT2','B_AMT3','B_AMT4','B_AMT5','B_AMT6',                  'P_AMT1','P_AMT2','P_AMT3','P_AMT4','P_AMT5','P_AMT6','DEFAULT']

catcol=['SEX','EDU','MARRG','PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','DEFAULT']
numcol=['LMT_B','AGE','B_AMT1','B_AMT2','B_AMT3','B_AMT4','B_AMT5','B_AMT6','P_AMT1','P_AMT2','P_AMT3','P_AMT4','P_AMT5','P_AMT6']
rawdata=rawdata.drop("ID",1)


# In[208]:


###1.1.1 summaries of data
pd.set_option('display.max_columns', 600)
rawdata.info()
rawdata.head(15)


# In[209]:


###1.1.2 check missing values 
check_null=pd.isnull(rawdata)
Nullcheck=pd.DataFrame(check_null.sum(axis = 0, skipna = False))
Nullcheck.T


# In[210]:


##1.2 dealing with the categorical attributes
###1.2.1 check the categorical attributes' values
for x in catcol:
    y=pd.DataFrame(rawdata[x].value_counts().sort_index())
    print(y.T)
##EDUCATION and MARRAGE have some unknown data,need imputation


# In[211]:


###1.2.2 categorical attributes imputation (cleaning data)
data_cln1=rawdata.copy(deep=True)

##For Education: replace values =0,5,6 to 4
for x in [0,5,6]:
    data_cln1.loc[data_cln1['EDU']==x, 'EDU'] = 4

##For Marriage: replace value = 0 to 3
data_cln1.loc[data_cln1['MARRG']==0, 'MARRG'] = 3
   
for x in ["EDU","MARRG"]:
    y=pd.DataFrame(data_cln1[x].value_counts().sort_index())
    print(y)


# In[212]:


###1.2.3 data distrubution of the categorical attributes
#Draw countplot  
sns.set_style("darkgrid")
fig1, ((ax1, ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2,figsize=(15,4))
df1=data_cln1.loc[:,['SEX','EDU','MARRG','DEFAULT']]
axes_1=[ax1, ax2, ax3,ax4]
SEX_label=['Male','Female']
EDU_label=['Grad','Undergrad','HighSchool','Others']
MARRG_label=['Married','Single','Others']
DEFAULT_label=['No','Yes']
xtick_labels_1=[SEX_label,EDU_label,MARRG_label,DEFAULT_label]
for i in range(0,4):
    sns.countplot(x=df1.columns[i],data=df1,ax=axes_1[i])
    xlbl=axes_1[i].get_xlabel()
    axes_1[i].set_xlabel(xlbl,fontsize=10)
    axes_1[i].set_ylabel('COUNT',fontsize=10)
    axes_1[i].set_xticklabels(xtick_labels_1[i])
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.3)
plt.show()


# In[213]:


fig2, ((ax4, ax5),(ax6,ax7),(ax8,ax9)) = plt.subplots(nrows=3, ncols=2,figsize=(20,6))
df2=data_cln1[catcol].drop(['SEX','EDU','MARRG','DEFAULT'],1)
axes_2=[ax4,ax5,ax6,ax7,ax8,ax9]
for i in range(0,6):
    sns.countplot(x=df2.columns[i],data=df2,ax=axes_2[i])
    xlbl=axes_2[i].get_xlabel()
    axes_2[i].set_xlabel(xlbl,fontsize=10)
    axes_2[i].set_ylabel('COUNT',fontsize=10)
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
plt.show()


# In[214]:


###1.2.4 relationship between class attribute and categorical attributes
####(1) one categorical attribute vs class attribute
fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,figsize=(16,4))
df1=data_cln1.loc[:,['SEX','EDU','MARRG','DEFAULT']]
axes1=[ax1,ax2,ax3]
xtick_labels1=[SEX_label,EDU_label,MARRG_label]
for i in range(0,3):
    b=sns.catplot(x=df1.columns[i],y='DEFAULT',data=df1, kind="point",
                    ax=axes1[i])
    axes1[i].set_ylabel('Default Probability',fontsize=12)
    xlbl=axes1[i].get_xlabel()
    axes1[i].set_xlabel(xlbl,fontsize=12)
    axes1[i].set_xticklabels(xtick_labels1[i])
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.4)
for i in range(0,3):
    plt.close(i+2)
plt.show()


# In[215]:



fig2, ((ax4, ax5),(ax6,ax7),(ax8,ax9)) = plt.subplots(nrows=3, ncols=2,figsize=(20,12))
df2=data_cln1[catcol].drop(['SEX','EDU','MARRG'],1)
axes2=[ax4,ax5,ax6,ax7,ax8,ax9]
for i in range(0,6):
    sns.catplot(x=df2.columns[i],y='DEFAULT',data=df2, kind="point",
                    ax=axes2[i])
    axes2[i].set(ylabel='Default Probability')
    xlbl=axes2[i].get_xlabel()
    axes2[i].set_xlabel(xlbl,fontsize=12)
for i in range(0,6):
    plt.close(i+2)
plt.show()


# In[216]:


#####(2)two categorical attribute vs class attribute
###SEX,EDU vs Default
fig1,((ax1,ax2,ax3))=plt.subplots(nrows=1,ncols=3,figsize=(16,4))

Sex_Edu=sns.catplot(x='EDU',y='DEFAULT',hue='SEX',data=data_cln1, kind="point",ax=ax1)
ax1.set(ylabel='Default Probility')
ax1.set_xticklabels(EDU_label)
current_handles, current_labels=ax1.get_legend_handles_labels()
ax1.legend(current_handles,SEX_label,title="SEX")

Sex_Marrg=sns.catplot(x='MARRG',y='DEFAULT',hue='SEX',data=data_cln1, kind="point",ax=ax2)
ax2.set(ylabel='Default Probility')
ax2.set_xticklabels(MARRG_label)
current_handles, current_labels=ax2.get_legend_handles_labels()
ax2.legend(current_handles,SEX_label,title="SEX")

Marrg_Edu=sns.catplot(x='EDU',y='DEFAULT',hue='MARRG',data=data_cln1, kind="point",ax=ax3)
ax3.set(ylabel='Default Probility')
ax3.set_xticklabels(EDU_label)
current_handles, current_labels=ax3.get_legend_handles_labels()
ax3.legend(current_handles,MARRG_label,title="Marriage")

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.4)

for i in range(0,3):
    plt.close(i+2)
plt.show()


# In[217]:


####(3)categorical attributes correlations
cat_df=data_cln1[catcol]
for col in catcol:
    cat_df[col]=cat_df[col].astype('category')
cat_df   
def cramers_v(x,y):
    confusion_matrix = pd.crosstab(x,y).values
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

corr_df = pd.DataFrame(index=catcol, columns=catcol)
corr_df= corr_df.fillna(0)
for col in cat_df.columns:
    for indx in cat_df.columns:
        corr_df.loc[indx,col]=cramers_v(cat_df[indx],cat_df[col]).round(2)
plt.figure(figsize=(16,10))
ax=sns.heatmap(corr_df,annot=True,cmap=sns.cubehelix_palette(8))
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5) 

plt.show()


# In[218]:


## 1.3 dealing with the numeric attributes
###1.3.1 check numeric attributes's values
data_cln1[numcol].describe()


# In[219]:


###1.3.2 boxplot of numeric attributes
fig, ((ax1, ax2,ax3,ax4,ax5,ax6,ax7),(ax8,ax9,ax10,ax11,ax12,ax13,ax14)) = plt.subplots(nrows=2, ncols=7,figsize=(30,10))
ax=[ax1, ax2, ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14]
for i in range(0,14):
    sns.boxplot(x=data_cln1[data_cln1[numcol].columns[i]],ax=ax[i])
plt.show()


# In[220]:


###1.3.3 outliers detection (replace outliers values with median values).
def find_outliers(x):
    Q1=np.percentile(x,25)
    Q3=np.percentile(x,75)
    IQR=Q3-Q1
    L_OUT=Q1-1.5*IQR
    H_OUT=Q3+1.5*IQR
    indices=list(x.index[(x<L_OUT)|(x>H_OUT)])
    values=list(x[indices])
    return indices, values
Results=pd.DataFrame(columns=['min_out','max_out','number of out'],index=numcol)
for col in numcol:
        pos,outliers=find_outliers(data_cln1[col])
        num=len(outliers)
        Results.loc[col,:]=[min(outliers),max(outliers),num]   
Results


# In[221]:


###1.3.4 relations between numeric attributes and class attribute
####(1) single numeric attribute vs class attribute
combo_attr=numcol[:]
combo_attr.append("DEFAULT")
combo_df=data_cln1[combo_attr]
df1=combo_df.loc[:,['LMT_B',"AGE","DEFAULT"]]

fig1,((ax1,ax2)) =plt.subplots(nrows=1, ncols=2,figsize=(15,4))
axes1=[ax1,ax2]
for i in range(0,2):
    p1=sns.violinplot(x='DEFAULT',y=df1.columns[i],data=df1,ax=axes1[i])   
    axes1[i].set_xticklabels(DEFAULT_label)


# In[222]:


df2=combo_df.drop(['LMT_B',"AGE"],1)
fig2, ((ax3,ax4,ax5,ax6),(ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14)) = plt.subplots(nrows=3, ncols=4,figsize=(20,9))
axes2=[ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14]
for i in range(0,12):
    p2=sns.violinplot(x='DEFAULT',y=df2.columns[i],data=df2,ax=axes2[i])
    axes2[i].set_xticklabels(DEFAULT_label)
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.4)   
plt.show()


# In[223]:


####(2) two numeric attributes vs class attribute
plt.figure(figsize=(13,8))
p=sns.scatterplot(x=df1['LMT_B'], y=df1['AGE'],hue=df1['DEFAULT'])
current_handles, current_labels=p.get_legend_handles_labels()
p.legend(current_handles,['DEFAULT', 'No', 'Yes'],loc='upper right',prop={'size': 10})
plt.show()


# In[224]:


####(3) correlation between numeric attribues
corr_num=combo_df.corr(method ='kendall')
plt.figure(figsize=(16,10))
ax=sns.heatmap(corr_num,annot=True,cmap="Blues")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5) 

plt.show()


# In[225]:


#2 Feature Selections
##2.1 Data transformation
###2.1.1 get dummies 
DummyList=catcol[0:9]

df_dummy=data_cln1[:]

for x in DummyList:
    df_dummy[x]=df_dummy[x].astype("category")
    
df_dummy=pd.get_dummies(df_dummy)
df_dummy.head(10)


# In[608]:


###2.1.2 Data Robust scaling
scaler=RobustScaler()
df_scale=df_dummy.copy(deep=True)
scale=pd.DataFrame(scaler.fit_transform(df_scale[numcol]),columns=df_scale[numcol].columns)
for x in numcol:
    df_scale[x]=scale[x]
df_scale.head(10)


# In[227]:


##2.2 Feature-Selection models

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.4, random_state=0)
X_df=df_scale.drop("DEFAULT",1)
X=X_df.values
y_df=df_scale["DEFAULT"]
y=y_df.values

col_names=X_df.columns

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


# In[228]:


####2.2.1 Models choosen
ridgeCV=RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])

lassoCV=LassoCV(cv=5,random_state=0)

larsCV=LarsCV(cv=5)

lalaIC=LassoLarsIC(criterion='bic')

bayeRdg=BayesianRidge()

rfe = RFE(LogisticRegression(),n_features_to_select=5)

adboost = AdaBoostRegressor(random_state=0)

et = ExtraTreesRegressor(random_state=0)

rf = RandomForestRegressor(random_state=0)

gb=GradientBoostingRegressor(random_state=0)

models=[ridgeCV,lassoCV,larsCV,lalaIC,bayeRdg,rfe,adboost,et,rf,gb]
models_names=['ridgeCV','lassoCV','larsCV','lalaIC','bayeRdg','rfe','adboost','et','rf','gb']


# In[229]:


####2.2.2 Feature Selection Model building
avg_scores = []
ranks = {}

for i in range(10): 
    scores=[]
    coefs=[]
    for train_index,test_index in sss.split(X,y):
        x_train = X_df.iloc[train_index]
        x_test = X_df.iloc[test_index]
        y_train = y_df.iloc[train_index]
        y_test = y_df.iloc[test_index] 
        Model=models[i].fit(x_train,y_train)
        scores.append(Model.score(x_test,y_test))
        if i<5:
            coefs.append(np.abs(Model.coef_))
        elif i==5:
            coefs.append(np.abs(Model.ranking_))
        elif i>5:
            coefs.append(np.abs(Model.feature_importances_))
    avg_score=np.mean(scores)
    if i == 5:
        ranks[models_names[i]] = rank_to_dict(pd.DataFrame(coefs).mean().values,col_names,order=-1)
    else:
        ranks[models_names[i]] = rank_to_dict(pd.DataFrame(coefs).mean().values,col_names,order=1)
    avg_scores.append(avg_score)


# In[230]:


###2.2.4 Feature selection results
pd.set_option('display.max.rows', 600)

results=pd.DataFrame.from_dict(ranks)
avg_scores


# In[231]:


adj_r=results.copy(deep=True)
for i in range(10):
    adj_r.iloc[:,i]=(adj_r.iloc[:,i]*avg_scores[i])/np.sum(avg_scores)*10


# In[281]:


#Feature Ranks
adj=adj_r.copy(deep=True)
adj['MeanScore']=adj_r.mean(axis=1)
Feature_Ranks=adj.sort_values('MeanScore',ascending= False)
Feature_Ranks.iloc[np.r_[0:5, -5:0]]


# In[785]:


Feature_Ranks.loc[Feature_Ranks['MeanScore']<0.2,:].index


# In[289]:


###2.2.5 Feature selected data 
####(1)selection threshold is MeanScore>0.1
Fs_df1=df_scale.loc[:,Feature_Ranks.loc[Feature_Ranks['MeanScore']>=0.1,:].index]
Fs_df1["DEFAULT"]=df_scale["DEFAULT"]
####(2)selection thresold is MeanScore>0.2
Fs_df2=df_scale.loc[:,Feature_Ranks.loc[Feature_Ranks['MeanScore']>=0.2,:].index]
Fs_df2["DEFAULT"]=df_scale["DEFAULT"]
####(3)Dataset without Feature selection
NoFs_df=df_scale

### check data structure
print(len(Fs_df2.columns),len(NoFs_df.columns))


# In[233]:


####export dataframes
NoFs_df.to_csv('/Users/jingjingsun/Desktop/df.csv')
Fs_df1.to_csv('/Users/jingjingsun/Desktop/df1.csv')
Fs_df2.to_csv('/Users/jingjingsun/Desktop/df2.csv')


# In[487]:


#3. Classification
##3.1 Naive Bayes algorithm
Naive=pd.read_csv("/Users/jingjingsun/Desktop/nb_final.csv")
Naive=Naive.drop(Naive.columns[0],1)
Naive


# In[489]:


fig, ((axe1, axe2),(axe3,axe4)) = plt.subplots(nrows=2, ncols=2,figsize=(25,10))
axen=[axe1,axe2,axe3,axe4,axe5,axe6,axe7,axe8,axe9,axe10]
linestyles=['--','-','-.',':']
for i in range(0,4):
    sns.catplot(x='OutLoop_No',y=Naive.columns[i+2],hue='methods',data=Naive,kind="point",ax=axen[i],height=15,linestyles=linestyles)
    axen[i].set_ylabel(Naive.columns[i+2],fontsize=16)
    axen[i].set_xlabel("OutLoop_No",fontsize=16)
    axe1.set_ylim([0.76,0.86])
    axe2.set_ylim([0.10,0.50])
    axe3.set_ylim([0.20,0.60])
    axe4.set_ylim([0.70,0.80])
for i in range(0,4):
    plt.close(i+2)
plt.show()


# In[490]:


fig, ((axe5,axe6),(axe7,axe8),(axe9,axe10)) = plt.subplots(nrows=3, ncols=2,figsize=(25,15))
axen=[axe5,axe6,axe7,axe8,axe9,axe10]
linestyles=['--','-','-.',':']
for i in range(0,6):
    sns.catplot(x='OutLoop_No',y=Naive.columns[i+6],hue='methods',data=Naive,kind="point",ax=axen[i],height=10,linestyles=linestyles)
    axen[i].set_ylabel(Naive.columns[i+6],fontsize=16)
    axen[i].set_xlabel("OutLoop_No",fontsize=16)
    axe5.set_ylim([-0.2,1.2])
    axe6.set_ylim([-0.2,1.2])
    axe7.set_ylim([-0.2,1.2])
    axe8.set_ylim([-0.2,1.2])
    axe9.set_ylim([-0.2,1.2])
    axe10.set_ylim([-0.2,1.2])
for i in range(0,6):
    plt.close(i+2)
plt.show()


# In[491]:


##metrics median for each methods
df4=Naive.drop(Naive.columns[[6,7]],1)
methods=df4.methods.unique()
colnames=df4.columns[range(2,6)]
Results4=pd.DataFrame(columns=colnames,index=methods)

for md in methods:
    for col in colnames:
        Results4.loc[md,col]=df4.loc[df4.iloc[:,0]==md,col].median()
Results4.columns=['Median_Acc',"Median_Recall","Median_F1","Median_AUC"]
Results4


# In[492]:


#### caculate the distance from each loops data to median values
df4_med=df4.iloc[:,range(6)]
df4_med['Acc_dist']=0
df4_med['Recall_dist']=0
df4_med['F1_dist']=0
df4_med['Roc_dist']=0
df4_med['dist']=0
for ind in Results4.index:
    df4_med.loc[df4_med.methods==ind,'Acc_dist']=df4_med.loc[df4_med.methods==ind,'Accuracy']-Results4.loc[ind,'Median_Acc']
    df4_med.loc[df4_med.methods==ind,'Recall_dist']=df4_med.loc[df4_med.methods==ind,'Recall']-Results4.loc[ind,'Median_Recall']
    df4_med.loc[df4_med.methods==ind,'F1_dist']=df4_med.loc[df4_med.methods==ind,'F1']-Results4.loc[ind,'Median_F1']
    df4_med.loc[df4_med.methods==ind,'Roc_dist']=df4_med.loc[df4_med.methods==ind,'AUC']-Results4.loc[ind,'Median_AUC']
for i in range(20):
    df4_med.iloc[i,10]=math.sqrt(df4_med.iloc[i,6]**2+df4_med.iloc[i,7]**2+df4_med.iloc[i,8]**2+df4_med.iloc[i,9]**2)
                                                
df4_med


# In[493]:


df4_Results=Results4.copy(deep=True)
df4_Results.columns=['Accuracy','Recall','F1','AUC']
df4_Results['OutLoop_No']=0
df4_Results.loc["T88_NaiveBayes",:]=df4_med.iloc[df4_med.loc[df4_med.methods=="T88_NaiveBayes",'dist'].idxmin(),range(1,6)]
df4_Results.loc["T48_NaiveBayes",:]=df4_med.iloc[df4_med.loc[df4_med.methods=="T48_NaiveBayes",'dist'].idxmin(),range(1,6)]
df4_Results.loc["T88_smote_NaiveBayes",:]=df4_med.iloc[df4_med.loc[df4_med.methods=="T88_smote_NaiveBayes",'dist'].idxmin(),range(1,6)]
df4_Results.loc["T48_smote_NaiveBayes",:]=df4_med.iloc[df4_med.loc[df4_med.methods=="T48_smote_NaiveBayes",'dist'].idxmin(),range(1,6)]
df4_Results


# In[494]:


df4_Results['methods']=df4_Results.index


fig, ax1= plt.subplots(figsize=(8,5))
color = 'tab:red'
ax1.set_xlabel('methods')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(df4_Results.loc[:,'methods'], df4_Results.loc[:,'Accuracy'],marker='o', color=color,label="Acurracy", linestyle=":")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0.70,0.85])
ax1.set_ylabel('Accuracy',fontsize=12)
ax1.set_xlabel('Methods',fontsize=12)


ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('AUC', color=color)  
ax2.plot(df4_Results.loc[:,'methods'], df4_Results.loc[:,'AUC'], marker='^',color=color,label="AUC", linestyle=":")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0.70,0.85])
ax2.set_ylabel('AUC',fontsize=12)

plt.show()


# In[495]:


fig, ax3= plt.subplots(figsize=(8,5))

color = 'tab:green'
ax3.set_xlabel('methods')
ax3.set_ylabel('Recall', color=color)
ax3.plot(df4_Results.loc[:,'methods'], df4_Results.loc[:,'Recall'],marker='o', color=color,label="Recall", linestyle=":")
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_ylim([0.10,0.25])
ax3.set_ylabel('Recall',fontsize=12)
ax3.set_xlabel('Methods',fontsize=12)


ax4 = ax3.twinx()  

color = 'tab:red'
ax4.set_ylabel('F1', color=color)  
ax4.plot(df4_Results.loc[:,'methods'], df4_Results.loc[:,'F1'], marker='^',color=color,label="F1", linestyle=":")
ax4.tick_params(axis='y', labelcolor=color)
ax4.set_ylim([0.20,0.35])
ax4.set_ylabel('F1',fontsize=12)

plt.show()

#we choose T48_smote_NaiveBayes as our model. this model is stable and the metric values are good


# In[503]:


####hyperparameters for selected model
Naive.loc[(Naive.methods=='T48_smote_NaiveBayes')&(Naive.OutLoop_No==1),Naive.columns[range(6,12)]]


# In[524]:


##3.2 CART algorithm (Recursive Partitioning And Regression Trees)
CART=pd.read_csv("/Users/jingjingsun/Desktop/rp_final.csv")
CART=CART.drop(CART.columns[0],1)
CART


# In[525]:


fig, ((axe1, axe2),(axe3,axe4),(axe5,axe6)) = plt.subplots(nrows=3, ncols=2,figsize=(25,15))
axen=[axe1,axe2,axe3,axe4,axe5,axe6]
linestyles=['--','-','-.',':']
for i in range(0,6):
    sns.catplot(x='OutLoop_No',y=CART.columns[i+2],hue='methods',data=CART,kind="point",ax=axen[i],height=10,linestyles=linestyles)
    axen[i].set_ylabel(CART.columns[i+2],fontsize=16)
    axen[i].set_xlabel("OutLoop_No",fontsize=16)
    axe1.set_ylim([0.76,0.86])
    axe2.set_ylim([0.2,0.60])
    axe3.set_ylim([0.40,0.60])
    axe4.set_ylim([0.65,0.85])
    axe5.set_ylim([0,0.007])
    axe6.set_ylim([0,0.007])
for i in range(0,6):
    plt.close(i+2)
plt.show()


# In[526]:


##metrics median for each methods
df1=CART.drop(CART.columns[[6,7]],1)
methods=df1.methods.unique()
colnames=df1.columns[range(2,6)]
Results1=pd.DataFrame(columns=colnames,index=methods)

for md in methods:
    for col in colnames:
        Results1.loc[md,col]=df1.loc[df1.iloc[:,0]==md,col].median()
Results1.columns=['Median_Acc',"Median_Recall","Median_F1","Median_AUC"]      
Results1


# In[527]:


#### caculate the distance from each loops data to median values
df1_med=df1.iloc[:,range(6)]
df1_med['Acc_dist']=0
df1_med['Recall_dist']=0
df1_med['F1_dist']=0
df1_med['Roc_dist']=0
df1_med['dist']=0
for ind in Results1.index:
    df1_med.loc[df1_med.methods==ind,'Acc_dist']=df1_med.loc[df1_med.methods==ind,'Accuracy']-Results1.loc[ind,'Median_Acc']
    df1_med.loc[df1_med.methods==ind,'Recall_dist']=df1_med.loc[df1_med.methods==ind,'Recall']-Results1.loc[ind,'Median_Recall']
    df1_med.loc[df1_med.methods==ind,'F1_dist']=df1_med.loc[df1_med.methods==ind,'F1']-Results1.loc[ind,'Median_F1']
    df1_med.loc[df1_med.methods==ind,'Roc_dist']=df1_med.loc[df1_med.methods==ind,'AUC']-Results1.loc[ind,'Median_AUC']
for i in range(20):
    df1_med.iloc[i,10]=math.sqrt(df1_med.iloc[i,6]**2+df1_med.iloc[i,7]**2+df1_med.iloc[i,8]**2+df1_med.iloc[i,9]**2)
                                                
df1_med


# In[529]:


df1_Results=Results1.copy(deep=True)
df1_Results.columns=['Accuracy','Recall','F1','AUC']
df1_Results['OutLoop_No']=0
df1_Results.loc["T88_rpart",:]=df1_med.iloc[df1_med.loc[df1_med.methods=="T88_rpart",'dist'].idxmin(),range(1,6)]
df1_Results.loc["T48_rpart",:]=df1_med.iloc[df1_med.loc[df1_med.methods=="T48_rpart",'dist'].idxmin(),range(1,6)]
df1_Results.loc["T88_smote_rpart",:]=df1_med.iloc[df1_med.loc[df1_med.methods=="T88_smote_rpart",'dist'].idxmin(),range(1,6)]
df1_Results.loc["T48_smote_rpart",:]=df1_med.iloc[df1_med.loc[df1_med.methods=="T48_smote_rpart",'dist'].idxmin(),range(1,6)]
df1_Results


# In[532]:


df1_Results['methods']=df1_Results.index


fig, ax1= plt.subplots(figsize=(8,5))
color = 'tab:red'
ax1.set_xlabel('methods')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(df1_Results.loc[:,'methods'], df1_Results.loc[:,'Accuracy'],marker='o', color=color,label="Acurracy", linestyle=":")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0.75,0.85])
ax1.set_ylabel('Accuracy',fontsize=12)
ax1.set_xlabel('Methods',fontsize=12)


ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('AUC', color=color)  
ax2.plot(df1_Results.loc[:,'methods'], df1_Results.loc[:,'AUC'], marker='^',color=color,label="AUC", linestyle=":")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0.65,0.75])
ax2.set_ylabel('AUC',fontsize=12)

plt.show()


# In[531]:


fig, ax3= plt.subplots(figsize=(8,5))

color = 'tab:green'
ax3.set_xlabel('methods')
ax3.set_ylabel('Recall', color=color)
ax3.plot(df1_Results.loc[:,'methods'], df1_Results.loc[:,'Recall'],marker='o', color=color,label="Recall", linestyle=":")
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_ylim([0.30,0.60])
ax3.set_ylabel('Recall',fontsize=12)
ax3.set_xlabel('Methods',fontsize=12)


ax4 = ax3.twinx()  

color = 'tab:red'
ax4.set_ylabel('F1', color=color)  
ax4.plot(df1_Results.loc[:,'methods'], df1_Results.loc[:,'F1'], marker='^',color=color,label="F1", linestyle=":")
ax4.tick_params(axis='y', labelcolor=color)
ax4.set_ylim([0.30,0.60])
ax4.set_ylabel('F1',fontsize=12)

plt.show()

#we choose df_rpart_smote as our model. this model is stable and the metric values are good


# In[549]:


##3.3 LogitBoost algorithm
LogBoost=pd.read_csv("/Users/jingjingsun/Desktop/lg_final.csv")
LogBoost=LogBoost.drop(LogBoost.columns[0],1)
LogBoost


# In[550]:


fig, ((axe1, axe2),(axe3,axe4),(axe5,axe6)) = plt.subplots(nrows=3, ncols=2,figsize=(25,15))
axen=[axe1,axe2,axe3,axe4,axe5,axe6]
linestyles=['--','-','-.',':']
for i in range(0,6):
    sns.catplot(x='OutLoop_No',y=LogBoost.columns[i+2],hue='methods',data=LogBoost,kind="point",ax=axen[i],height=10,linestyles=linestyles)
    axen[i].set_ylabel(LogBoost.columns[i+2],fontsize=16)
    axen[i].set_xlabel("OutLoop_No",fontsize=16)
    axe1.set_ylim([0.70,0.85])
    axe2.set_ylim([0.15,0.65])
    axe3.set_ylim([0.20,0.60])
    axe4.set_ylim([0.60,0.75])
    axe5.set_ylim([0,100])
    axe6.set_ylim([0,100])
for i in range(0,6):
    plt.close(i+2)
plt.show()


# In[551]:


##metrics median for each methods

df2=LogBoost.drop(LogBoost.columns[[6,7]],1)

methods=df2.methods.unique()
colnames=df2.columns[range(2,6)]
Results2=pd.DataFrame(columns=colnames,index=methods)

for md in methods:
    for col in colnames:
        Results2.loc[md,col]=df2.loc[df2.iloc[:,0]==md,col].median()
Results2.columns=['Median_Acc',"Median_Recall","Median_F1","Median_AUC"]        
Results2


# In[552]:


#### caculate the distance from each loops data to median values
df2_med=df2.iloc[:,range(6)]
df2_med['Acc_dist']=0
df2_med['Recall_dist']=0
df2_med['F1_dist']=0
df2_med['Roc_dist']=0
df2_med['dist']=0
for ind in Results2.index:
    df2_med.loc[df2_med.methods==ind,'Acc_dist']=df2_med.loc[df2_med.methods==ind,'Accuracy']-Results2.loc[ind,'Median_Acc']
    df2_med.loc[df2_med.methods==ind,'Recall_dist']=df2_med.loc[df2_med.methods==ind,'Recall']-Results2.loc[ind,'Median_Recall']
    df2_med.loc[df2_med.methods==ind,'F1_dist']=df2_med.loc[df2_med.methods==ind,'F1']-Results2.loc[ind,'Median_F1']
    df2_med.loc[df2_med.methods==ind,'Roc_dist']=df2_med.loc[df2_med.methods==ind,'AUC']-Results2.loc[ind,'Median_AUC']
for i in range(20):
    df2_med.iloc[i,10]=math.sqrt(df2_med.iloc[i,6]**2+df2_med.iloc[i,7]**2+df2_med.iloc[i,8]**2+df2_med.iloc[i,9]**2)
                                                
df2_med


# In[553]:


df2_Results=Results2.copy(deep=True)
df2_Results.columns=['Accuracy','Recall','F1','AUC']
df2_Results['OutLoop_No']=0
df2_Results.loc["T88_lgBoost",:]=df2_med.iloc[df2_med.loc[df2_med.methods=="T88_lgBoost",'dist'].idxmin(),range(1,6)]
df2_Results.loc["T48_lgBoost",:]=df2_med.iloc[df2_med.loc[df2_med.methods=="T48_lgBoost",'dist'].idxmin(),range(1,6)]
df2_Results.loc["T88_smote_lgBoost",:]=df2_med.iloc[df2_med.loc[df2_med.methods=="T88_smote_lgBoost",'dist'].idxmin(),range(1,6)]
df2_Results.loc["T48_smote_lgBoost",:]=df2_med.iloc[df2_med.loc[df2_med.methods=="T48_smote_lgBoost",'dist'].idxmin(),range(1,6)]
df2_Results


# In[554]:


df2_Results['methods']=df2_Results.index


fig, ax1= plt.subplots(figsize=(8,5))
color = 'tab:red'
ax1.set_xlabel('methods')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(df2_Results.loc[:,'methods'], df2_Results.loc[:,'Accuracy'],marker='o', color=color,label="Acurracy", linestyle=":")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0.70,0.85])
ax1.set_ylabel('Accuracy',fontsize=10)
ax1.set_xlabel('Methods',fontsize=10)


ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('AUC', color=color)  
ax2.plot(df2_Results.loc[:,'methods'], df2_Results.loc[:,'AUC'], marker='^',color=color,label="AUC", linestyle=":")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0.65,0.73])
ax2.set_ylabel('AUC',fontsize=10)

plt.show()


# In[556]:


fig, ax3= plt.subplots(figsize=(8,5))

color = 'tab:green'
ax3.set_xlabel('methods')
ax3.set_ylabel('Recall', color=color)
ax3.plot(df2_Results.loc[:,'methods'], df2_Results.loc[:,'Recall'],marker='o', color=color,label="F1", linestyle=":")
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_ylim([0.25,0.45])
ax3.set_ylabel('Recall',fontsize=12)
ax3.set_xlabel('Methods',fontsize=12)


ax4 = ax3.twinx()  

color = 'tab:red'
ax4.set_ylabel('F1', color=color)  
ax4.plot(df2_Results.loc[:,'methods'], df2_Results.loc[:,'F1'], marker='^',color=color,label="F1", linestyle=":")
ax4.tick_params(axis='y', labelcolor=color)
ax4.set_ylim([0.35,0.55])
ax4.set_ylabel('F1',fontsize=12)

plt.show()
##we choose df2_logBoost as the model 


# In[564]:


##3.4 glmnet (Lasso and Elastic-Net Regularized Generalized Linear Models)
glmnet=pd.read_csv("/Users/jingjingsun/Desktop/net_final.csv")
glmnet=glmnet.drop(glmnet.columns[0],1)
glmnet


# In[565]:


fig, ((axe1, axe2),(axe3,axe4),(axe5,axe6),(axe7,axe8)) = plt.subplots(nrows=4, ncols=2,figsize=(25,15))
axen=[axe1,axe2,axe3,axe4,axe5,axe6,axe7,axe8]
linestyles=['--','-','-.',':']
for i in range(0,8):
    sns.catplot(x='OutLoop_No',y=glmnet.columns[i+2],hue='methods',data=glmnet,kind="point",ax=axen[i],height=10,linestyles=linestyles)
    axen[i].set_ylabel(glmnet.columns[i+2],fontsize=16)
    axen[i].set_xlabel("OutLoop_No",fontsize=16)
    axe1.set_ylim([0.75,0.85])
    axe2.set_ylim([0.25,0.60])
    axe3.set_ylim([0.20,0.60])
    axe4.set_ylim([0.70,0.80])
    axe5.set_ylim([0,2.0])
    axe6.set_ylim([0,0.2])
    axe7.set_ylim([0,2.0])
    axe8.set_ylim([0,0.2])
for i in range(0,8):
    plt.close(i+2)
plt.show()


# In[566]:


##metrics median for each methods

df3=glmnet.drop(glmnet.columns[[6,7]],1)

methods=df3.methods.unique()
colnames=df3.columns[range(2,6)]
Results3=pd.DataFrame(columns=colnames,index=methods)

for md in methods:
    for col in colnames:
        Results3.loc[md,col]=df3.loc[df3.iloc[:,0]==md,col].median()
Results3.columns=['Median_Acc',"Median_Recall","Median_F1","Median_AUC"]       
Results3


# In[567]:


#### caculate the distance from each loops data to median values
df3_med=df3.iloc[:,range(6)]
df3_med['Acc_dist']=0
df3_med['Recall_dist']=0
df3_med['F1_dist']=0
df3_med['Roc_dist']=0
df3_med['dist']=0
for ind in Results3.index:
    df3_med.loc[df3_med.methods==ind,'Acc_dist']=df3_med.loc[df3_med.methods==ind,'Accuracy']-Results3.loc[ind,'Median_Acc']
    df3_med.loc[df3_med.methods==ind,'Recall_dist']=df3_med.loc[df3_med.methods==ind,'Recall']-Results3.loc[ind,'Median_Recall']
    df3_med.loc[df3_med.methods==ind,'F1_dist']=df3_med.loc[df3_med.methods==ind,'F1']-Results3.loc[ind,'Median_F1']
    df3_med.loc[df3_med.methods==ind,'Roc_dist']=df3_med.loc[df3_med.methods==ind,'AUC']-Results3.loc[ind,'Median_AUC']
for i in range(20):
    df3_med.iloc[i,10]=math.sqrt(df3_med.iloc[i,6]**2+df3_med.iloc[i,7]**2+df3_med.iloc[i,8]**2+df3_med.iloc[i,9]**2)
                                                
df3_med


# In[569]:


#distiance results
df3_Results=Results3.copy(deep=True)
df3_Results.columns=['Accuracy','Recall','F1','AUC']
df3_Results['OutLoop_No']=0
df3_Results.loc["T88_net",:]=df3_med.iloc[df3_med.loc[df3_med.methods=="T88_net",'dist'].idxmin(),range(1,6)]
df3_Results.loc["T48_net",:]=df3_med.iloc[df3_med.loc[df3_med.methods=="T48_net",'dist'].idxmin(),range(1,6)]
df3_Results.loc["T88_smote_net",:]=df3_med.iloc[df3_med.loc[df3_med.methods=="T88_smote_net",'dist'].idxmin(),range(1,6)]
df3_Results.loc["T48_smote_net",:]=df3_med.iloc[df3_med.loc[df3_med.methods=="T48_smote_net",'dist'].idxmin(),range(1,6)]
df3_Results


# In[570]:


df3_Results['methods']=df3_Results.index


fig, ax1= plt.subplots(figsize=(8,5))
color = 'tab:red'
ax1.set_xlabel('methods')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(df3_Results.loc[:,'methods'], df3_Results.loc[:,'Accuracy'],marker='o', color=color,label="Acurracy", linestyle=":")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0.70,0.85])
ax1.set_ylabel('Accuracy',fontsize=10)
ax1.set_xlabel('Methods',fontsize=10)


ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('AUC', color=color)  
ax2.plot(df3_Results.loc[:,'methods'], df3_Results.loc[:,'AUC'], marker='^',color=color,label="AUC", linestyle=":")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0.70,0.85])
ax2.set_ylabel('AUC',fontsize=10)

plt.show()


# In[571]:


fig, ax3= plt.subplots(figsize=(8,5))

color = 'tab:green'
ax3.set_xlabel('methods')
ax3.set_ylabel('Recall', color=color)
ax3.plot(df3_Results.loc[:,'methods'], df3_Results.loc[:,'Recall'],marker='o', color=color,label="F1", linestyle=":")
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_ylim([0.25,0.45])
ax3.set_ylabel('Recall',fontsize=12)
ax3.set_xlabel('Methods',fontsize=12)


ax4 = ax3.twinx()  

color = 'tab:red'
ax4.set_ylabel('F1', color=color)  
ax4.plot(df3_Results.loc[:,'methods'], df3_Results.loc[:,'F1'], marker='^',color=color,label="F1", linestyle=":")
ax4.tick_params(axis='y', labelcolor=color)
ax4.set_ylim([0.35,0.55])
ax4.set_ylabel('F1',fontsize=12)

plt.show()
##we choose df_net for this algorithm.


# In[573]:


## choose model from three methods
#df_rpart_smote, df2_logBoost, df_net
NaiveB=df4_Results.loc[df4_Results.index=="T48_smote_NaiveBayes",:]
rpart=df1_Results.loc[df1_Results.index=='T88_smote_rpart',:]
lgboost=df2_Results.loc[df2_Results.index=="T48_lgBoost",:]
net=df3_Results.loc[df3_Results.index=="T48_smote_net",:]
compare=pd.DataFrame(NaiveB)
compare=compare.append(rpart)
compare=compare.append(lgboost)
compare=compare.append(net)
compare


# In[574]:


fig, ax1= plt.subplots(figsize=(8,5))
color = 'tab:red'
ax1.set_xlabel('methods')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(compare.loc[:,'methods'], compare.loc[:,'Accuracy'],marker='o', color=color,label="Acurracy", linestyle=":")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0.75,0.85])
ax1.set_ylabel('Accuracy',fontsize=10)
ax1.set_xlabel('Methods',fontsize=10)


ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('AUC', color=color)  
ax2.plot(compare.loc[:,'methods'], compare.loc[:,'AUC'], marker='^',color=color,label="AUC", linestyle=":")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0.68,0.85])
ax2.set_ylabel('AUC',fontsize=10)

plt.show()


# In[575]:


fig, ax3= plt.subplots(figsize=(8,5))

color = 'tab:green'
ax3.set_xlabel('methods')
ax3.set_ylabel('Recall', color=color)
ax3.plot(compare.loc[:,'methods'], compare.loc[:,'Recall'],marker='o', color=color,label="F1", linestyle=":")
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_ylim([0.15,0.45])
ax3.set_ylabel('Recall',fontsize=12)
ax3.set_xlabel('Methods',fontsize=12)


ax4 = ax3.twinx()  

color = 'tab:red'
ax4.set_ylabel('F1', color=color)  
ax4.plot(compare.loc[:,'methods'], compare.loc[:,'F1'], marker='^',color=color,label="F1", linestyle=":")
ax4.tick_params(axis='y', labelcolor=color)
ax4.set_ylim([0.25,0.55])
ax4.set_ylabel('F1',fontsize=12)

plt.show()


# In[660]:


#4. Unsupervised learning (clustering)
##4.1 plot clusters in a 2-Dimensional plot 
###4.1.1 PCA 2-D plot
df_2D=df_scale.copy(deep=True)
pca_2D=df_2D.drop("DEFAULT",1).values
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_2D)


# In[661]:


df_2D['pca-one'] = pca_result[:,0]
df_2D['pca-two'] = pca_result[:,1] 

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# In[662]:


np.random.seed(42)
rndperm = np.random.permutation(df_2D.shape[0])


# In[663]:


####(1) True DEFAULT data plotting

plt.figure(figsize=(8,8))
ax=sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="DEFAULT",
    palette=sns.color_palette("hls", 2),
    data=df_2D.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
current_handles, current_labels=ax.get_legend_handles_labels()
ax.legend(current_handles,['DEFAULT','No','Yes'],loc='upper right',prop={'size': 10})
ax.set_ylabel("PCA-Two",fontsize=16)
ax.set_xlabel("PCA-One",fontsize=16)
plt.show()


# In[581]:


####(2) K-Means clustering results (K=2) on PCA 2-dimensional plot
df_2D=df_scale.copy(deep=True)
km_X=df_2D.drop("DEFAULT",1).values
km = KMeans(n_clusters=2, random_state=0)
km.fit(km_X)
est_y = km.predict(km_X)


# In[582]:


df_2D['pca-one'] = pca_result[:,0]
df_2D['pca-two'] = pca_result[:,1] 


# In[583]:


est_y[np.where(est_y==1)]=2
est_y[np.where(est_y==0)]=1
est_y[np.where(est_y==2)]=0

df_2D["est_y"]=est_y


# In[588]:


#### plotting results
plt.figure(figsize=(8,8))
ax=sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="est_y",
    palette=sns.color_palette("hls", 2),
    data=df_2D.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
current_handles, current_labels=ax.get_legend_handles_labels()
ax.legend(current_handles,['DEFAULT', 'No', 'Yes'],loc='upper right',prop={'size': 10})
ax.set_ylabel("PCA-Two",fontsize=16)
ax.set_xlabel("PCA-One",fontsize=16)
plt.show()


# In[814]:


##4.1.2 T-Distributed Stochastic Neighbouring Entities (t-SNE)plot
df_2D=df_scale.copy(deep=True)
tsne_2D=df_2D.drop("DEFAULT",1).values
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250)
tsne_results = tsne.fit_transform(tsne_2D)


# In[815]:


df_2D=df_scale.copy(deep=True)
km_X=df_2D.drop("DEFAULT",1).values
km = KMeans(n_clusters=2, random_state=0)
km.fit(km_X)
est_y = km.predict(km_X)


# In[816]:


df_2D['tsne-2d-one'] = tsne_results[:,0]
df_2D['tsne-2d-two'] = tsne_results[:,1]
df_2D['est_y']=est_y


# In[817]:


np.random.seed(42)
rndperm = np.random.permutation(df_2D.shape[0])


# In[818]:


####(1)True DEFAULT data 
plt.figure(figsize=(8,8))
ax=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="DEFAULT",
    palette=sns.color_palette("hls", 2),
    data=df_2D,
    legend="full",
    alpha=0.3
)
current_handles, current_labels=ax.get_legend_handles_labels()
ax.legend(current_handles,['DEFAULT', 'No', 'Yes'],loc='upper right',prop={'size': 15})
ax.set_ylabel("TSNE-Two",fontsize=16)
ax.set_xlabel("TSNE-One",fontsize=16)
plt.show()


# In[801]:


np.random.seed(33)
rndperm = np.random.permutation(df_2D.shape[0])


# In[819]:


####(2) K-Means (K=2) clustering results 
plt.figure(figsize=(8,8))
ax=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="est_y",
    palette=sns.color_palette("hls", 2),
    data=df_2D,
    legend="full",
    alpha=0.3
)
current_handles, current_labels=ax.get_legend_handles_labels()
ax.legend(current_handles,['DEFAULT', 'No', 'Yes'],loc='upper right',prop={'size': 15})
ax.set_ylabel("TSNE-Two",fontsize=16)
ax.set_xlabel("TSNE-One",fontsize=16)
plt.show()


# In[820]:


#4.2 K-Means (K=10) clustering results 
## removed the instances that having no default value
df_2D=df_scale.copy(deep=True)
df_2D=df_2D.drop(df_2D.loc[df_2D.DEFAULT==0,:].index,0)


# In[821]:


km_X=df_2D.drop("DEFAULT",1)
km = KMeans(n_clusters=10, random_state=0)
kmfit=km.fit(km_X)
est_y = km.predict(km_X)


# In[822]:


cluster_result=pd.DataFrame()
cluster_result=km_X
cluster_result["clusters"]=kmfit.labels_
results=pd.DataFrame(cluster_result["clusters"].value_counts().sort_index()).T
results['total']= results.values.sum()
results


# In[825]:


tsne_2D=df_2D.drop("DEFAULT",1)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250)
tsne_results = tsne.fit_transform(tsne_2D)


# In[826]:


df_2D['tsne-2d-one'] = tsne_results[:,0]
df_2D['tsne-2d-two'] = tsne_results[:,1]
df_2D["est_y"]=est_y


# In[828]:


np.random.seed(34)
rndperm = np.random.permutation(df_2D.shape[0])


# In[830]:


####4.2 plot 10 clusters on tsne_2-dimension plot
plt.figure(figsize=(12,8))
ax=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="est_y",
    palette=sns.color_palette("hls", 10),
    data=df_2D,
    legend="full",
    alpha=0.3
)
ax.set_ylabel("TSNE-Two",fontsize=16)
ax.set_xlabel("TSNE-One",fontsize=16)
plt.show()

