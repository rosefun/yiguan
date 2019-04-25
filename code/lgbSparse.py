# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:17:23 2018

@author: Rosefun
"""

import datetime
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb
import warnings
import time
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import csr_matrix
os.chdir(r'F:\比赛\易观')
warnings.filterwarnings("ignore")
version='_1006_v74(Sparse2lgb)'
import scipy

def basicProcess(deviceIdAll):
    '''
    对deviceIdAll进行基本处理
    '''
    print('fillna')
    for col in ['brand','model']:
        deviceIdAll[col].fillna(-1,inplace=True)
    #对训练测试数据进行一个简单处理
    print('sex age 处理')
    deviceIdAll['sex']=deviceIdAll['sex'].apply(lambda x:str(x))
    deviceIdAll['age']=deviceIdAll['age'].apply(lambda x:str(x))
    def tool(x):
        if x=='nan':
            return x
        else:
            return str(int(float(x)))
    deviceIdAll['sex']=deviceIdAll['sex'].apply(tool)
    deviceIdAll['age']=deviceIdAll['age'].apply(tool)
    deviceIdAll['sex_age']=deviceIdAll['sex']+'-'+deviceIdAll['age']
    deviceIdAll=deviceIdAll.replace({'nan':np.NaN,'nan-nan':np.NaN})
    
    #把重复的列名，只保留一个
    deviceIdAll=deviceIdAll.ix[:,~deviceIdAll.columns.duplicated()]
    #drop unique 列
    def dropUniqueColumns(df):
        """
        Drops constant value columns of pandas dataframe.
        """
        dropCol=[]
        for column in df.columns:
            if len(df[column].unique()) == 1:
                dropCol.append(column)
                df.drop(column,inplace=True,axis=1)
        return df,dropCol
    
    deviceIdAll,dropCol=dropUniqueColumns(deviceIdAll)
    print('deviceIdALl unique 值有:',dropCol)
    return deviceIdAll

def sub(deviceIdAll):
    train=deviceIdAll[deviceIdAll['sex'].notnull()]
    test=deviceIdAll[deviceIdAll['sex'].isnull()]

    X=train.drop(['sex','age','sex_age','device_id'],axis=1)
    Y=train['sex_age']
    Y_CAT=pd.Categorical(Y)
    X_train,X_test, y_train, y_test =train_test_split(X,Y_CAT.labels,test_size=0.1, random_state=1028)
    print('X_train.shape',X_train.shape)
    
def addAppUseTime(deviceIdAll,deviceid_package_start_close):
    '''
    统计每个device_id使用所有apps的时间，存储为一个稀疏矩阵
    '''
    
    if os.path.exists( '中间文件/useTimeCsr.npz') and True:
        print('load_csr---------')
        useTimeCsr = scipy.sparse.load_npz('中间文件/useTimeCsr.npz')
    else:
        #对时间戳数据进行简单的处理
        # newDeviceIdTime=pd.DataFrame({'device_id':[],'app_id':[],'startTime':[],'endTime':[]})
        # newDeviceIdTime['device_id']=deviceid_package_start_close['device_id']
        # newDeviceIdTime['app_id']=newDeviceIdTime['app_id']
        # newDeviceIdTime['startTime']=deviceid_package_start_close['startTime'].apply(lambda x:timeStamp(x))
        # newDeviceIdTime['endTime']=deviceid_package_start_close['endTime'].apply(lambda x:timeStamp(x))
        # newDeviceIdTime.to_pickle('中间文件/newDeviceIdTime.pickle')

        deviceid_package_start_close['useTime']=((deviceid_package_start_close['endTime']-
                                   deviceid_package_start_close['startTime'])/1000).astype('int')
        deviceid_package_start_close['startTime']=deviceid_package_start_close['startTime'].apply(
               lambda x:timeStamp(x))
        deviceid_package_start_close['endTime']= deviceid_package_start_close['endTime'].apply(
               lambda x:timeStamp(x))
            
        deviceIdUseTime= deviceid_package_start_close
        deviceIdUseTime2= pd.DataFrame(data=deviceIdUseTime.groupby(
               ['device_id','app_id'])['useTime'].agg('sum'))
        deviceIdUseTime2.reset_index(inplace=True)
        idList= list(deviceIdAll['device_id'])
        appList= list(sorted(deviceIdUseTime2.app_id.unique()))
        data=deviceIdUseTime2['useTime'].tolist()
        row= deviceIdUseTime2.device_id.astype('category',categories= idList).cat.codes
        col= deviceIdUseTime2.app_id.astype('category',categories= appList).cat.codes
        useTimeCsr=csr_matrix((data,(row,col)),shape=(len(idList),len(appList)))

        scipy.sparse.save_npz('中间文件/useTimeCsr.npz', useTimeCsr)

    return useTimeCsr    
    
    
    
def getDeviceIdTime(deviceid_package_start_close):
   print('对时间戳数据进行简单的处理')
   # newDeviceIdTime=pd.DataFrame({'device_id':[],'app_id':[],'startTime':[],'endTime':[]})
   # newDeviceIdTime['device_id']=deviceid_package_start_close['device_id']
   # newDeviceIdTime['app_id']=newDeviceIdTime['app_id']
   # newDeviceIdTime['startTime']=deviceid_package_start_close['startTime'].apply(lambda x:timeStamp(x))
   # newDeviceIdTime['endTime']=deviceid_package_start_close['endTime'].apply(lambda x:timeStamp(x))
   # newDeviceIdTime.to_pickle('中间文件/newDeviceIdTime.pickle')
    
   deviceid_package_start_close['useTime']=((deviceid_package_start_close['endTime']-
                               deviceid_package_start_close['startTime'])/1000).astype('int')
   deviceid_package_start_close['startTime']=deviceid_package_start_close['startTime'].apply(
           lambda x:timeStamp(x))
   deviceid_package_start_close['endTime']= deviceid_package_start_close['endTime'].apply(
           lambda x:timeStamp(x))
   # deviceid_package_start_close.to_pickle('中间文件/newDeviceIdTimeV2.pickle')
   
   return deviceid_package_start_close
    
def getRankAppTime(deviceid_package_start_close):
    #统计出使用时间最长的APP。
    deviceid_package_start_close['useTime']=((deviceid_package_start_close['endTime']-
                               deviceid_package_start_close['startTime'])/1000).astype('int')
    sumDeviceTime=deviceid_package_start_close.groupby(['device_id','app_id'])['useTime'].sum()
    sumDeviceTime=sumDeviceTime.to_frame().reset_index()
    sumDeviceTime['time1'] = sumDeviceTime.groupby('device_id').useTime.rank(
           method = 'first', ascending = False).map({1.0 : 'top1TimeApp', 
          2.0 : 'top2TimeApp',3.0:'top3TimeApp',4.0:'top4TimeApp',5.0:'top5TimeApp'})
    sumDeviceTime = sumDeviceTime.dropna().pivot('device_id', 'time1', 'app_id')
    sumDeviceTime.columns.name = None
    sumDeviceTime.reset_index(inplace = True)
    
    return sumDeviceTime
    
if __name__=='__main__':
    print('读取设备数据、APP数据、机型数据、应用时间数据、训练测试数据\n')
    #设备数据
   deviceid_packages=pd.read_csv('./data/Demo.tar/Demo/deviceid_packages.tsv',
                                 sep='\t',names=['device_id','apps'],encoding='utf-8',)

    #APP数据
   package_label=pd.read_csv('./data/Demo.tar/Demo/package_label.tsv',
                     sep='\t',names=['app_id','upperCategory','detailCategory'],encoding='utf-8')

    #机型数据
   deviceid_brand=pd.read_csv('./data/Demo.tar/Demo/deviceid_brand.tsv',
                              sep='\t',names=['device_id','brand','model'],encoding='utf-8',)

    #应用时间数据
   deviceid_package_start_close=pd.read_csv('./data/Demo.tar/Demo/deviceid_package_start_close.tsv',
                    sep='\t',names=['device_id','apps','startTime','endTime'],encoding='utf-8',)

    #训练测试数据
   deviceid_test=pd.read_csv('./data/Demo.tar/Demo/deviceid_test.tsv',sep='\t',names=['device_id'])
   deviceid_train=pd.read_csv('./data/Demo.tar/Demo/deviceid_train.tsv',
                              sep='\t',names=['device_id','sex','age'])
   deviceIdAll=pd.concat([deviceid_train,deviceid_test],axis=0)

    print('read all file done!')

    dfs=[]
    for col in ['brand','model']:
        s = deviceid_brand.set_index('device_id')[col]
        val = deviceIdAll['device_id'].map(s).rename(col)
        dfs.append(val)    
    df = pd.concat(dfs, axis=1)
    df.reset_index(drop=True,inplace=True)
    deviceIdAll.reset_index(drop=True,inplace=True)
    deviceIdAll=pd.concat([deviceIdAll,df],axis=1)
    
    dfs=[]
    for col in ['apps']:
        s = deviceid_packages.set_index('device_id')[col]
        val = deviceIdAll['device_id'].map(s).rename(col)
        dfs.append(val)    
    df = pd.concat(dfs, axis=1)
    df.reset_index(drop=True,inplace=True)
    deviceIdAll.reset_index(drop=True,inplace=True)
    deviceIdAll=pd.concat([deviceIdAll,df],axis=1)
    
    #把device_id使用APP的startTime，endTime转化为文本信息
    DPSC=deviceid_package_start_close#.copy()
    DPSC['startHour'] = DPSC['startTime'].apply(lambda x: 
        int(time.strftime("%H", time.localtime(float(x/1000)))))
    DPSC['endHour'] = DPSC['endTime'].apply(lambda x: 
        int(time.strftime("%H", time.localtime(float(x/1000)))))
        
    startHours=pd.DataFrame(DPSC.groupby('device_id')['startHour'].apply(
            lambda x:",".join((str(s) for s in x))))
    endHours=pd.DataFrame(DPSC.groupby('device_id')['endHour'].apply(
            lambda x:",".join((str(s) for s in x))))
    dfs=[]
    for col in ['startHour']:
        s = startHours[col]
        val = deviceIdAll['device_id'].map(s).rename(col+'s')
        dfs.append(val)    
    df = pd.concat(dfs, axis=1)
    df.reset_index(drop=True,inplace=True)
    deviceIdAll.reset_index(drop=True,inplace=True)
    deviceIdAll=pd.concat([deviceIdAll,df],axis=1)
    
    dfs=[]
    for col in ['endHour']:
        s = endHours[col]
        val = deviceIdAll['device_id'].map(s).rename(col+'s')
        dfs.append(val)    
    df = pd.concat(dfs, axis=1)
    df.reset_index(drop=True,inplace=True)
    deviceIdAll.reset_index(drop=True,inplace=True)
    deviceIdAll=pd.concat([deviceIdAll,df],axis=1)
    
    #2.增加每个device_id top5 APPs
    rankDeviceAppTime['endHour'] = DPSC['endTime'].apply(lambda x: 
        int(time.strftime("%H", time.localtime(float(x/1000)))))
        
    startHours=pd.DataFrame(rankDeviceAppTime.groupby('device_id')['startHour'].apply(
            lambda x:",".join((str(s) for s in x))))
    
    rankDeviceAppTime= getRankAppTime(deviceid_package_start_close)
    rankDeviceAppTime['top5Apps']= rankDeviceAppTime[rankDeviceAppTime.columns[1:]].\
    apply(lambda x: ','.join(x.fillna(str(-1)).astype(str)),axis=1)
#    rankDeviceAppTime['top5Apps']= rankDeviceAppTime[rankDeviceAppTime.columns[1:]].\
#    apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
    
    dfs = []
    for col in ['top5Apps']:
        s = rankDeviceAppTime.set_index('device_id')[col]
        val = deviceIdAll['device_id'].map(s).rename(col)
        dfs.append(val)
    df = pd.concat(dfs, axis=1)
    deviceIdAll.reset_index(drop=True,inplace=True)
    df.reset_index(drop=True,inplace=True)
    deviceIdAll=pd.concat([deviceIdAll,df],axis=1)
    
    cate_feature = [ 'brand', 'model']
    
    deviceIdAll=basicProcess(deviceIdAll)
    for i in cate_feature:
        deviceIdAll[i] = deviceIdAll[i].map(dict(zip(deviceIdAll[i].unique(), 
                   range(0, deviceIdAll[i].nunique()))))
        
    #4 对每个用户的apps进行统计
   deviceIdUseTime= getDeviceIdTime(deviceid_package_start_close)
   useAllApps= pd.DataFrame(deviceIdUseTime.groupby('device_id')['app_id'].apply(
           lambda x: ",".join((str(s)for s in x))))

    dfs=[]
    for col in ['app_id']:
        s= useAllApps[col]
        val = deviceIdAll['device_id'].map(s).rename(col)
        dfs.append(val)
    df=pd.concat(dfs,axis=1)
    deviceIdAll.reset_index(drop=True,inplace=True)
    df.reset_index(drop=True,inplace=True)
    deviceIdAll=pd.concat([deviceIdAll,df],axis=1)
    
    #4 使用LDA进行降维
    deviceid_packages['appsList']=deviceid_packages['apps'].apply(lambda x:x.split(','))
    deviceid_packages['app_length']=deviceid_packages['appsList'].apply(lambda x:len(x))
    
    apps=deviceid_packages['appsList'].apply(lambda x:' '.join(x)).tolist()
    vectorizer=CountVectorizer()
    cntTf = vectorizer.fit_transform(apps)
    word=vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_topics=5,
                                   learning_offset=50.,
                                   random_state=666)
    docres = lda.fit_transform(cntTf)
    docresDf= docres 
    for num in range(5):
        docresDf.rename(columns={num:'LDA'+str(num)},inplace=True)
    deviceid_packages=pd.concat([deviceid_packages,docresDf],axis=1)
    temp=deviceid_packages.drop(['apps','appsList'],axis=1)
    deviceIdAll=pd.merge(deviceIdAll,temp,on=['device_id'],how='left')

    #5.增加apps 的标签信息，作为文本处理
   dfs = []
   allCol=tempAppSplit.columns.tolist()
       for col in allCol:
   #        dfs.append(tempAppSplit[col])
           val = tempAppSplit[col].map(s).rename(colB+'_'+ col)
   #        dfs.append(pd.Series(np.where(val.notnull(), a[col], np.nan), name='keyB_' + col))
           dfs.append(val)
       
    newAppStat = pd.concat(dfs, axis=1)

    for colB in ['upperCategory','detailCategory']:
        s = package_label.set_index('app_id')[colB]

    print('开始对APP类别进行统计或者直接读入生成的中间文件\n')  

    listUpper=[]
    for i in range(174):
        listUpper.append('upperCategory_apps_'+str(i))
    upperAppStat=newAppStat[listUpper]
    listDetail=[]
    for i in range(174):
        listDetail.append('detailCategory_apps_'+str(i))
    detailAppStat=newAppStat[listDetail]
    
#    deviceIdAll['upperLabels']=upperAppStat.\
#    apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
    deviceIdAll['upperLabels']=upperAppStat.\
    apply(lambda x: ','.join(x.fillna(str(-1)).astype(str)),axis=1)
    
#    deviceIdAll['detailLabels']=detailAppStat.\
#    apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
    deviceIdAll['detailLabels']=detailAppStat.\
    apply(lambda x: ','.join(x.fillna(str(-1)).astype(str)),axis=1)
    
    #7 增加 app_id的降维信息(这个是device_id 使用过app_id 的信息)
    app_idS=deviceIdAll['app_id'].apply(lambda x:x.split(','))
    appIds=app_idS.apply(lambda x:' '.join(x)).tolist()
    vectorizer=CountVectorizer()
    cntTf = vectorizer.fit_transform(appIds)
    word=vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_topics=5,
                                    learning_offset=50.,
                                    random_state=1028)
    docres = lda.fit_transform(cntTf)
    docres= pd.DataFrame(data=docres)
    docresDf=docres
    for num in range(5):
        docresDf.rename(columns={num:'appIdLDA'+str(num)},inplace=True)

    deviceIdAll=pd.concat([deviceIdAll,docresDf],axis=1)
    
    #8 增加useTime LDA
    useTimeCsr= addAppUseTime(deviceIdAll,deviceid_package_start_close)
    #使用LDA进行降维，提高特征的粒度
    lda = LatentDirichletAllocation(n_topics=5,
                                   learning_offset=50.,
                                   random_state=2018)
    useTimeCsr2 = lda.fit_transform(useTimeCsr)
    useTimeCsr2= pd.DataFrame(useTimeCsr2)

    for num in range(5):
        useTimeCsr2.rename(columns={num:'useTimeLDA'+str(num)},inplace=True)
    useTimeCol= useTimeCsr2.columns.tolist()
    for col in useTimeCol:
        deviceIdAll[col]=useTimeCsr2[col]
        

    #将用户所有的使用APP 时间进行统计并存放到一个稀疏矩阵
    useTimeCsr= addAppUseTime(deviceIdAll,deviceid_package_start_close)


    train = deviceIdAll[deviceIdAll['sex'].notnull()]
    predict = deviceIdAll[deviceIdAll['sex'].isnull()]
    predict_result = predict[['device_id']]
    train_ySex = pd.Categorical(train['sex'])
    train_yAge = pd.Categorical(train['age'])
    train_y = pd.Categorical(train['sex_age'])
    train_x=train.drop(['sex','age','sex_age'],axis=1)
    predict_x=predict.drop(['sex','age','sex_age'],axis=1)

#    for i in origin_cate_list:
#        deviceIdAll[i] = deviceIdAll[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
    
    # 默认加载 如果 增加了cate类别特征 请改成false重新生成
    cate_feature = [ 'brand', 'model']
    if os.path.exists( './feature/base_train_csr.npz') and True:
        print('load_csr---------')
        base_train_csr = sparse.load_npz('./feature/base_train_csr.npz').tocsr().astype('bool')
        base_predict_csr = sparse.load_npz( './feature/base_predict_csr.npz').\
        tocsr().astype('bool')
    else:
        base_train_csr = sparse.csr_matrix((len(train_x), 0))
        base_predict_csr = sparse.csr_matrix((len(predict_x), 0))
    
        enc = OneHotEncoder()
        for feature in cate_feature:
            enc.fit(deviceIdAll[feature].values.reshape(-1, 1))
            base_train_csr = sparse.hstack((base_train_csr, enc.transform(
                    train_x[feature].values.reshape(-1, 1))), 'csr',
                                           'bool')
            base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(
                    predict_x[feature].values.reshape(-1, 1))),
                                             'csr','bool')
        print('one-hot prepared !')
    
        cv = CountVectorizer(min_df=1)#20

        for feature in ['apps','startHours','endHours','top5Apps','upperLabels',
                        'detailLabels','app_id']:
            deviceIdAll[feature] = deviceIdAll[feature].astype(str)
            cv.fit(deviceIdAll[feature])
#            for i in range(5):
            base_train_csr = sparse.hstack((base_train_csr, cv.transform(
                    train_x[feature].astype(str))), 'csr', 'bool')
            base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(
                    predict_x[feature].astype(str))), 'csr','bool')
        print('cv prepared !')
    
        sparse.save_npz( './feature/base_train_csr.npz', base_train_csr)
        sparse.save_npz( './feature/base_predict_csr.npz', base_predict_csr)

    num_feature= useTimeCol+ ['app_length','LDA0','LDA1','LDA2','LDA3','LDA4',
                              'appIdLDA0','appIdLDA1','appIdLDA2','appIdLDA3','appIdLDA4',]
    train_csr = sparse.hstack(
        (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype(
        'float32')
    predict_csr = sparse.hstack(
        (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')
    #额外添加useTimeCsr
    train_csr = sparse.hstack((train_csr,useTimeCsr[0:len(train_x)]),'csr').astype('float32')
    predict_csr=sparse.hstack((predict_csr,useTimeCsr[len(train_x):]),'csr').astype('float32')
    print(train_csr.shape)
    
    
    #使用两个lgb进行分别预测性别、年龄
    print('predict sex...')
    feature_select = SelectPercentile(chi2, percentile=95)
    feature_select.fit(train_csr, train_y)
    train_csrSex = feature_select.transform(train_csr)
    predict_csrSex = feature_select.transform(predict_csr)
    print('feature select')
    print('train_csrSex',train_csrSex.shape)
    
    
    lgb_model = lgb.LGBMClassifier(
        boosting_type='gbdt', 
#        num_leaves=32, 
#        reg_alpha=0, 
#        reg_lambda=0.1,
        max_depth= -1, 
#        min_data_in_leaf=10,
        n_estimators=5000, 
        subsample=0.85, 
        colsample_bytree=0.5, 
#        subsample_freq=1,
        learning_rate=0.05, 
        num_leaves= 32,
        random_state=2018, 
        n_jobs=-1,
        verbose_eval= 10,
#        device='gpu',
    )
    predict_resultSex =pd.DataFrame(0, index=np.arange(len(predict_x)), columns=list(np.arange(2)))
    train_resultSex= pd.DataFrame(0, index=np.arange(len(train_x)), columns=list(np.arange(2)))
    nfolds=5
    skf = StratifiedKFold(n_splits=nfolds, random_state=2018, shuffle=True)
    best_score = []
    saveSex=False
    for index, (train_index, test_index) in enumerate(skf.split(train_csrSex, train_ySex)):
        lgb_model.fit(train_csrSex[train_index], train_ySex[train_index],
                      eval_set=[(train_csrSex[train_index], train_ySex[train_index]),
                                (train_csrSex[test_index], train_ySex[test_index])], 
                      early_stopping_rounds=50)
        best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
        print(best_score)
        
        if (saveSex):
            pred=lgb_model.predict(train_csrSex[test_index])
            indices = [i for i,v in enumerate(pred) if pred[i]!= train_ySex[i]]
            wrongSexDf=train_x.iloc[indices,:]
            wrongSexDf['pred']=pred[indices]
            wrongSexDf['trueLabel']= train_ySex[indices]
            wrongSexDf.to_pickle('中间文件/wrongSexDf.pickle')
            saveSex=False
            
        test_pred = pd.DataFrame(lgb_model.predict_proba(predict_csrSex, 
                                         num_iteration=lgb_model.best_iteration_))
#        print('test mean:', test_pred.mean())
        for col in predict_resultSex.columns.tolist():
            predict_resultSex[col] = (predict_resultSex[col] + test_pred[col])
           
        
        train_pred = pd.DataFrame(lgb_model.predict_proba(train_csrSex, 
                                         num_iteration=lgb_model.best_iteration_))
        for col in train_resultSex.columns.tolist():
            train_resultSex[col] = (train_resultSex[col] + train_pred[col])
            
    print(np.mean(best_score))
    for col in predict_resultSex.columns.tolist():
            predict_resultSex[col] = predict_resultSex[col]/nfolds
    for col in train_resultSex.columns.tolist():
            train_resultSex[col] = train_resultSex[col]/nfolds
            
    predict_resultSex= predict_resultSex.rename(columns={0:'sex1',1:'sex2'})
    predict_csrAge = sparse.hstack(
        (sparse.csr_matrix(predict_resultSex[['sex1','sex2']]), predict_csr), 'csr').astype('float32')
    
    train_resultSex= train_resultSex.rename(columns={0:'sex1',1:'sex2'})
    train_csrAge = sparse.hstack(
        (sparse.csr_matrix(train_resultSex[['sex1','sex2']]), train_csr), 'csr').astype('float32')
    
    print('predict Age...')
    feature_select = SelectPercentile(chi2, percentile=95)
    feature_select.fit(train_csrAge, train_yAge)
    train_csrAge = feature_select.transform(train_csrAge)
    predict_csrAge = feature_select.transform(predict_csrAge)
    print('feature select')
    print('train_csrAge',train_csrAge.shape)
    
    lgb_model = lgb.LGBMClassifier(
        boosting_type='gbdt', 
#        num_leaves=32, 
#        reg_alpha=0, 
        reg_lambda=0.1,
        max_depth= -1, 
#        min_data_in_leaf=10,
        n_estimators=5000, 
        subsample=0.9, 
        colsample_bytree=0.8, 
#        subsample_freq=1,
        learning_rate=0.05, 
        num_leaves=32,
        random_state=2018, 
        n_jobs=-1,
        verbose_eval= 10,
#        device='gpu',
    )
    predict_resultAge =pd.DataFrame(0, index=np.arange(len(predict_x)), columns=list(np.arange(11)))
    train_resultAge= pd.DataFrame(0, index=np.arange(len(train_x)), columns=list(np.arange(11)))
    
#    nfolds=3
    skf = StratifiedKFold(n_splits=nfolds, random_state=2018, shuffle=True)
    best_score = []
    saveAge=False
    for index, (train_index, test_index) in enumerate(skf.split(train_csrAge, train_yAge)):
        lgb_model.fit(train_csrAge[train_index], train_yAge[train_index],
                      eval_set=[(train_csrAge[train_index], train_yAge[train_index]),
                                (train_csrAge[test_index], train_yAge[test_index])], early_stopping_rounds=50)
        best_score.append(lgb_model.best_score_['valid_1']['multi_logloss'])
        print(best_score)
        
        if (saveAge):
            pred=lgb_model.predict(train_csrAge[test_index])
            indices = [i for i,v in enumerate(pred) if pred[i]!= train_yAge[i]]
            wrongAgeDf=train_x.iloc[indices,:]
            wrongAgeDf['pred']=pred[indices]
            wrongAgeDf['trueLabel']= train_yAge[indices]
            wrongAgeDf.to_pickle('中间文件/wrongAgeDf.pickle')
            saveAge=False
        
        test_pred = pd.DataFrame(lgb_model.predict_proba(predict_csrAge, 
                                         num_iteration=lgb_model.best_iteration_))
        for col in predict_resultAge.columns.tolist():
            predict_resultAge[col] = (predict_resultAge[col] + test_pred[col])
    print(np.mean(best_score))
    for col in predict_resultAge.columns.tolist():
            predict_resultAge[col] = predict_resultAge[col]/nfolds
    print('train done!')
            
    predict_resultAge.columns=train_yAge.categories
    predict_resultAge=predict_resultAge[[ '0', '1','2','3','4','5','6','7','8','9','10']]
    pre_x= pd.concat([predict_resultSex,predict_resultAge],axis=1)
    
    for sexCol in ['1','2']:
        for ageCol in ['0', '1','2','3','4','5','6','7','8','9','10']:
            pre_x[sexCol+'-'+ageCol]=pre_x['sex'+sexCol]*pre_x[ageCol]
            
    sub=pre_x[['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', 
             '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]
    sub['DeviceID']=predict['device_id'].values
    sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', 
             '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]
    sub.to_csv( "./result/lgb_LgbSparse"+version+"(_2lgb)"+"nfolds"+str(nfolds)+".csv" , 
                          index=False)
    print('write result successfully!')