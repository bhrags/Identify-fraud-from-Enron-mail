
# coding: utf-8

# In[46]:
from __future__ import division
import sys
import pickle
import numpy
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.feature_selection import SelectFromModel,SelectKBest
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import feature_selection,svm,tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit,train_test_split
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,make_scorer,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from time import time
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def plotting(data,x,y,xlab,ylab):
    for point in data:
        X = point[x]
        Y = point[y]
        plt.scatter( X, Y )
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    
def replaceNaNZero(value):
    if value =='NaN':
        value = 0
    return value

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

def exploredata(data_dict):   
    #Exploratory Data Analysis
    print "EXPLORING THE DATA SET"
    print "Number of data points :", len(data_dict)
    #Number of features
    lengths = [len(v) for v in data_dict.values()]
    print "Number of features for each data point:", len(data_dict.values()[0])

    #Number of records which are POIs and non POIs
    poi_counter = 0
    for i in data_dict.values():
        if i['poi']!= 0:
            poi_counter = poi_counter+1
    print "Number of POIs:", poi_counter
    print "Number of non POIs:", len(data_dict)-poi_counter

    # checking for NaNs
    Nandict = {}
    for keys,values in data_dict.items():
        for key,value in values.items():
            if value == "NaN":
                if Nandict.get(key) != None:
                    Nandict[key]= Nandict[key]+1
                else:
                    Nandict[key] =1
    print "Number of records with NaN values against various Features:","\n",Nandict

### Task 2: Identifying and Removal/treatment of outliers
def outlier_handling(data_dict,features_list):
    print "--------------------OUTLIER IDENTIFICATION AND TREATMENT -----------------------------------"
    #Is there any record with all NaN's or not a name
    to_remove =[]

    for key,valuedict in data_dict.items(): 
        counter =0
        if  len(key) ==29: #The travel agency entry
            print "Key with not a regular Name:",key
            to_remove.append(key)
        for value in valuedict.values():
            if value !='NaN':
                continue
            else :
                counter = counter+1
            if counter == 19:
                print "Key having all NaNs:",key
                to_remove.append(key)
    #Removing keys with no financial data
    finance_feature = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',                        'restricted_stock_deferred',                       'deferred_income', 'total_stock_value','expenses', 'exercised_stock_options',                       'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
    for key in data_dict:
        counter =0
        for feature in finance_feature:
            if data_dict[key][feature] == 'NaN':
                counter =counter+1
            if counter == 10:
                #print data_dict[key]             
                to_remove.append(key)
    for i in to_remove:
        data_dict.pop(i,0)
    
    #Detecting other outlier based on values
    data = featureFormat(data_dict, features_list, sort_keys = True)
    salary = features_list.index('salary')
    bonus = features_list.index('bonus')
    plotting(data,salary,bonus,features_list[1],features_list[2])

    #identifying the outlier value and key
    index  = numpy.argsort(data[:,1])
    HV_index = index[len(index)-1]
    print "Outlier value:", data[HV_index][1]
    for i in data_dict:
        if data_dict[i]['salary'] == data[HV_index][1]:
            key = i
    print "Outlier key :",key

    #Removing the outlier
    data_dict.pop('TOTAL',0)
       
    data = featureFormat(data_dict, features_list, sort_keys = True)

    #Scatter plot after removing outliers
    salary = features_list.index('salary')
    bonus = features_list.index('bonus')
    plotting(data,salary,bonus,features_list[1],features_list[2])

    # detecting people with salary greater than 1 million and bonus greater than 5000000
    for i in data_dict:
        if data_dict[i]['bonus'] > 5000000 and  data_dict[i]['salary']>1000000:
            if data_dict[i]['bonus'] != 'NaN' and  data_dict[i]['salary'] != 'NaN':
                print "People with high bonus and high salary:",i,data_dict[i]['bonus']

    # Total payments versus salary
    salary = features_list.index('salary')
    total_payments = features_list.index('total_payments')
    plotting(data,salary,total_payments,features_list[1],features_list[3])

    for i in data_dict:
        if data_dict[i]['total_payments'] > 50000000 and  data_dict[i]['salary']>1000000:
            if data_dict[i]['total_payments'] != 'NaN' and  data_dict[i]['salary'] != 'NaN':
                print "People with high total payments and high salary:",i,data_dict[i]['total_payments']


    # Expenses versus salary
    salary = features_list.index('salary')
    expenses = features_list.index('expenses')
    plotting(data,salary,expenses,features_list[1],features_list[4])


    #Total Stock value versus salary
    salary = features_list.index('salary')
    total_stock_value = features_list.index('total_stock_value')
    plotting(data,salary,total_stock_value,features_list[1],features_list[5])

    for i in data_dict:
        if data_dict[i]['total_stock_value'] > 30000000 and  data_dict[i]['salary']>1000000:
            if data_dict[i]['total_stock_value'] != 'NaN' and  data_dict[i]['salary'] != 'NaN':
                print "People with high total stock value and high salary:",i,data_dict[i]['total_stock_value']
    return data_dict

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

#Creating fraction of POI interaction/total interaction
def feature_engg(data_dict):
    print "--------------------FEATURE ENGINEERING -----------------------------------"
    for key in data_dict:
        from_this_person_to_poi  = replaceNaNZero (data_dict[key]['from_this_person_to_poi'])
        shared_receipt_with_poi = replaceNaNZero(data_dict[key]['shared_receipt_with_poi'])
        from_poi_to_this_person = replaceNaNZero(data_dict[key]['from_poi_to_this_person'])

        POI_interaction = from_this_person_to_poi+shared_receipt_with_poi+from_poi_to_this_person
        to_messages = replaceNaNZero (data_dict[key]['to_messages'])
        from_messages = replaceNaNZero(data_dict[key]['from_messages'])

        Mail_interaction = to_messages +from_messages 
        data_dict[key]["Mail_interaction"] = Mail_interaction 
        if  (POI_interaction ==0) &  (Mail_interaction ==0) :
            data_dict[key]["fraction_POI_interaction"]= 0
        else :
            data_dict[key]["fraction_POI_interaction"]= float(POI_interaction)/float(Mail_interaction)

    #Plotting the calculated fraction data
    my_features_list = ['poi','fraction_POI_interaction','Mail_interaction']
    data = featureFormat(data_dict, my_features_list, sort_keys = True)
    plt.scatter(data[:,2],data[:,1],s=50,c=data[:,0], marker = 'o', cmap = plt.cm.coolwarm );
    plt.xlabel("Total Mails")
    plt.ylabel("Fraction of POI Mails")
    plt.show()

    #Checking if any of the values are greater than 1 for the fraction and adjusting the same
    for key in data_dict:
        if data_dict[key]['fraction_POI_interaction'] >1:
            #print data_dict[key]
            data_dict[key]['fraction_POI_interaction'] =1

    # Creating a feature TOTAL compensation = Total payments + Total Stock value
    for key in data_dict:
        TP  = replaceNaNZero (data_dict[key]['total_payments'])
        TSV = replaceNaNZero(data_dict[key]['total_stock_value'])
        data_dict[key]["total_compensation"]  = TP+TSV
    my_features_list = ['poi','salary','total_compensation']
    data = featureFormat(data_dict, my_features_list, sort_keys = True)
    plt.scatter(data[:,1],data[:,2],s=50,c=data[:,0], marker = 'o', cmap = plt.cm.coolwarm );
    plt.xlabel("Salary")
    plt.ylabel("Total Compensation")
    plt.show()
    
    #Converting the NaN salary values to median values 
    salsum = []
    for key in data_dict:
        if data_dict[key]['salary'] != 'NaN' :
            salsum.append(data_dict[key]['salary'])
    Median = numpy.median(numpy.array(salsum))
    for key in data_dict:
        if data_dict[key]['salary'] == 'NaN' :
            data_dict[key]['salary'] =Median
    return data_dict

def feature_selection(data_dict):

### Extract features and labels from dataset for local testing
    print "--------------------FEATURE SELECTION -----------------------------------"
    my_dataset = data_dict
    features_list =['poi','salary','deferral_payments','total_payments','loan_advances','bonus',                    'restricted_stock_deferred','deferred_income','total_stock_value','expenses',                    'exercised_stock_options','other','long_term_incentive','restricted_stock',                    'to_messages','from_poi_to_this_person','from_messages',                    'from_this_person_to_poi','shared_receipt_with_poi','fraction_POI_interaction','Mail_interaction']
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    #FEATURE SELECTION
    #The feature importances have some variability in its output hence getting the output  
    #and fixing the features_list based on the below output
    #checking Kbest
    print "From KBEST- Scores:",SelectKBest(k='all').fit(features,labels).scores_
    clf  =tree.DecisionTreeClassifier()
    clf = clf.fit (features,labels)
    print "Importance of features:",clf.feature_importances_
    model = SelectFromModel(clf,prefit=True)
    mask= numpy.nonzero(model.get_support())[0]
    print mask
    only_features = features_list[1:len(features_list)]
    '''print "Chosen features by algorithm:","\n"
    for i in range(len(only_features)):
        if i in mask:
            print only_features[i]'''
    #Freezing the feature list based on multiple iterations of the above and the choices given by KBest
    features_list =['poi','salary','total_payments','loan_advances','bonus',                    'deferred_income','total_stock_value','expenses',                    'exercised_stock_options','long_term_incentive','other','restricted_stock',                    'from_poi_to_this_person',                    'from_this_person_to_poi','shared_receipt_with_poi','fraction_POI_interaction']
    print "Chosen Features:",features_list[1:len(features_list)]
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    return features_list,features,labels

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

def classifier_trials(features,labels):
    print "------------------NON TUNED ACCURACY SCORES---------------------------------"
    
    features_train, features_test, labels_train, labels_test =         train_test_split(features, labels, test_size=0.3, random_state=42)
    #NAIVE BAYES
    clf = GaussianNB()
    clf = clf.fit(features_train,labels_train)
    labels_predict = clf.predict(features_test)
    print "Accuracy Score of Naive Bayes:",accuracy_score(labels_predict,labels_test)
     

    #SVC  
    clf = svm.SVC()
    clf = clf.fit(features_train,labels_train)
    labels_predict = clf.predict(features_test)
    print "Accuracy Score of SVC:",accuracy_score(labels_predict,labels_test)

    #Tree
    clf =tree.DecisionTreeClassifier()
    clf = clf.fit(features_train,labels_train)
    labels_predict = clf.predict(features_test)
    print "Accuracy Score of Tree:",accuracy_score(labels_predict,labels_test)

    #Random forest  
    clf = RandomForestClassifier()
    clf = clf.fit(features_train,labels_train)
    labels_predict = clf.predict(features_test)
    print "Accuracy Score of Random Forest:",accuracy_score(labels_predict,labels_test)
    


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#TUNING THE PARAMETERS
# Example starting point. Try investigating other evaluation techniques!
def NBtuning(my_dataset,feature_list):
    def custom_scorer(labels_test, labels_predict):
        precision,recall = precision_recall(labels_test, labels_predict)
        min_score = min(precision,recall)
        return min_score
    def precision_recall(labels_test, labels_predict):
        precision = precision_score(labels_test, labels_predict)
        recall = recall_score(labels_test, labels_predict)
        return precision,recall
    #For NB corelated features pruned
    features_list =['poi','salary','loan_advances','bonus',                    'deferred_income','expenses','director_fees','deferral_payments',                    'exercised_stock_options','long_term_incentive','other','restricted_stock',                    'fraction_POI_interaction']
    data = featureFormat(my_dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    score = make_scorer(custom_scorer,greater_is_better = True)
    
    #using PCA to check on the components
    pca = PCA()
    pca.fit(features)
    print "PCA Explained Variances:",pca.explained_variance_
    #Plotting the PCA
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    plt.show()
    t0=time()
    n_components = [1,2,3,4,5,6,7,8]
    NB = GaussianNB()
    param_grid = {
                   "pca__n_components" : n_components,
                    "pca__random_state": [42]
                }

    pipe = Pipeline(steps=[('pca', pca),('scale',scaler),('NB', NB)])
    sss = StratifiedShuffleSplit(labels,n_iter=100,random_state=42)
    GS = GridSearchCV(pipe,param_grid=param_grid,scoring = score,cv=sss)
    GS = GS.fit(features,labels)
    print "------------------------------------TUNED NB------------------------------"
    print GS.best_params_
    NBclf = GS.best_estimator_
    labels_predict = NBclf.predict(features)
    print "Accuracy Score :",accuracy_score(labels_predict,labels)
    NBscore= precision_recall_fscore_support(labels_predict,labels,average="macro")
    print "Precision:", NBscore[0]
    print "Recall:", NBscore[1]
    print "F1score:",NBscore[2]
    t1 = time()
    print "Time taken for NB:",t1-t0
    return NBclf
def DTtuning(features,labels):
    print "------------------------------------TUNING PROCESS------------------------------"
    
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    t0= time()
    param_grid ={   
                    "DT__criterion": ["gini", "entropy"],
                    "DT__max_depth":[2,3,4,5],
                    "DT__min_samples_leaf": [1,3,5],
                    "DT__class_weight":["balanced"],
                    "DT__random_state": [42],
                    "DT__max_features":["auto","log2"]
                                   
    }
    DT = tree.DecisionTreeClassifier()
    pipe = Pipeline(steps=[('scale',scaler),('DT', DT)])
    sss = StratifiedShuffleSplit(labels, n_iter=100 , random_state=42)
    GS = GridSearchCV(pipe,param_grid= param_grid,scoring = "f1",cv=sss)
    GS = GS.fit(features,labels)
    print "------------------------------------TUNED DT------------------------------"
    print GS.best_params_
    DTclf = GS.best_estimator_
    labels_predict = DTclf.predict(features)
    print "Accuracy Score :",accuracy_score(labels_predict,labels)
    DTscore= precision_recall_fscore_support(labels_predict,labels,average="macro")
    print "Precision:", DTscore[0]
    print "Recall:", DTscore[1]
    print "F1score:",DTscore[2]
    t1 = time()
    print "Time taken for DT:",t1-t0
    return DTclf
def SVCtuning(features,labels):
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    pca = PCA()
    
    #For SVC using PCA and gridsearchgeatures

    pca.fit(features)
    print "PCA Explained Variances:",pca.explained_variance_
    #Plotting the PCA
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    plt.show()

    #coosing number of components based on graph plotted
    n_components = [2,3,4,5,8]
    t0= time()
    svc = svm.SVC()
    param_grid = {
        'pca__n_components' : n_components,
        'svc__C': [1, 10, 100, 1000],
        'svc__gamma': [0.1,0.01,0.001, 0.0001],
        'svc__kernel': ['rbf','poly'],
        'svc__random_state':[42],
        }
    
    pipe = Pipeline(steps=[('pca', pca),('scale',scaler),('svc', svc)])
    sss = StratifiedShuffleSplit(labels)
    GS = GridSearchCV(pipe,param_grid=param_grid,cv=sss,scoring ="f1")
    GS = GS.fit(features,labels)
    print "------------------------------------TUNED SVC------------------------------"
    print GS.best_params_
    svcclf = GS.best_estimator_
    #Prediction using the best parameters
    labels_predict = svcclf.predict(features)
    print "Accuracy Score :",accuracy_score(labels_predict,labels)
    svcscore= precision_recall_fscore_support(labels_predict,labels,average="macro")
    print "Precision:", svcscore[0]
    print "Recall:", svcscore[1]
    print "F1score:",svcscore[2]
    t1 = time ()
    print "time taken for SVC:",t1-t0
    return svcclf

def RFtuning(features,labels):
    t0=time()
    rfc = RandomForestClassifier()
    param_grid = {"max_depth": [2,3,None],
                  "max_features": ["auto","log2",0.9],
                  "min_samples_split": [2,4,6],
                  "min_samples_leaf": [1,10,20],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "n_estimators": [10,15],
                  "bootstrap":[True],
                  "oob_score": [True],
                  "n_jobs": [-1],
                  "random_state":[50]}
    sss = StratifiedShuffleSplit(labels,n_iter=100,random_state=42)
    GS = GridSearchCV(estimator = rfc,param_grid=param_grid,scoring="f1",cv=sss)
    GS = GS.fit(features,labels)
    print "------------------------------------TUNED RFC------------------------------"
    print GS.best_params_
    rfcclf = GS.best_estimator_
    #Prediction using the best parameters
    labels_predict = rfcclf.predict(features)
    print "Accuracy Score :",accuracy_score(labels_predict,labels) 
    rfcscore= precision_recall_fscore_support(labels_predict,labels,average="macro")
    print "Precision:", rfcscore[0]
    print "Recall:", rfcscore[1]
    print "F1score:",rfcscore[2]
    t2= time()
    print "Time taken for RFC:",t2-t0
    return rfcclf
    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
def main():
    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    exploredata(data_dict)
    temp_feature_list = ['poi','salary','bonus','total_payments','expenses','total_stock_value']
    data_dict =outlier_handling(data_dict,temp_feature_list)
    data_dict = feature_engg(data_dict)
    features_list,feature,labels = feature_selection(data_dict)
    classifier_trials(feature,labels)
    clf =DTtuning(feature,labels)
    #clf = NBtuning(my_dataset,features_list)
    #clf = SVCtuning(feature,labels)
    #clf = RFtuning(feature,labels)
    dump_classifier_and_data(clf, data_dict, features_list)


if __name__ == '__main__':
    main()


