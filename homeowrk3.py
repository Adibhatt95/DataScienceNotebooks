import pandas as pd 
import numpy as np 
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt

data = pd.read_csv("cell2cell_data.csv",sep='/t',engine='python')
ads = pd.DataFrame(data)
#print(ads)
# df = pd.DataFrame(np.random.rand(39858,12))
# msk = np.random.rand(len(df)) < 0.8
# train = df[msk]
# test = df[~msk]

def train_validate_test_split(df, train_percent=.8, validate_percent=.0, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    #validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[train_end:]]
    return train, test

train, test = train_validate_test_split(ads)
#print(train)

#print("and this is test:")

#print(test)
#print(test.size())
test = pd.DataFrame(test)
test = test[test.columns[0]].str.split(',', expand=True)
train = train[train.columns[0]].str.split(',',expand=True)
y_test = test[test.columns[11]]
x_test = test.loc[:,test.columns[0]:test.columns[10]]
# print(x_test)
y_train = train[train.columns[11]]
x_train = train.loc[:,train.columns[0]:train.columns[10]]
# #y = test[test.columns[11]]
# print(y_train)
# print(test)

model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)

y_predict = model.predict(x_test)

print(y_predict)
treeObj = model.tree_
print(treeObj)
print(treeObj.node_count)
print(accuracy_score(y_test,y_predict))
print(model.feature_importances_)

# fig, ax = plt.subplots()
# width = 0.35
# ax.bar(np.arange(12), model.feature_importances_, width, color='r')
# ax.set_xticks(np.arange(len(model.feature_importances_)))
# ax.set_xticklabels(train.drop(lab,1).columns.values,rotation=90)
# plt.title('Feature Importance from DT')
# ax.set_ylabel('Normalized Gini Importance')

ads = ads[ads.columns[0]].str.split(',',expand=True)
adsChurn0 = ads.loc[ads[ads.columns[11]] == '1']
adsChurn0Col1 = adsChurn0[adsChurn0.columns[0]]
adsChurn0Col2 = adsChurn0[adsChurn0.columns[1]]
adsChurn0Col5 = adsChurn0[adsChurn0.columns[4]]

ads = adsChurn0
#print(adsChurn0)
# corr12 = ads[ads.columns[0]].corr(ads[ads.columns[1]])
# corr25 = ads[ads.columns[1]].corr(ads[ads.columns[4]])
# corr15 = ads[ads.columns[0]].corr(ads[ads.columns[4]])
col1 = pd.to_numeric(ads[ads.columns[0]])
col2 = pd.to_numeric(ads[ads.columns[1]])
col5 = pd.to_numeric(ads[ads.columns[4]])
print("{} {} {}".format(col1.mean(),col2.mean(),col5.mean()))
corr12 = col1.corr(col2)
corr25 = col2.corr(col5)
corr15 = col5.corr(col1)

#print("{} {} {}".format(corr12,corr25,corr15))

# model2 = tree.DecisionTreeClassifier(criterion='entropy')
# model2.fit(x_train,y_train)
# y_predict = model2.predict(x_test)
# #print(y_predict)

def testTrees(X_train,Y_train,X_test,Y_test,dep,leaf,auc):
    print("{} {}".format(dep,leaf))
    clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=leaf,min_samples_split=dep)
    clf = clf.fit(X_train,Y_train)
    print(clf)
    if (auc==0):
        cm = confusion_matrix(clf.predict(X_test),y_test)
        return (cm[0][0]+cm[1][1])/float(sum(cm))
    else:
        Y_predict = clf.predict(X_test)
        print(Y_predict)
        return accuracy_score(Y_test,Y_predict)
    

depths=[8000,5000,3000,1000]
leaves=[50,100,200,300,500,800,3000]

#Run all of the options
run=1
res=dict()
if (run==1):
    #Initialize dictionary of results
    for d in depths:
        res[d]=list()

    #Now train and get results for each option
    for d in depths:
        for l in leaves:
            res[d].append(testTrees(x_train,y_train,x_test,y_test,d,l,1))

print(res[depths[0]])
print(res[depths[1]])
print(res[depths[2]])
print(res[depths[3]])
#Now plot            
fig = plt.figure()
ax=fig.add_subplot(111)
plt.plot(leaves,res[depths[0]],'b-',label='Depth={}'.format(depths[0]))
plt.plot(leaves,res[depths[1]],'r-',label='Depth={}'.format(depths[1]))
plt.plot(leaves,res[depths[2]],'y-',label='Depth={}'.format(depths[2]))
plt.plot(leaves,res[depths[3]],'g-',label='Depth={}'.format(depths[3]))
plt.legend(loc=4)
ax.set_xlabel('Min Leaf Size')
ax.set_ylabel('Test Set AUC')
plt.title('Holdout AUC by Hyperparameters')
plt.show()