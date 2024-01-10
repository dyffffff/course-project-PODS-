#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:20:16 2023

@author: dingyufu
"""

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import permutation_test
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import ks_2samp, norm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from scipy.special import expit
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file = pd.read_csv(
    "/Users/dingyufu/Desktop/data science/principle of DS/capstone project/spotify52kData.csv")


file = file.dropna()


def plothist(pdcolumns, x):
    plt.hist(pdcolumns)
    plt.xlabel(x)
    plt.ylabel("count")
    plt.axvline(pdcolumns.mean(), color='red',
                linestyle='dashed', linewidth=2, label='Mean')
    plt.title("the distribution of "+x)
    plt.show()


# %% Q1
Q1A = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# draw distribution of each feature
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
for i in range(10):
    axes[i].hist(file[Q1A[i]], bins=100, color='blue', alpha=0.7)
    axes[i].set_title("Distribution of "+Q1A[i])
    axes[i].set_xlabel(Q1A[i])
    axes[i].set_ylabel('Count')


# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
# %%
# determine if danceability is normally distributed
df1 = pd.DataFrame(index=Q1A,columns=['p value'])
for i in Q1A:
    statD, p_valueD = stats.kstest(file[i], "norm", args=(
    file[i].mean(), file[i].std()))
    df1.loc[i,"p value"]=p_valueD
    

    if p_valueD < 0.05:
        print("reject the null hypothesis: The danceability data appears to be normally distributed.")
    else:
        print("fail to reject the null hypoethesis: The danceability data does not appear to be normally distributed.")

''' 
statT, p_valueT = shapiro(file["tempo"])

if p_valueT > 0.05:
    print("The tempo danceability data appears to be normally distributed.")
else:
    print("The tempo data does not appear to be normally distributed.")
 '''
# %%Q2
# z score the duration and popularity
# z score the data
data2 = file[['duration', 'popularity']]
scaler = StandardScaler()
data2Norm = pd.DataFrame(scaler.fit_transform(data2), columns=data2.columns)

# calculate correlation between duration and popularity
corre_p = data2["duration"].corr(data2["popularity"])
correNorm_p = data2Norm["duration"].corr(data2Norm["popularity"])

rho, p_value = spearmanr(data2Norm["duration"], data2Norm["popularity"])

print("the pearson correlation coefficient between duration and popularity is "+str(correNorm_p))
print("the Spearman correlation coefficient between duration and popularity is "+str(rho))

plt.scatter(data2Norm['duration'], data2Norm['popularity'])
plt.xlabel("duration")
plt.ylabel("popularity")
plt.title("correlation between duration and popularity")
plt.show()
# %%Q3
# first divide data into 2 group, with explicit and non explicit
# data3T is explicit data3F is not explicit
data3T = pd.DataFrame(columns=["explicit", "popularity"])
data3F = pd.DataFrame(columns=["nonexplicit", "popularity"])
TrueValue = 0
FalseValue = 0
for i in range(len(file)):
    if file.loc[i, "explicit"] == True:
        data3T.loc[TrueValue, "explicit"] = file.loc[i, "explicit"]
        data3T.loc[TrueValue, "popularity"] = file.loc[i, "popularity"]
        TrueValue = TrueValue+1
    elif file.loc[i, "explicit"] == False:
        data3F.loc[FalseValue, "nonexplicit"] = file.loc[i, "explicit"]
        data3F.loc[FalseValue, "popularity"] = file.loc[i, "popularity"]
        FalseValue = FalseValue+1

plt.hist(data3T["popularity"])
plt.xlabel("popularity")
plt.ylabel("count")
plt.title("distribution of explicit popularity")
plt.axvline(data3T["popularity"].mean(), color='red',
            linestyle='dashed', linewidth=2, label='Mean')
plt.show()

plt.hist(data3F["popularity"])
plt.xlabel("popularity")
plt.ylabel("count")
plt.title("distribution of non-explicit popularity")
plt.axvline(data3F["popularity"].mean(), color='red',
            linestyle='dashed', linewidth=2, label='Mean')
plt.show()

print("median of explicit: "+str(data3T["popularity"].median()))
print("median of nonexplicit: "+str(data3F["popularity"].median()))

#%% decide which test to use
'''
welch t test; need data normally distributed; however since two groups have different
size, I can't use t test
z test requires normal distribution : consider the group with more group as population

Mnn-Whitney U test might be a choice
'''
'''
d3Fmean =data3F["popularity"].mean()
d3Tmean =data3T["popularity"].mean()

D3FStd = data3F["popularity"].std()
SEM = D3FStd/math.sqrt(5597)
zScore = (d3Tmean-d3Fmean)/SEM
zScore roughly -10
'''
# perform Mann-Whitney U test abput popularity between explicit and non explicit to know whether if explicit
# has effect on popularity
# define statistic, difference bewteen mean

from scipy.stats import mannwhitneyu

def statistic(x, y, axis):
    return np.mean(x, axis=axis)-np.mean(y, axis=axis)


U_test_E, p_val_E = stats.mannwhitneyu(np.array(data3F["popularity"], dtype=float),
                                    np.array(data3T["popularity"], dtype=float)
                                    )
# p_val = 2.9549869188182215e-21<alpha =0.05  we drop the hypothesis that explicit is more popular than non-explicit is due to chance

# perform permutation test since the data is not normally distributed about popularity between explicit and non explicit
resultE = permutation_test((np.array(data3F["popularity"], dtype=float),
                            np.array(data3T["popularity"], dtype=float)), statistic, n_resamples=1000)


print("this is p value of U test "+str(p_val_E))
print("below is p value of the permutation test")
print(resultE.pvalue)
# p-val from permutation test is 0.001998001998001998 also smaller than 0.05, drop the null hypothesis

print("this is mean of non-explicit: "+str(data3F["popularity"].mean()))
print("this is mean of explicit: "+str(data3T["popularity"].mean()))

# %%Q4
'''
data4Ma =pd.DataFrame(columns=["major","popularity"])
data4Mi = pd.DataFrame(columns=["minor","popularity"])

MiValue = 0
MaValue = 0
#divide key into minor and major
for i in range(len(file)):
    if file.loc[i,"key"] in [0 , 2 , 4 , 5 , 7 , 9 , 11]:
        data4Ma.loc[MaValue,"major"]=file.loc[i,"key"]
        data4Ma.loc[MaValue,"popularity"]=file.loc[i,"popularity"]
        MaValue = MaValue+1
    elif file.loc[i,"key"]in [ 1 , 3 , 6 , 8 , 10]:
        data4Mi.loc[MiValue,"minor"]=file.loc[i,"key"]
        data4Mi.loc[MiValue,"popularity"]=file.loc[i,"popularity"]
        MiValue = MiValue+1
'''
'''        
#perform welch t test for key
t_test_K,p_val_K = stats.ttest_ind(np.array(data4Ma["popularity"],dtype=float),
                               np.array(data4Mi["popularity"],dtype=float),
                               equal_var=False)
'''
'''
I used the wrong variable, I used key instead of mode here
p_val_key is 0.07605>0.05, therefore it is possible that minor/major key causes
popularity differences is due to chance
#for key using permutation test p value is 0.08591> 0.05, therefore we fail to reject null hypothesis
'''
data4Ma = pd.DataFrame(columns=["major", "popularity"])
data4Mi = pd.DataFrame(columns=["minor", "popularity"])

MiValue = 0
MaValue = 0
# divide key into minor and major
for i in range(len(file)):
    if file.loc[i, "mode"] == 1:
        data4Ma.loc[MaValue, "major"] = file.loc[i, "mode"]
        data4Ma.loc[MaValue, "popularity"] = file.loc[i, "popularity"]
        MaValue = MaValue+1
    elif file.loc[i, "mode"] == 0:
        data4Mi.loc[MiValue, "minor"] = file.loc[i, "mode"]
        data4Mi.loc[MiValue, "popularity"] = file.loc[i, "popularity"]
        MiValue = MiValue+1

#calculte median of each group
print("median of major: "+str(data4Ma["popularity"].median()))
print("median of minor: "+str(data4Mi["popularity"].median()))
'''
below is the mode 
'''
U_test_M, p_val_M = stats.mannwhitneyu(np.array(data4Ma["popularity"], dtype=float),
                                    np.array(data4Mi["popularity"], dtype=float))
                                    

print("this is U test p value for the mode difference")
print(p_val_M)

'''
this is p value for the mode difference
1.6610913055004772e-06
below is p value of the permutation test
0.001998001998001998
since both of them are smaller than 0.05, therefore the difference in popularity caused by minor
and major is not due to chance
'''

''''
plothist(data4Ma["popularity"],"popularity of major key")
plothist(data4Mi["popularity"],"popularity of minor key")
'''

plt.hist(data4Ma["popularity"])
plt.xlabel("popularity")
plt.ylabel("count")
plt.title("distribution of major key")
plt.axvline(data4Ma["popularity"].mean(), color='red',
            linestyle='dashed', linewidth=2, label='Mean')
plt.show()

plt.hist(data4Mi["popularity"])
plt.xlabel("popularity")
plt.ylabel("count")
plt.title("distribution of minor key")
plt.axvline(data4Mi["popularity"].mean(), color='red',
            linestyle='dashed', linewidth=2, label='Mean')
plt.show()

major_mean = data4Ma["popularity"].mean()
minor_mean = data4Mi["popularity"].mean()

# perform permutation test
# define statistic, difference bewteen mean


def statistic(x, y, axis):
    return np.mean(x, axis=axis)-np.mean(y, axis=axis)


resultK = permutation_test((np.array(data4Ma["popularity"], dtype=float), np.array(
    data4Mi["popularity"], dtype=float)), statistic, n_resamples=1000)
print("below is p value of the permutation test")
print(resultK.pvalue)


# %%Q5

# calculate correlation between energy and loudness
data5 = file[['energy', 'loudness']]
data5Norm = pd.DataFrame(scaler.fit_transform(data5), columns=data5.columns)
corre5 = data5["energy"].corr(data5["loudness"])
corre5Norm = data5Norm["energy"].corr(data5Norm["loudness"])

print("the correlation coefficient between energy and loudness is "+str(corre5Norm))

plt.scatter(data5['energy'], data5['loudness'])
plt.xlabel("energy")
plt.ylabel("loudness")
plt.title("correlation between energy and loudness")
plt.show()

# %% Q6
# single linear regression for each feature


data6R = pd.DataFrame(index=Q1A, columns=["slope", "intercept", "Rsq"])
for i in np.arange(len(Q1A)):
    print(Q1A[i])
    # split data into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        file[Q1A[i]], file["popularity"], test_size=0.2, random_state=12718276)
    #model = LinearRegression().fit(np.array(file[i]).reshape(-1,1),file["popularity"])

    model = LinearRegression().fit(np.array(X_train).reshape(-1, 1), y_train)

   # make prediction
    y_pred = model.predict(np.array(X_test).reshape(-1, 1))
   # calculte r sq
    Rsq = r2_score(y_test, y_pred)

    data6R.loc[Q1A[i], "slope"] = model.coef_
    data6R.loc[Q1A[i], "intercept"] = model.intercept_
    data6R.loc[Q1A[i], "Rsq"] = Rsq

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
for i in range(10):
    axes[i].scatter(file[Q1A[i]], file["popularity"], color='blue', alpha=0.7)
    axes[i].set_title("relationship between  "+Q1A[i]+" and popularity")
    axes[i].set_xlabel(Q1A[i])
    axes[i].set_ylabel('popularity')

    x_range = np.linspace(file[Q1A[i]].min(), file[Q1A[i]].max(), 100)
    y_range = data6R.loc[Q1A[i], "slope"] * \
        x_range + data6R.loc[Q1A[i], "intercept"]
    axes[i].plot(x_range, y_range, color='red', linewidth=2)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
# %% Q7
from sklearn.model_selection import cross_val_score
# multiregression

# split data into training and test

X_train, X_test, y_train, y_test = train_test_split(
    file[Q1A], file["popularity"], test_size=0.2, random_state=12718276)

# use training set to build model
modelFull = LinearRegression().fit(X_train, y_train)
# make prediction
y_pred = modelFull.predict(X_test)
# calculte r sq
RsqFull = r2_score(y_test, y_pred)
RMSEFull = math.sqrt(mean_squared_error(
    file["popularity"], modelFull.predict(file[Q1A])))
b0, b1 = modelFull.intercept_, modelFull.coef_

summaryLR = {"R Square": RsqFull,
             "RMSE": RMSEFull, "intercept": b0, "coef": b1}
data7 = pd.DataFrame(summaryLR, index=Q1A)
cv_multireg_score = cross_val_score(LinearRegression(),X_train,y_train)
print("this is mean crross validation score of multi regression model: "+str(cv_multireg_score.mean()))


# %% Q8

# calculate correlation between each column
correMatrix = file[Q1A].corr()

# z score the data
ZScoreData = stats.zscore(file[Q1A])

# perform PCA
pca = PCA().fit(ZScoreData)

# eigenvalues
eVal = pca.explained_variance_
# 2.73393;1.61739;1.38461---3 meaningful principal conponent based on kaiser

# caculate how much variance each PC can account for

PC_Variance_Matrix = pd.DataFrame(
    columns=["eigenvalue", "variance account for %"])
PC_Variance_Matrix["eigenvalue"] = eVal
for i in range(len(eVal)):
    PC_Variance_Matrix.loc[i, "variance account for %"] = 100*eVal[i]/sum(eVal)

# calculalate mow many important pc based on eigensum criterior
sumV = 0
indexV = 1
for i in eVal:
    sumV = sumV+100*i/sum(eVal)
    if sumV > 90:
        print("there are " + str(indexV)+" important PC based on Eigensum rule")
        break
    indexV = indexV+1
'''  
7 meaningful principal component based on eigensum principal  
since correlation between each feature is not high, pca can't reduce all 10 features into fewer
dimension

'''
# screeplot
X = np.linspace(1, len(eVal), len(eVal))
plt.bar(X, eVal, color="green")
plt.xlabel("Principal components")
plt.ylabel("Eigenvalue")
plt.title("screeplot")
for i, value in enumerate(eVal):
    plt.text(i+1, value, str(f"{value:.2f}"), ha='center', va='bottom')

# Add a horizontal line at y = 1
plt.axhline(y=1, color='red', linestyle='--',
            linewidth=2, label='Threshold at y=1')
plt.show()
# evectors  #Rows: Eigenvectors. Columns: Where they are pointing
eVec = pca.components_

# transform data into new coordinate
rotatedData = pca.fit_transform(ZScoreData)
# %% Q8 clusters
# Store our transformed data - the predictors - as x:
x_cluster = np.column_stack(
    (rotatedData[:, 0], rotatedData[:, 1], rotatedData[:, 2]))

# %% 2i) How many clusters k to ask for? Silhouette:
# How close are points to other points in the cluster, vs. the neighboring cluster, to quantify
# the arbitrariness of the clustering

# Remember: each data point gets its own silhouette coefficient ranging
# from 0 (arbitrary classification) to 1 (ideal classification).

# Init:
numClusters = 9  # how many clusters are we looping over? (from 2 to 10)
sSum = np.empty([numClusters, 1])*np.NaN  # init container to store sums

# Compute kMeans for each k:
for ii in range(2, numClusters+2):  # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters=int(ii)).fit(
        x_cluster)  # compute kmeans using scikit
    cId = kMeans.labels_  # vector of cluster IDs that the row belongs to
    # coordinate location for center of each cluster
    cCoords = kMeans.cluster_centers_
    # compute the mean silhouette coefficient of all samples
    s = silhouette_samples(x_cluster, cId)
    sSum[ii-2] = sum(s)  # take the sum
    # Plot data:
    plt.subplot(3, 3, ii-1)
    plt.hist(s, bins=20)
    plt.xlim(-0.2, 1)
    plt.ylim(0, 250)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    # sum rounded to nearest integer
    plt.title('Sum: {}'.format(int(sSum[ii-2])))
    plt.tight_layout()  # adjusts subplot

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2, numClusters, 9), sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()

# kMeans yields the coordinates centroids of the clusters, given a certain number k
# of clusters. Silhouette yields the number k that yields the most unambiguous clustering
# This number k is the maximum of the summed silhouette scores.
'''
based on silhouette, either 2 or 3 cluster is fine
'''
# %% Q9

# handle data
data9 = pd.DataFrame(columns=["valence-x", "key-y"])

data9["valence-x"] = file["valence"]
for i in range(len(file)):
    # major = 1
    if file.loc[i, "key"] in [0, 2, 4, 5, 7, 9, 11]:
        data9.loc[i, "key-y"] = 1
    #minor = 0
    if file.loc[i, "key"] in [1, 3, 6, 8, 10]:
        data9.loc[i, "key-y"] = 0
# spli data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data9["valence-x"], data9["key-y"], test_size=0.2, random_state=12718276)
# do logistic regression

x_train_array = np.array(X_train).reshape(-1, 1)
y_train_array = np.array(y_train).astype(int)
modelLog = LogisticRegression().fit(
    np.array(X_train).reshape(-1, 1), y_train_array)

x1 = np.linspace(min(X_train), max(X_train), 500)
y1 = x1 * modelLog.coef_ + modelLog.intercept_
sigmoid = expit(y1)

# Plot:
# the ravel function returns a flattened array
plt.plot(x1, sigmoid.ravel(), color='red', linewidth=3)
plt.scatter(X_train, y_train, color='black')
plt.hlines(0.5, min(X_train), max(X_train), colors='gray', linestyles='dotted')
plt.xlabel('valence')
plt.xlim([min(X_train), max(X_train)])
plt.ylabel('key')
plt.yticks(np.array([0, 1]))
plt.show()

# logistic regression doesn't work to classify key based on valence
# %%Q9 part 2

'''
this block of codes are for key instead of mode, made a mistake here
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
for i in range(10):
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(file[Q1A[i]], data9["key-y"], test_size=0.2, random_state=42)
    x_train_array2 = np.array(X_train_2).reshape(-1,1)
    y_train_array2 = np.array(y_train_2).astype(int)
    modelLog2 = LogisticRegression().fit(np.array(X_train).reshape(-1,1),y_train_array2)
'''

data9AUC = pd.DataFrame(index=Q1A, columns=["AUC"])

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
for i in range(10):
    # split data into test and train
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
        file[Q1A[i]], file["mode"], test_size=0.2, random_state=12718276)
    x_train_array2 = np.array(X_train_2).reshape(-1, 1)
    y_train_array2 = np.array(y_train_2).astype(int)

    # build logistic model
    modelLog2 = LogisticRegression().fit(
        np.array(X_train_2).reshape(-1, 1), y_train_2)
    # make prediction

    # Make predictions on the test set
    y_pred_proba = modelLog2.predict_proba(
        np.array(X_test_2).reshape(-1, 1))[:, 1]  # Probability of the positive class

    # Calculate AUC-ROC score
    auc_roc = roc_auc_score(y_test_2, y_pred_proba)
    data9AUC.loc[Q1A[i], "AUC"] = auc_roc

    x2 = np.linspace(min(X_train_2), max(X_train_2), 52000)
    y2 = x2 * modelLog2.coef_ + modelLog2.intercept_
    sigmoid2 = expit(y2)

    # the ravel function returns a flattened array
    axes[i].plot(x2, sigmoid2.ravel(), color='red', linewidth=3)
    axes[i].scatter(X_train_2, y_train_2, color='black')
    axes[i].hlines(0.5, min(X_train_2), max(X_train_2),
                   colors='gray', linestyles='dotted')
    axes[i].set_xlabel(Q1A[i])
   # axes[i].xlim([min(X_train_2),max(X_train_2)])
    axes[i].set_ylabel('mode')
   # axes[i].yticks(np.array([0,1]))


# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
# %% try to find the relationship between mode and key


# %%Question 10 part 1 using original data  and decidion tree to classify


genre = file["track_genre"]

# convert string label of genre to numerical label
le = preprocessing.LabelEncoder()
genre = le.fit_transform(genre)

X_train, X_test, y_train, y_test = train_test_split(
    file[Q1A], genre, test_size=0.2, random_state=12718276)

classifier10_1 = DecisionTreeClassifier(random_state=12718276)

# Create a decision tree classifier
classifier10_1 = DecisionTreeClassifier(random_state=12718276)

# Fit the classifier to the training data
classifier10_1.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier10_1.predict(X_test)

# summary dataframe of evaluation of model
data10Acc = pd.DataFrame(index=["original data", "principal components"], columns=[
                         "cv_scoreDT", "cv_scoreRF"])


# use cross validation to assess model
cv_scores_1 = cross_val_score(classifier10_1, X_train, y_train, cv=5)
mean_cv_score_1 = np.mean(cv_scores_1)
print("this is mean of cross validation of model original data "+str(mean_cv_score_1))



# Evaluate the model
accuracy10_1 = accuracy_score(y_test, y_pred)
data10Acc.loc["original data", "cv_scoreDT"] = mean_cv_score_1
'''
conf_matrix10_1 = confusion_matrix(y_test, y_pred)
data10Acc.loc["original data", "confusion matrix"] = conf_matrix10_1
class_report10_1 = classification_report(y_test, y_pred)
data10Acc.loc["original data", "classification_report"] = class_report10_1
'''
#%% use random forest to classify
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest Classifier
rf = RandomForestClassifier(random_state=12718276)
rf.fit(X_train, y_train)

# Predicting model
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("accuracy of random forest model is "+str(accuracy_rf))

# use cross validation to assess random forest model
cv_rf_score = cross_val_score(rf,X_train,y_train,cv=5)
print("this is mean cross validation score of random forest model: "+
      str(cv_rf_score.mean()))

data10Acc.loc["original data", "cv_scoreRF"] = cv_rf_score.mean()




# %% part 2 use PCA to predict genre

# extract first 3 PC (kaiser method)
X_pca_3 = rotatedData[:,:3]

X_train, X_test, y_train, y_test = train_test_split(X_pca_3, genre, test_size=0.2, random_state=12718276)

#X_train, X_test, y_train, y_test = train_test_split(
#    rotatedData[], genre, test_size=0.2, random_state=12718276)

# Create a decision tree classifier
classifier10_2 = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data with the selected principal components
classifier10_2.fit(X_train, y_train)

# Make predictions on the test set with the selected principal components
y_pred = classifier10_2.predict(X_test)

# Evaluate the model
accuracy10_2 = accuracy_score(y_test, y_pred)

#use cross validation to assess model
cv_scores_p = cross_val_score(classifier10_2, X_train, y_train, cv=5)
mean_cv_score_p = np.mean(cv_scores_p)
print("this is mean of cross validation of PCA data "+str(mean_cv_score_p))

data10Acc.loc["principal components", "cv_scoreDT"] = mean_cv_score_p



#use random forest to classify PCA data
# Train a Random Forest Classifier
rfP = RandomForestClassifier(random_state=12718276)
rfP.fit(X_train, y_train)

# Predicting model
y_pred_rfP = rfP.predict(X_test)
accuracy_rfP = accuracy_score(y_test, y_pred_rfP)

print("accuracy of random forest of PCA model is "+str(accuracy_rfP))

# use cross validation to assess random forest model
cv_rfP_score = cross_val_score(rfP,X_train,y_train,cv=5)
print("this is mean cross validation score of random forest of PCA model: "+
      str(cv_rfP_score.mean()))

data10Acc.loc["principal components", "cv_scoreRF"] = cv_rfP_score.mean()



#data10Acc.loc["principal components", "accuracy"] = accuracy10_2
#conf_matrix10_2 = confusion_matrix(y_test, y_pred)
#data10Acc.loc["principal components", "confusion matrix"] = conf_matrix10_2
#class_report10_2 = classification_report(y_test, y_pred)
#data10Acc.loc["principal components",
#              "classification_report"] = class_report10_2

bars = plt.bar(["original data", "principal components"],
               data10Acc.loc[:, "cv_scoreDT"], color=['blue', 'green'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01,
             round(yval, 2), ha='center', va='bottom')
plt.ylim(0, 0.40)
# Add labels and title
plt.xlabel('Groups')
plt.ylabel('cv score of decision tree model')
plt.title('Decision Tree CV Score Comparison Between Two Groups')

# Show the plot
plt.show()


bars = plt.bar(["original data", "principal components"],
               data10Acc.loc[:, "cv_scoreRF"], color=['blue', 'green'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01,
             round(yval, 2), ha='center', va='bottom')
plt.ylim(0, 0.40)
# Add labels and title
plt.xlabel('Groups')
plt.ylabel('cv score of random tree model')
plt.title('DRandom Forest CV Score Comparison Between Two Groups')

# Show the plot
plt.show()


#%% extra credit code
data4W =pd.DataFrame(columns=["major","popularity"])
data4B = pd.DataFrame(columns=["minor","popularity"])

BValue = 0
WValue = 0
#divide key into minor and major
for i in range(len(file)):
    if file.loc[i,"key"] in [0 , 2 , 4 , 5 , 7 , 9 , 11]:
        data4W.loc[WValue,"major"]=file.loc[i,"key"]
        data4W.loc[WValue,"popularity"]=file.loc[i,"popularity"]
        WValue = WValue+1
    elif file.loc[i,"key"]in [ 1 , 3 , 6 , 8 , 10]:
        data4B.loc[BValue,"minor"]=file.loc[i,"key"]
        data4B.loc[BValue,"popularity"]=file.loc[i,"popularity"]
        BValue = BValue+1
        
plt.hist(data4W["popularity"])
plt.xlabel("popularity")
plt.ylabel("count")
plt.title("distribution of white key popularity")
plt.axvline(data4W["popularity"].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.show()


plt.hist(data4B["popularity"])
plt.xlabel("popularity")
plt.ylabel("count")
plt.title("distribution of black key popularity")
plt.axvline(data4B["popularity"].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.show()


print("mean of white key is "+str(data4W["popularity"].mean()))
print("mean of black key is "+str(data4B["popularity"].mean()))

## perform welch t test

t_test_K,p_val_K = stats.ttest_ind(np.array(data4W["popularity"],dtype=float),
                               np.array(data4B["popularity"],dtype=float),
                               equal_var=False)
print("below is p value of the welch t test")
print(p_val_K)
#perform whitney u test for key
U_test_KE,p_val_KE = stats.mannwhitneyu(np.array(data4W["popularity"],dtype=float),
                               np.array(data4B["popularity"],dtype=float))

print("below is p value of the mann whitney u test")
print(p_val_KE)
#permutation test
resultKE = permutation_test((np.array(data4W["popularity"], dtype=float), np.array(
    data4B["popularity"], dtype=float)), statistic, n_resamples=1000)
print("below is p value of the permutation test")
print(resultKE.pvalue)
#I used the wrong variable, I used key instead of mode here
#p_val_key is 0.07605>0.05, therefore it is possible that minor/major key causes
#popularity differences is due to chance
#for key using permutation test p value is 0.08591> 0.05, therefore we fail to reject null hypothesis

print("median of white key is: "+str(data4W["popularity"].median()))
print("median of black key is: "+str(data4B["popularity"].median()))

