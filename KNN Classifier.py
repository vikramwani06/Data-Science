
# Statistics and machine Learning

# <li> sklearn.neighbors provides functionality for unsupervised and supervised neighbors-based learning methods. 
# <li> Unsupervised nearest neighbors is the foundation of many other learning methods, notably manifold learning and spectral clustering. 
# <li>Supervised neighbors-based learning comes in two flavors: classification for data with discrete labels, and regression for data with continuous labels.

# In[1]:


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# <li> Using the above block of code, we import all the libraries required for carrying out various operations on the dataset.

# In[5]:


# Importing the dataset
dataset = pd.read_csv('C:/Users/viki4/Desktop/Data Science/Oil_PricesFinal.csv')


# <li>The above code reads the "Oil_PricesFinal" csv from the directory.

# In[6]:


X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:,5].values


# <li> Using the above block of code, we split the dataset into dependent and indepedent features.
# <li> Here, variable X stores all the independent variables 
# <li> Variable y stores the dependent variables
# 

# In[8]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# <li> The Scikitlearn Library is simple and efficient tool for data mining and data analysis
# <li> This library is built on top of Numpy, Scipy and Matplotlib
# <li> The cross_validation library in the scikit learn module has a class named train_test_split which implicitly splits the data into training and test sets
# <li> Here, we apply the train_test_split function on the variables X_test, X_train, y_test and y_train variables, which takes as parameters the dependent variables, the independent variables, the test size is set to 0.2 which means 20% of the data falls under the test set and remaining 80% falls under training set.

# In[9]:


from sklearn.neighbors import KNeighborsClassifier

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)


# <li>The sklearn.neighbors module is used to implement the k-nearest neighbors algorithm.
# <li> 

# In[10]:


# fitting the model
knn.fit(X_train,y_train)


# In[11]:


# predict the response
pred = knn.predict(X_test)# evaluate accuracy


# In[12]:


y_test


# In[13]:


from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, pred))print("Test Set Score : {:.2f}".format(knn.score(X_test,y_test)))


# In[28]:


from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, pred))


# In[33]:


k_range=range(0,10)
scores=[]

for K in k_range:
    k_value=K+1
    neigh=neighbors.KNeighborsClassifier(n_neighbors = k_value, weights='uniform', algorithm='auto')
    neigh.fit(X-train,y_train)
    y_pred=neigh.predict(X_test)
    print("Accuracy is", metrics.accuracy_score(y_test,y_pred)*100,"% for K-Value :", k_value)
    scores.append(metrics.accuracy_score(y_test,y_pred))


# In[23]:


# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print (accuracy_score(y_test, pred))


# In[25]:


# creating odd list of K for KNN
from sklearn.cross_validation import cross_val_score
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())


# In[26]:


# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print ("Best value of k:",optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


# In[29]:


def train(X_train, y_train):
	# do nothing 
	return


# In[30]:


def predict(X_train, y_train, x_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# first we compute the euclidean distance
		distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
		# add it to list of distances
		distances.append([distance, i])

	# sort the list
	distances = sorted(distances)

	# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

	# return most common target
	return Counter(targets).most_common(1)[0][0]


# In[31]:


def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	# train on the input data
	train(X_train, y_train)

	# loop over all observations
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))


# In[32]:


# making our predictions 
from collections import Counter
predictions = []

kNearestNeighbor(X_train, y_train, X_test, predictions, 3)

# transform the list into an array
predictions = np.asarray(predictions)

# evaluating accuracy
accuracy = accuracy_score(y_test, predictions)
print(accuracy*100)

