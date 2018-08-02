from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

iris = load_iris()

iris_df = pd.DataFrame(iris.data)
cross_corr = iris_df.corr()

plt.matshow(cross_corr)
plt.yticks([0,1,2,3],iris.feature_names)
plt.title("Cross correlation matrix")
plt.colorbar(ticks=[cross_corr.min().min(), 1])
fig = plt.gcf()
fig.set_size_inches(7,5)

# The least correlated features are sepal_length and sepal_width, which are the first and second column of the data

# Estimator

lin_reg = LinearRegression().fit(iris.data[:,0:2],iris.target)
knn = neighbors.KNeighborsClassifier(n_neighbors=1).fit(iris.data[:,0:2],iris.target)
bnb = GaussianNB().fit(iris.data[:,0:2],iris.target)
dtc = DecisionTreeClassifier().fit(iris.data[:,0:2],iris.target)
pct = Perceptron(max_iter=10,tol=1e-3,eta0=.01).fit(iris.data[:,0:2],iris.target)
svc_rbf = SVC(kernel='rbf', C=1e3, gamma=0.1).fit(iris.data[:,0:2],iris.target)
svc_lin = SVC(kernel='linear', C=1e3).fit(iris.data[:,0:2],iris.target)
svc_poly = SVC(kernel='poly', C=1e3, degree=2).fit(iris.data[:,0:2],iris.target)
# MLP works bad, most likely because the input is not scaled - which makes sense because also the perceprton works real bad
mlp = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(50), random_state=1,activation='logistic').fit(iris.data[:,0:2],iris.target)


xx = np.linspace(iris.data[:,0].min()-.1,iris.data[:,0].max()+.1,100)
yy = np.linspace(iris.data[:,1].min()-.1,iris.data[:,1].max()+.1,100)
XX, YY = np.meshgrid(xx,yy)

fig = plt.figure(figsize=(8.5,8.5))
for ind, model in enumerate([lin_reg,knn,bnb,dtc,pct,svc_rbf,svc_lin,svc_poly,mlp]):
    pred_mesh = model.predict(np.c_[XX.ravel(),YY.ravel()])

    # Visualize decision bundaries
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    ax = fig.add_subplot(3,3,ind+1)
    ax.pcolormesh(XX, YY, pred_mesh.reshape(XX.shape), cmap=cmap_light)
    ax.scatter(iris.data[:,0],iris.data[:,1],c=iris.target)
    if ind in [6,7,8]:
        ax.set_xlabel(iris.feature_names[0])
    if ind in [0,3,6]:
        ax.set_ylabel(iris.feature_names[1])
    ax.set_title("%s" %(model.__class__.__name__),fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
   
#formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
#plt.colorbar(ticks=[0,1,2],format=formatter)


fig1 = plt.figure()
new_data = PCA(n_components=2).fit_transform(iris.data)
ax = fig1.add_subplot(111)
ax.scatter(new_data[:,0],new_data[:,1],c=iris.target)

plt.show()




