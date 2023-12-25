import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
url="https://archive.ics.uci.edu/static/public/545/data.csv"
df=pd.read_csv(url)
df.head(10)

features=['Area','Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length','Eccentricity','Convex_Area','Extent']
x=df.loc[:,features].values
y=df.loc[:,['Class']].values
x=StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents
              ,columns=['principal component 1','principal component 2'])

finalDf = pd.concat([principalDf,df[['Class']]],axis=1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1',fontsize=15)
ax.set_ylabel('Principal Component 2',fontsize=15)
ax.set_title('2 component PCA',fontsize=20)
targets=['Cammeo','Osmancik']
colors=['r','g']
for target, color in zip(targets,colors):
  indicesToKeep = finalDf['Class']==target
  ax.scatter(finalDf.loc[indicesToKeep,'principal component 1']
             ,finalDf.loc[indicesToKeep,'principal component 2']
             ,c=color
             ,s=50)
  ax.legend(targets)
  ax.grid()

pca.explained_variance_ratio_
