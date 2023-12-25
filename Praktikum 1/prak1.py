import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

url="https://archive.ics.uci.edu/static/public/545/data.csv"
dataset=pandas.read_csv(url)

print(dataset.shape)
print(dataset.head(20))
print(dataset.groupby('Class').size())
print(dataset.describe())

dataset.plot(kind='box',subplots=True,layout=(2,4),sharex=False,sharey=False)
plt.show()

dataset.hist()
plt.tight_layout()
plt.show()

scatter_matrix(dataset)
plt.show()
