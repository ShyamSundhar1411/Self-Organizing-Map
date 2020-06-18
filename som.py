#Importing liraries
import numpy as np
import matplotlib.pyplot as p
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone,pcolor,colorbar,plot,show
#Importing Dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#Feature Scaling
o = MinMaxScaler(feature_range = (0,1))
x = o.fit_transform(x)
#Training Self-Organising Map
som = MiniSom(x = 10,y = 10,input_len = 15,sigma = 1.0, learning_rate = 4.5 )
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)
#Visualizing Results
bone()
pcolor(som.distance_map().T)
colorbar()
mark = ['^', 'p']
color = ['r', 'b']
for i,x in enumerate(x):
    t = som.winner(x)
    plot(t[0]+0.5,t[1]+0.5,mark[y[i]],markeredgecolor = color[y[i]],
         markerfacecolor = 'None',markersize = 10, markeredgewidth = 2)
    
show()
#Finding Frauds
mop = som.win_map(x)
frauds = np.concatenate(mop[(8,1)], mop[(6,8)], axis = 0)
frauds = o.inverse_transform(frauds)
print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))
