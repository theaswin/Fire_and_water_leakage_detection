
import skimage
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import sklearn

flat_data_arr = []
target_arr = []
Categories = ['fire','water']

datadir = '/home/user/Desktop/deep/DataForfireWater'
for i in Categories:
  print("loading.......",i)
  path = os.path.join(datadir,i)
  for img in os.listdir(path):
    img_array = imread(os.path.join(path,img))
    img_resized = resize(img_array,(150,150,3))
    flat_data_arr.append(img_resized.flatten())
    target_arr.append(Categories.index(i))
  print("loaded",i)

df = pd.DataFrame(flat_data_arr)

df['target'] = target_arr

df

X=df.iloc[:,:-1].values
X

y = df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

path = '/home/user/Desktop/deep/test/iStock_000010471290_Small.jpg'
img_array =imread(path)
img_resize = resize(img_array,(150,150,3)).flatten().reshape(1,-1)
y_pred = model.predict(img_resize)

if y_pred == 0:
  print("Fire ")
else:
  print("Water leakage")