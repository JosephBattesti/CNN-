# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:39:20 2021

@author: JoJo
"""
from skimage.feature import hog
import pickle
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import time
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import LogisticRegression


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])


t0=time.time()
#Open the data
with open('x_train.pkl', 'rb') as f:
    x_train_raw = pickle.load(f)

with open('x_test.pkl', 'rb') as f:
    x_test_raw = pickle.load(f)
    
with open('y_train.pkl', 'rb') as f:
    y_train_label = pickle.load(f)
    
t1=time.time()-t0
print('opened in %.2f s'  % t1)
# %% PROCESSING 
# Create the transformers
hogify = HogTransformer(
    pixels_per_cell=(14, 14), 
    cells_per_block=(2,2), 
    orientations=9, 
    block_norm='L2-Hys'
)

#Scale data
scaler = preprocessing.StandardScaler()



#Perform the transform over train and test
x_train_hog = hogify.fit_transform(x_train_raw)
x_train_scaled = scaler.fit_transform(x_train_hog.reshape(-1, x_train_hog.shape[-1])).reshape(x_train_hog.shape)

x_test_hog = hogify.fit_transform(x_test_raw)
x_test = scaler.fit_transform(x_test_hog.reshape(-1, x_test_hog.shape[-1])).reshape(x_test_hog.shape)

# Encode the classes

le=preprocessing.LabelEncoder()
le.fit(y_train_label)
list(le.classes_)
y_train=le.transform(y_train_label)
Y_train_OH=to_categorical(y_train)



#split
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_scaled, 
    y_train, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
)


t2=time.time()-t1-t0
print('preprocessed in %.2f s'  % t2)
# %% Training

tuned_parameters = [{
    
    "penalty":['l2'],
      "tol":[1e-2,1e-4,1e-6],
      "C":[1e-4,1e-2,100],
      "class_weight":["balanced",None],
      "solver":["newton-cg", "lbfgs","sag"]
}
]

tuned_parameters = [{
    
    "penalty":['l2'],
      "tol":[1e-2,1e-3,5e-3],
      "C":[1e-3,5e-3,1e-2],
      "class_weight":["balanced"],
      "solver":["newton-cg"]
}
]

tuned_parameters = [{
    
    "penalty":['l2'],
      "tol":[1e-2,5e-2,8e-2],
      "C":[1e-3,2e-3,3e-3,4e-3,5e-3],
      "class_weight":["balanced"],
      "solver":["newton-cg"]
}
]
tuned_parameters = [{
    
    "penalty":['l2'],
      "tol":[1e-2],
      "C":[1e-3,1e-2],
      "class_weight":["balanced"],
      "solver":["newton-cg"]
}
]


metric = 'f1_macro'

grid_search = GridSearchCV(
    LogisticRegression(), tuned_parameters, scoring=metric, cv=5, refit=True,verbose=4
)

grid_search.fit(x_train,y_train)

print('The best validation score found is %1.3f' % grid_search.best_score_)
print('The best parameters are ', grid_search.best_params_)
print('The best test score is %1.3f ' % grid_search.score(x_valid,y_valid))


# best_predictor=LogisticRegression(**grid_search.best_params_)

# best_predictor.fit(np.concatenate((x_train,x_valid)),np.concatenate((y_train,y_valid)))
# y_pred = best_predictor.predict(x_test)




t3=time.time()-t2-t0
print('Trained in %.2f s'  % t3)

# %% Kaggle Submisson
import csv

def submit_data(y_pred):
    with open('submission.csv', 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["ID", "class"])
        for i in range(y_pred.shape[0]):
            writer.writerow([i,y_pred[i]])


 
