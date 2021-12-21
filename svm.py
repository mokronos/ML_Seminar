import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn import svm

path = './data'
df = gen_dataset(path)
df = df.sort_values(by=['rpm'])
plots = df.shape[0] 

train_x = df["clip"].tolist()
train_y = df["rpm"].values
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(train_x[:-1], train_y[:-1])
prediction = clf.predict(train_x[-1:])
print(f"prediction:{prediction}")
print(f"ground truth: {train_y[-1]}")
