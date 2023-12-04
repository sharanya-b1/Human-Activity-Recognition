from keras import models, layers
from keras.layers import Input, Dense, Softmax, Dropout
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
import keras
from keras.utils import to_categorical
from collections import defaultdict
import pickle
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd
import numpy as np



def solve_logistic(x_test, y_test):
  print("\nSelect C value")
  d = {1:1e-5, 2:1e-4, 3:1e-3, 4:1e-2, 5:1e-1, 6:1, 7:10}
  # 0.736, 0.815, 0.848, 0.859, 0.863, 0.864, 0.864
  # 0.580, 0.655, 0.667, 0.669, 0.669, 0.669, 0.669
  for x in d:
    print(f'{x}) {d[x]}')
  c = int(input())

  filename = f'./models/logistic/logistic{c}_full'
  loaded_model = pickle.load(open(filename, 'rb'))
  y_pred = loaded_model.predict(x_test)
  print(classification_report(y_pred, y_test, digits=3))


# def solve_svm(x_test, y_test):
#   # kernel linear, C = 0.1
#   print("\nThe best hyperparameters for the SVM have been chosen during training")

def solve_nn(x_test, y_test):
  print("\nThe best hyperparameters for the neural network have been chosen during training")
  model = keras.models.load_model('./models/neural network/neural_network_full')
  d = {6: 0, 13: 1, 1: 2, 4: 3, 5: 4, 17: 5, 3: 6, 2: 7, 7: 8, 16: 9, 24: 10, 12: 11}
  y_pred = model.predict(x_test).argmax(axis=-1).tolist()
  d_rev = {d[x]:x for x in d}
  for i in range(len(y_pred)):
    y_pred[i] = d_rev[y_pred[i]]
  print(classification_report(y_pred, y_test, digits=5))


def solve_decision_tree(x_test, y_test):
  print("\nSelect type")
  print("1) Ordinary Decision Tree")
  print("2) Random Forest")
  print("3) Boosted Decision Tree")
  t = int(input())
  if t == 1:
    path = './models/decision trees/decision_tree_full'
  elif t == 2:
    path = './models/decision trees/random_forest_full'
  elif t == 3:
    path = './models/decision trees/boosted_tree_full'
  with open(path, 'rb') as file:
    model = pickle.load(file)

  y_pred = model.predict(x_test)
  print(classification_report(y_pred, y_test, digits=5))




with open('./x_test', 'rb') as file:
    x_test = pickle.load(file)
with open('./y_test', 'rb') as file:
    y_test = pickle.load(file)

while True:
  print("\n\n\nSelect Model")
  print('1) Logistic Regression')
  print('2) Neural Network')
  print('3) Decision Tree')
  m = int(input())
  if m == 1:
    solve_logistic(x_test, y_test)
  # elif m == 2:
  #   solve_svm(x_test, y_test)
  elif m == 2:
    solve_nn(x_test, y_test)
  elif m == 3:
    solve_decision_tree(x_test, y_test)

