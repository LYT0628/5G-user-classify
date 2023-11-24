import os
import logging

import numpy as np
from numpy import array
from sklearn.metrics import roc_auc_score


import KNN


base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'train.csv')
N = 10000 

def load_data():
  dataset = []
  labels = []
  count = 0
  try:
    f = open(file_path)
    f.readline()
    line  = ""
    for line in f.readlines():
      props = line.split(',')
      props_number = []
      for prop in props:
        props_number.append(float(prop))

      in_x = props_number[1:-1]
      label = props_number[-1]
      dataset.append(in_x)
      labels.append(int(label))

      print("inX:",  end="\n")
      print(in_x)
      print("label:")
      print(label)
      count +=1
      if(count >= N):
        break
  except Exception as e:
    print(e)
  return dataset, labels

def train_test_split(dataset, labels):
  size  = len(dataset)
  ratio = 0.8
  sep = int((size-1) * ratio)
  # train_x, train_y, test_x, test_y
  return dataset[:sep], labels[:sep], \
         dataset[sep:], labels[sep:]



if __name__ == '__main__':
  dataset, labels = load_data()
  train_x, train_y, test_x, test_y = train_test_split(dataset, labels)

  # using KNN 
  test_res = []
  idx = 0
  for in_x in array(test_x):
    res = KNN.classify(in_x, array(train_x), array(train_y), 4)
    test_res.append(int(res))

    print(idx, ":   ", res, end="\t")
    print(test_y[idx], end="\n")
    idx +=1

  print(roc_auc_score(test_y, test_res))

