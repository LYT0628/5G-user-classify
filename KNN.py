import os 
import operator

import numpy as np
from numpy import tile
from numpy import array 


def classify(in_x, dataset, labels, k):
  # size: the size of dataset

  size = dataset.shape[0]
  # copy x to count the diff 
  diff_mat = tile(in_x, (size, 1)) - dataset
  sq_diff_mat = diff_mat ** 2 
  sq_distances = sq_diff_mat.sum(axis = 1)
  distances = sq_distances ** 0.5
  
  # sort distance ,and index as the value of new array
  sorted_distances = distances.argsort()
  
  # dict contains the votes
  count = {}
  for i in range(k):
    label_index = labels[sorted_distances[i]]
    count[label_index] = count.get(label_index,0) + 1
  # sort by votes
  sorted_count  = sorted(count.items(), 
                         key= operator.itemgetter(1), 
                         reverse=True)

  return sorted_count[0][0]