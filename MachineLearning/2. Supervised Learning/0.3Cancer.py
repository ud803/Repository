'''
3. Wisconsin Breast Cancer dataset

  # 569 data points, 30 features each !

'''

import mglearn
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.datasets import load_breast_cancer


# Load Datasets
cancer = load_breast_cancer()

# Show Dataset Keys
print("In[4]\n")
print("cancer.keys(): \n{}".format(cancer.keys()))

# Show All Datasets related to each Key
# In[7]도 여기에 포함됨
for _ in cancer :
    print("cancer.{}: \n{}".format(_,cancer[_]))
    print("\n\n\n")

# Shape
print("In[5]")
print("Shape of cancer data: {}".format(cancer.data.shape))

# Sample counts per class
print("\n\nIn[6]")
print("Sample counts per class:\n{}".format({n:v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
# 위 함수에 대한 설명은 python 문서로
