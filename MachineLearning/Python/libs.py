import numpy as np
from sklearn.model_selection import train_test_split


#1. numpy.arange(n)
# 해당 n의 개수만큼 0부터 정수를 생성
# numpy.arange(3,7) - 3부터 7미만까지 정수를 생성
aray = np.arange(3)


#2. numpy.bincount(x), x는 array_like, 1dimension, nonnegative ints
# 해당 list에 0부터 최대값까지의 개수를 각각 셈

np.bincount(np.arange(5))

# array([1,1,1,1,1]) # 왜? np.arange는 0부터 4까지 생성, 각각 1개씩 있으므로 1

np.bincount([0,1,3])
# array([1,1,0,1]) # 최대값인 3까지 개수를 세지만, 2는 없기에 2의 자리는 0


#3. numpy.reshape(x,y)
# 해당 list를 x행 y열로 쪼갬



#4. sklearn.model_selection.train_test_split(*arrays, **options)
# array나 matrice를 train과 test 셋으로 나눈다
# options
#   train_size  : 0.0~1.0
#   test_size   : 0.0~1.0 / trainsize와 complementary관계
#   random_state: int값이 주어지면 rand의 seed가 된다.
#   shuffle     : boolean, shuffle 여부를 확인 (default : True)
#   stratify    : array-like or None
#                 shuffle이 True일 때, data를 계층을 나누어 쪼갠다

X, y = np.arange(20).reshape((10,2)), range(10)

X_train, y_train, X_test, y_test = train_test_split(X,y, test_size=0.25)
X_train2, y_train2, X_test2, y_test2 = train_test_split(X,y,test_size=0.40)
X_train3, y_train3, X_test3, y_test3 = train_test_split(X,y,test_size=0.60)
X_train4, y_train4, X_test4, y_test4 = train_test_split(X,y,test_size=0.80)

print(X,y)
print (X_test)
print (X_test2)
print (X_test3)
print (X_test4)
