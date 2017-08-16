'''
k-NN algo is the simplest machine learning algorithm. Building the model consists only of storing the training dataset. To make a prediction for a new data point, the algo finds the closes data points in the training dataset - its "nearest neighbors."
'''

# 1. k-Neighbors Classification
    # 가장 단순한 형태는 가장 가까운 오직 하나의 점만을 고려하는 방법이다.
    # 우리가 예측하고 싶은 데이터와 가장 가까운 트레이닝셋의 라벨이 예측값이 된다.
    # 하나의 데이터가 아닌, k개의 이웃으로부터 가까운 알고리즘을 만들 수도 있다.
    # 하나 이상의 이웃일 때, 우리는 "voting"을 써서 라벨을 결정한다.
    # 즉, 각 클래스 별로 vote를 하여 가장 가까운 값으로 라벨이 결정된다.

# forge 예제 및 필요 라이브러리 불러오기
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt
X, y = mglearn.datasets.make_forge()

# X,y를 Test set과 Training set으로 분리함
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

# kNN 함수 호출하여 인스턴스화, 여기서 이웃(k의 값)은 3
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

# kNN 모델에 training set을 대입
clf.fit(X_train, y_train)

# Test set 예측 및 점수 측정
print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set accuracy: {:.2f}".format(clf.score(X_test,y_test)))


# 2. Analyzing KNeighborsClassifier
    # 2차원 x,y plane에서 각 지점이 어느 클래스에 속하는지를 시각화
    # k가 1, 3, 9 일 때의 모습이다.
    # 그림을 참고했을 때,
    # k의 값이 작아질수록 더 복잡한 모델을 의미하고
    # k의 값이 커질수록 더 간단한 모델을 의미한다 (실제로 경계선이 부드러워진다)
    # 만약 k = n of samples라면 최대 sample이 모든 지점에서의 label이 된다
fig, axes = plt.subplots(1,3,figsize=(10,3))

for n_neighbors, ax in zip([1,3,9], axes) :
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True, eps=0.5,ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{} neighbors(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()

'''
그렇다면 여기서 '모델의 복잡도'와 'generalization' 사이의 관계를 말할 수 있을까? 이 모델은 힘들고 실제 사례를 가져와야 한다.
아래에서부터는 Breast Cancer 예제를 사용
'''

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
