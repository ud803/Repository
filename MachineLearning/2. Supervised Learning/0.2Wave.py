import mglearn
import matplotlib.pyplot as plt

'''
2. Regression Data set - wave dataset
    # single input feature, continuous taret variable
    # single feature on the x, target on the y

'''

# generate dataset
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
print("X.shape: {}".format(X.shape))
plt.show()
