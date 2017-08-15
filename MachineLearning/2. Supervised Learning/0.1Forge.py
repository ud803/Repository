

import mglearn
import matplotlib.pyplot as plt

'''
1. Classification Data Set - forge dataset

    # An exmaple of a synthetic two-class classification dataset

    # "Forge Dataset" - two features


    # Creates a scatter plot visualizing all of the data points in the set.
    # First feature on the x-axis, second feature on the y.
    # Color and shape of the dot indicates its classes.

'''

# generate datset
X, y = mglearn.datasets.make_forge()

#plot dataset
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))    #shape = 26 datapoints, 2 fts
plt.show()
