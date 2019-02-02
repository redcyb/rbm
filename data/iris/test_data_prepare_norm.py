import os
import numpy as np

training_set = np.loadtxt(os.path.realpath("iris_TRAINING_Cls_4X_3Y.dat"), usecols=[0, 1, 2, 3, 4, 5, 6])
test_set = np.loadtxt(os.path.realpath("iris_TEST_Cls_4X_3Y.dat"), usecols=[0, 1, 2, 3, 4, 5, 6])

Xss = training_set[:, 0:4]

Xss_training_norm = []

for i in range(Xss.shape[1]):
    x = Xss[:, i]
    xn = (x - x.min()) / (x.max() - x.min())
    Xss_training_norm.append(xn)

Xss_training_norm = np.array(Xss_training_norm).T

# with open("iris_TRAINING_Cls_4X_3Y_NORM.dat", "w") as f:
#     for d in Xss_training_norm:
#         f.write("\t".join(d) + "\n")

np.savetxt("iris_TRAINING_Cls_4X_3Y_NORM.dat", Xss_training_norm)

# with open("iris_TEST_Cls_4X_3Y.dat", "w") as f:
#     for d in test_set:
#         f.write("\t".join(d) + "\n")
