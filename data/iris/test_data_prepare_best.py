import pandas
import numpy

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

filename = "bezdekIris.data"
names = ["sepal length", "sepal width", "petal length", "petal width", "clss"]
dataset = pandas.read_csv(filename, names=names)

xs = dataset.iloc[:, :-1].values
y_pre = dataset.iloc[:, -1].values

label_encoder = LabelEncoder()
y_labels = label_encoder.fit_transform(y_pre)

one_hot_encoder = OneHotEncoder()
ys = one_hot_encoder.fit_transform(y_labels.reshape(1, -1).T).toarray()

result = numpy.hstack((xs, ys))
numpy.savetxt(fname="iris_prepared_one_hot.csv", X=result, fmt="%g", delimiter=",")
