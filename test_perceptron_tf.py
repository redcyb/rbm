from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, np

mnist = input_data.read_data_sets('mnist/', one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels

# model = Sequential([
#     Dense(392, input_shape=(784,)),
#     Activation('sigmoid'),
#     Dense(64),
#     Activation('sigmoid'),
#     Dense(10),
#     Activation('sigmoid'),
# ])
model = Sequential([
    Dense(64, input_shape=(784,)),
    Activation('sigmoid'),
    Dense(10),
    Activation('sigmoid'),
])

model.compile(optimizer='rmsprop', loss='mse')

# Train the model, iterating on the data in batches of 32 samples
model.load_weights("./weights/keras___784_64_10___100")
# model.fit(x_train, y_train, epochs=100, batch_size=32)
# model.save_weights("./weights/keras___784_64_10___100")

predict = model.predict_classes(x_test)
classes = np.argmax(y_test, axis=1)

result = np.vstack((classes, predict))
result = np.vstack((result, predict == classes)).T

match = np.count_nonzero(result[:, 2])

print(result)
