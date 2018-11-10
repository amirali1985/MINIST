import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import tensorflow as tf
print("We're using TF", tf.__version__)

import sys
sys.path.append("../..")

import keras_utils
import preprocessed_mnist
X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_mnist.load_dataset()

# X contains rgb values divided by 255
print("X_train [shape %s] sample patch:\n" % (str(X_train.shape)), X_train[1, 15:20, 5:10])
print("A closeup of a sample patch:")
plt.imshow(X_train[1, 15:20, 5:10], cmap="Greys")
plt.show()
print("And the whole sample:")
plt.imshow(X_train[1], cmap="Greys")
plt.show()
print("y_train [shape %s] 10 samples:\n" % (str(y_train.shape)), y_train[:10])

X_train_flat = X_train.reshape((X_train.shape[0], -1))
print(X_train_flat.shape)

X_val_flat = X_val.reshape((X_val.shape[0], -1))
print(X_val_flat.shape)
import keras

y_train_oh = keras.utils.to_categorical(y_train, 10)
y_val_oh = keras.utils.to_categorical(y_val, 10)

print(y_train_oh.shape)
print(y_train_oh[:3], y_train[:3])

s = tf.Session()

W = tf.get_variable("W", shape=[784, 10])
b = tf.get_variable("bias", shape=[1, 10])
print(W.shape[0])

input_X = tf.placeholder('float32', shape=(None, 784))
input_y = tf.placeholder('float32', shape=(None, 10))

hidden1 = tf.layers.dense(input_X, 256, activation=tf.nn.sigmoid)
hidden2 = tf.layers.dense(hidden1, 256, activation=tf.nn.sigmoid)
logits = tf.layers.dense(hidden2, 10, activation=tf.nn.sigmoid)
print(logits.shape)
probas = tf.nn.softmax(logits, 1)
print(probas.shape)
classes = tf.argmax(probas, axis=1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y))
step = tf.train.AdamOptimizer().minimize(loss)


s.run(tf.global_variables_initializer())

BATCH_SIZE = 512
EPOCHS = 40


for epoch in range(EPOCHS):  # we finish an epoch when we've looked at all training samples

    batch_losses = []
    for batch_start in range(0, X_train_flat.shape[0], BATCH_SIZE):  # data is already shuffled
        _, batch_loss = s.run([step, loss], {input_X: X_train_flat[batch_start:batch_start + BATCH_SIZE],
                                             input_y: y_train_oh[batch_start:batch_start + BATCH_SIZE]})

        batch_losses.append(batch_loss)

    train_loss = np.mean(batch_losses)
    val_loss = s.run(loss, {input_X: X_val_flat, input_y: y_val_oh})  # this part is usually small
    train_accuracy = accuracy_score(y_train, s.run(classes, {input_X: X_train_flat}))  # this is slow and usually skipped
    valid_accuracy = accuracy_score(y_val, s.run(classes, {input_X: X_val_flat}))
    print("valid accuracy is " + str(valid_accuracy))
