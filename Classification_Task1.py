import keras.optimizers
import numpy as np
import sklearn.metrics
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib as mpl
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import PIL.Image as P
import scipy as sci
from keras import backend as K

#Our option to solve this Task was CNN.
#We tried several methods to deal with imbalance and improve our F1_score such as:
  #1.Undersampling
  #2.Oversampling using several algortihims as SMOTE
  #3.Data augmentation
  #4.Weighted loss
  #5.Mixing the methods Above
#We ended up with just Data Augmentation because it gave us the best result

"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
ada = SMOTE(random_state=42)
X_train, y_train = ada.fit_resample(X_train, y_train)
X_train = X_train.reshape(X_train.shape[0], 30, 30 ,3)
X_test = X_test.reshape(X_test.shape[0], 30, 30 ,3)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def pos_and_neg(v):
    ones = 0
    zeros = 0
    for i in v:
        if i > 0.5:
            ones +=1
        else:
            zeros += 1
    return [zeros, ones]
print(pos_and_neg(y_train))
plt.subplot(5,5,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(X_train[0])
plt.show()
print(np.size(X_train[0], axis = 0))

EPOCHS = 100
BATCH_SIZE = 50






def make_model(metrics = f1_m):
    model = tf.keras.Sequential([
                           #Input layer
        tf.keras.Input(shape=(30,30,3), name='input'),
        layers.Rescaling(1.0 / 255.0),
        tf.keras.layers.Rescaling(1.0/255.0),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.40),

        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(
        optimizer= tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics = f1_m)

    return model

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=20,
    restore_best_weights=True)


Smotemodel = make_model()

Smote_history = Smotemodel.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    validation_data=(X_test, y_test), callbacks=early_stopping, batch_size=BATCH_SIZE )

loss,  f1_m, = Smotemodel.evaluate(X_test, y_test, verbose=0)


def plot_metrics(history):
  metrics = ['loss', 'f1_m']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color='b', label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color='b', linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    else:
      plt.ylim([0,1])

    plt.legend()

plot_metrics(Smote_history)
plt.show()

pred = Smotemodel.predict(X_test)
result = np.where(pred > 0.5, 1, 0)
cm = sklearn.metrics.confusion_matrix(result, y_test)
sns.heatmap(cm, annot=True, fmt="d")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print(metrics.f1_score(y_test,result))

plt.show()


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def pos_and_neg(v):
    ones = 0
    zeros = 0
    for i in v:
        if i > 0.5:
            ones +=1
        else:
            zeros += 1
    return [zeros, ones]
pos = pos_and_neg(y_train)[1]
neg = pos_and_neg(y_train)[0]
apended = 0
to_delete = []
i = 0
while (apended < neg-pos) and i < pos+neg:
        if y_train[i] < 0.5:
            to_delete.append(i)
            apended +=1
        i+=1
X_train = np.delete(X_train, to_delete, axis=0)
y_train = np.delete(y_train, to_delete, axis=0)


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(30,
                                  30,
                                  3)),
    layers.Rescaling(1.0/255.0),
    layers.RandomRotation(0.4),
    layers.RandomFlip(),
  ]
)






EPOCHS = 100
BATCH_SIZE = 200



def make_model(metrics = [f1_m]):
    model = tf.keras.Sequential([
                           #Input layer
        tf.keras.Input(shape=(30,30,3), name='input'),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.4),
        # # The fifth convolution
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras .layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics = [f1_m])

    return model

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=10,
    restore_best_weights=True)


Undermodel = make_model()

#Under_history = Undermodel.fit(
#    X_train,
#    y_train,
#    epochs=EPOCHS,
#    #batch_size=BATCH_SIZE,
#    callbacks=early_stopping,
#    validation_data=(X_test, y_test))
Under_history = Undermodel.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    validation_data=(X_test, y_test), callbacks=early_stopping, batch_size=BATCH_SIZE )

loss,  f1_m, = Undermodel.evaluate(X_test, y_test, verbose=0)

def plot_metrics(history):
  metrics = ['loss', 'f1_m']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color='b', label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color='b', linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    else:
      plt.ylim([0,1])

    plt.legend()

plot_metrics(Under_history)
plt.show()

pred = Undermodel.predict(X_test)
result = np.where(pred > 0.5, 1, 0)
cm = sklearn.metrics.confusion_matrix(result, y_test)
sns.heatmap(cm, annot=True, fmt="d")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print(metrics.f1_score(y_test,result))

plt.show()


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def pos_and_neg(v):
    ones = 0
    zeros = 0
    for i in v:
        if i > 0.5:
            ones +=1
        else:
            zeros += 1
    return [zeros, ones]
print(pos_and_neg(y_train))


plt.subplot(5,5,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(X_train[0])
plt.show()
print(np.size(X_train[0], axis = 0))

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(30,
                                  30,
                                  3)),
    layers.Rescaling(1.0/255.0),
    layers.RandomRotation(0.4),
    layers.RandomFlip()
  ]
)

EPOCHS = 100
BATCH_SIZE = 200



def make_model(metrics = [f1_m]):
    model = tf.keras.Sequential([
                           #Input layer
        tf.keras.Input(shape=(30,30,3), name='input'),
        data_augmentation,
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # # The fifth convolution
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics = [f1_m])

    return model

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=20,
    restore_best_weights=True)

weight_for_0 = (1 / pos_and_neg(y)[0]) * ((pos_and_neg(y)[0] + pos_and_neg(y)[1]) / 2.0)
weight_for_1 = (1 / pos_and_neg(y)[1]) * ((pos_and_neg(y)[0] + pos_and_neg(y)[1])/ 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

weighted_model = make_model()

weighted_history = weighted_model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=early_stopping,
    validation_data=(X_test, y_test),
    # The class weights go here
    class_weight=class_weight)

def plot_metrics(history):
  metrics = ['loss', 'f1_m']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color='b', label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color='b', linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    else:
      plt.ylim([0,1])

    plt.legend()

plot_metrics(weighted_history)
plt.show()

pred = weighted_model.predict(X_test)
result = np.where(pred > 0.5, 1, 0)
cm = sklearn.metrics.confusion_matrix(result, y_test)
sns.heatmap(cm, annot=True, fmt="d")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print(metrics.f1_score(y_test,result))

plt.show()"""

X = np.load("Xtrain_Classification1.npy")
y = np.load("ytrain_Classification1.npy")
X_final = np.load("Xtest_Classification1.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train = tf.reshape(X_train,(X_train.shape[0], 30, 30 ,3))
X_test = tf.reshape(X_test,(X_test.shape[0], 30, 30 ,3))


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

EPOCHS = 100
BATCH_SIZE = 200

data_augmentation = keras.Sequential(
  [
    layers.Rescaling(1.0/255.0),
    layers.RandomRotation(0.3),
    layers.RandomFlip(),
  ]
)




def make_model(metrics = [f1_m]):
    model = tf.keras.Sequential([
                           #Input layer
        tf.keras.Input(shape=(30,30,3), name='input'),
        data_augmentation,
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        # # The fifth convolution
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras .layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics = [f1_m])

    return model


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=30,
    restore_best_weights=True)
"""resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)
resampled_steps_per_epoch"""

model = make_model()

baseline_history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    validation_data=(X_test, y_test), callbacks=early_stopping, batch_size=BATCH_SIZE )

loss,  f1_m, = model.evaluate(X_test, y_test, verbose=0)

def plot_metrics(history):
  metrics = ['loss', 'f1_m']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color='b', label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color='b', linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    else:
      plt.ylim([0,1])

    plt.legend()

plot_metrics(baseline_history)
plt.show()

pred = model.predict(X_test)
result = np.where(pred > 0.5, 1, 0)
cm = sklearn.metrics.confusion_matrix(result, y_test)
sns.heatmap(cm, annot=True, fmt="d")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print(metrics.f1_score(y_test,result))

plt.show()

