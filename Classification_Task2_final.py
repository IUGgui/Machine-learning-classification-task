import keras.optimizers
import numpy as np
import sklearn.metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib as mpl
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from scipy import misc
from PIL import Image
from keras import layers
from keras import backend as K
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN


"""n_neighbors = 3

X_train, y_train = Adasyn.fit_resample(X_train, y_train)
clf = neighbors.KNeighborsClassifier(n_neighbors)
Fit = clf.fit(X_train, y_train)
cross = cross_val_score(Fit, X_train, y_train, scoring="balanced_accuracy", cv=5)
pred = clf.predict(X_test)
print(metrics.balanced_accuracy_score(y_test, pred))
print(np.mean(cross))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(metrics.balanced_accuracy_score(y_test, pred))"""



X = np.load("Xtrain_Classification2.npy")
y = np.load("ytrain_Classification2.npy")
X_final = np.load("Xtest_Classification2.npy")
X_final = np.reshape(X_final, (X_final.shape[0], 5, 5, 3))
X_final = X_final / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#under = TomekLinks(sampling_strategy= 'majority')
under = SMOTETomek()
X_train, y_train = under.fit_resample(X_train,y_train)
#over = SMOTE(random_state=42)
X_train = np.reshape(X_train,(X_train.shape[0], 5, 5 ,3))
X_test = np.reshape(X_test,(X_test.shape[0], 5, 5 ,3))
X_train = X_train / 255.0
X_test = X_test / 255.0
print(X_train[0].shape)

print(X.shape)
uh = []
for i in range(1000):
    if y_train[i] == 0:
        uh.append(i)
print(uh)

zeros = 0
ones = 0
twos = 0
for i in range(y_train.shape[0]):
    if y_train[i] == 0:
        zeros += 1
    if y_train[i] == 1:
        ones += 1
    if y_train[i] == 2:
        twos += 1

print(zeros)
print(ones)
print(twos)


print(X_train[uh[0]])



# Create the base model from the pre-trained model MobileNet V2
plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[uh[i]])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(y[uh[i]])
plt.show()





def recall(y_true, y_pred, c):
    y_true = K.flatten(y_true)
    pred_c = K.cast(K.equal(K.argmax(y_pred, axis=-1), c), K.floatx())
    true_c = K.cast(K.equal(y_true, c), K.floatx())
    true_positives = K.sum(pred_c * true_c)
    possible_postives = K.sum(true_c)
    return true_positives / (possible_postives + K.epsilon())



def recall_c0(y_true, y_pred):
    return recall(y_true, y_pred, 0)

def recall_c1(y_true, y_pred):
    return recall(y_true, y_pred, 1)



def recall_c2(y_true, y_pred):
    return recall(y_true, y_pred, 2)

def average_recall(y_true, y_pred):
    return (recall_c0(y_true, y_pred) + recall_c1(y_true, y_pred) + recall_c2(y_true, y_pred)) / 3


"""def output_cleaner(y_pred):
    result = []
    for line in range(np.shape(y_pred)[0]):
        values = y_pred[line]
        max = 0
        predy = -1
        for index in range(3):
            if values[index] > max:
                max = values[index]
                predy = index
        result.append(predy)

def BAAC(y_true, y_pred):
    return metrics.balanced_accuracy_score(y_true, output_cleaner(y_pred))"""

EPOCHS = 800
BATCH_SIZE = 100




def make_model(metrics = [average_recall]):
    model = tf.keras.Sequential([
                           #Input layer
        tf.keras.Input(shape=(5,5,3), name='input'),
        #data_augmentation,
        tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
        tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
        tf.keras.layers.Conv2D(128, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        layers.Dropout(0.3),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras .layers.Dense(256, activation='relu'),

        tf.keras.layers.Dense(3, activation='softmax')])

    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = [average_recall])

    return model


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=30,
    restore_best_weights=True)


model = make_model()

baseline_history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS, validation_data=(X_test, y_test),  batch_size=BATCH_SIZE )




def plot_metrics(history):
  metrics = ['loss', 'average_recall']
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









pred = model.predict(X_final)
result = []
for line in range(np.shape(pred)[0]):
    values = pred[line]
    max = 0
    predy = -1
    for index in range(3):
        if values[index] > max:
            max = values[index]
            predy = index
    result.append(predy)

"""cm = sklearn.metrics.confusion_matrix(y_test, result)
sns.heatmap(cm, annot=True, fmt="d")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print(metrics.balanced_accuracy_score(y_test, result))
plt.show()"""

np.save('Y_Classification_Task2', result)