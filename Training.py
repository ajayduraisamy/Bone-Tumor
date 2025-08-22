from sklearn.datasets import load_files       
from keras.utils import to_categorical
import numpy as np
from glob import glob
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint 

tar=2
path='./dataset/'
# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = to_categorical(np.array(data['target']), tar)
    return files, targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset(path)
test_files=train_files
test_targets = train_targets


burn_classes = [item[10:-1] for item in sorted(glob("./dataset/*/"))]
print('There are %d total categories.' % len(burn_classes))
print(burn_classes)
print('There are %s total  images.\n' % len(np.hstack([train_files, test_files])))
print('There are %d training images.' % len(train_files))
print('There are %d test images.'% len(test_files))

for file in train_files: assert('.DS_Store' not in file)



from tensorflow.keras.preprocessing import image                  
from tqdm import tqdm


def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_path, width, height) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

import keras
import timeit

# graph the history of model.fit
def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show() 

# callback to show the total time taken during training and for each epoch
class EpochTimer(keras.callbacks.Callback):
    train_start = 0
    train_end = 0
    epoch_start = 0
    epoch_end = 0
    
    def get_time(self):
        return timeit.default_timer()

    def on_train_begin(self, logs={}):
        self.train_start = self.get_time()
 
    def on_train_end(self, logs={}):
        self.train_end = self.get_time()
        print('Training took {} seconds'.format(self.train_end - self.train_start))
 
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = self.get_time()
 
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_end = self.get_time()
        print('Epoch {} took {} seconds'.format(epoch, self.epoch_end - self.epoch_start))

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint 

import matplotlib.pyplot as plt


img_width, img_height = 224, 224
batch_size = 4
epoch=100

########

img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 40
validation_steps = 40
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 3
lr = 0.0004

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import callbacks
import time

#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(tar, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

hist=model.fit(train_tensors, train_targets ,validation_split=0.3, epochs=epoch, batch_size=10)

show_history_graph(hist)
test_loss, test_acc = model.evaluate(train_tensors, train_targets)

y_pred=model.predict(train_tensors)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
cm = confusion_matrix(np.argmax(train_targets, axis=1),np.argmax(y_pred, axis=1))

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred

from sklearn.metrics import roc_curve
fpr ,tn, thresholds = roc_curve(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)
accuracycnn = accuracy_score(np.argmax(test_targets, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)
print(cm)
print('Accuracy=' +str(accuracycnn*100))

fpr ,tpr, thresholds = roc_curve(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)
Sensitivity= tpr / (tpr+fpr)

print('Sensitivity='+str(Sensitivity[1]))
precisioncnn = precision_score(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)

print('precision='+str(precisioncnn))

fpr ,tpr, thresholds = roc_curve(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)
f1scorecnn = f1_score(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)

print('f1-score='+str(f1scorecnn))

fpr ,tpr, thresholds = roc_curve(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)
recallscorecnn = recall_score(np.argmax(test_targets,axis=1)>=1,np.argmax(y_pred,axis=1)>=1)

print('recall-score='+str(recallscorecnn))

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(np.argmax(test_targets, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)
# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')
# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')
# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')
# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('CNN True Positive Rate')
plt.xlabel('CNN False Positive Rate')
plt.show()

model.save('trained_model_CNN.h5')

###############
