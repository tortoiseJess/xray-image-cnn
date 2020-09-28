"""
Trains a Densenet to recognize multi-class disease in X-ray images
"""

#tensorflow-gpu>=2.1, keras >=2.3.1

from os import listdir
from os.path import join
import glob
from pathlib import Path 
import os 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import os, shutil, time 
from itertools import chain 

import tensorflow as tf 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True 
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)  
import tensorflow.keras.backend as K

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import densenet
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,GlobalAveragePooling2D


#INPUT data which includes the csv file and image directory
#using sample images in the /data
CSV_FILE = '/data/Data_Entry_2017_v2020.csv'
IMG_DIR = '/data/sample_images'
#ouput location for storing the training weights, plots ...
OUTroot = '/'
files_sample = [f for f in listdir(IMG_DIR)]
image_labels_df = image_labels_df[image_labels_df["Image Index"].isin(files_sample)].dropna()

#preprocessing CSV_FILE for keras
labelDf=pd.read_csv(CSV_FILE)
image_labels_df = labelDf[['Image Index', 'Finding Labels']].copy()
image_labels_df['labels']=image_labels_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
all_labels = np.unique(list(chain(*image_labels_df['Finding Labels'].map(lambda x: x.split('|')).tolist()))).tolist()
print(all_labels)
print(image_labels_df.head())

#split test, valid, training set 
train_df, test_df = train_test_split(image_labels_df, test_size = 0.2, random_state = 2020)
print(len(train_df), len(test_df))

## preprocessing data create Generator -------------------------------------------------------------------------------
def get_data_generators(train_df, test_df, batchSize=32):

    ROTATE = 5
    dataGent = ImageDataGenerator(samplewise_center=True,
                                      samplewise_std_normalization=True,
                                     horizontal_flip = True,
                                     vertical_flip = False,
                                     rescale=1./255, rotation_range=ROTATE,
                                     validation_split=0.15)

    testGent = ImageDataGenerator(samplewise_center=True,
                                  samplewise_std_normalization=True,
                                horizontal_flip = True,
                                vertical_flip = False, rotation_range=ROTATE,
                                rescale=1./255)

    train_gen = dataGent.flow_from_dataframe(dataframe=train_df,
                                            directory=IMG_DIR,
                                            x_col = 'Image Index',
                                            y_col = 'labels',
                                            classes = all_labels,
                                            class_mode = 'categorical',
                                            target_size = (224,224),
                                            batch_size =batchSize,
                                            subset='training')

    valid_gen  = dataGent.flow_from_dataframe(dataframe=train_df,
                                            directory=IMG_DIR,
                                            x_col = 'Image Index',
                                            y_col = 'labels',
                                            classes = all_labels,
                                            class_mode = 'categorical',
                                            target_size = (224,224),
                                            batch_size =batchSize,
                                            subset='validation')

    test_gen = testGent.flow_from_dataframe(dataframe=test_df,
                                            directory=IMG_DIR,
                                            x_col = 'Image Index',
                                            y_col = 'labels',
                                            classes = all_labels,
                                            class_mode = 'categorical',
                                            target_size = (224,224),
                                            shuffle=False,
                                            batch_size =batchSize )
    return train_gen, valid_gen, test_gen


## neural network model --------------------------------------------------------------------------------------------


def densenet_fcn(resize=(224,224), pretrained="imagenet", freezeUpto=141, onFcLayers=True):
    """
    pretrained = imagenet / path to the saved weights
    """
    densenett = densenet.DenseNet121(input_shape =(*resize,3), 
                                     include_top = False,
                                    weights = pretrained,
                                    pooling='avg' ) 

    if freezeUpto>0:
        for layer in densenett.layers[:freezeUpto]:
          layer.trainable = False

    densenet_model = Sequential()
    densenet_model.add(densenett)
    if onFcLayers:
      densenet_model.add(Dropout(0.5))
      densenet_model.add(Dense(512))
      densenet_model.add(Dropout(0.5))
    densenet_model.add(Dense(15, activation = 'sigmoid'))

    return densenet_model

##training
def training(train_gen, valid_gen, model, bs, lr, epoch, outPath, arch_name, weights = None ):

  opt = optimizers.Adam(learning_rate=lr)
  model.compile(optimizer = opt,
              loss = 'binary_crossentropy',
              metrics = ['binary_accuracy']) 

  trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
  non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
  print('Number of trainable weights in model: ', trainable_count)
  print('Number of non trainable weights in model: ',non_trainable_count)
  #print(model.summary())

  #settings to save checkpoints and when to stop training.
  trained_model_path = os.path.join(outPath, arch_name, arch_name+'-{epoch:04d}-{val_loss:.2f}.hdf5') 
  print(trained_model_path)
  checkpoint = ModelCheckpoint(trained_model_path, 
                                monitor='val_loss', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='min')

  earlyStop = EarlyStopping(monitor="val_loss", 
                            mode="min", 
                            patience=5,
                            restore_best_weights=True)
  callbacks_list = [checkpoint, earlyStop]

  start_time = time.perf_counter()

  try:
      history = model.fit_generator(train_gen,
        validation_data = valid_gen,
        epochs = epoch,
        shuffle=True,
        class_weight=weights,
        callbacks=callbacks_list)

      elapsed = time.perf_counter() - start_time
      print('Elapsed %.3f seconds for training ' % elapsed)
      print('Trained using the following parameters: arhitecture {0}, batchsize {1}, lr {2}, epochs {3}'.format(arch_name, bs, lr, epoch)) 
      print(history.history)
      plot_history(history, arch_name, outPath)

  except Exception as e:
      print('Training encountered exception {0}'.format(e))

#calculates auc, auprc for test set
def testing(model_path, testGen, OUTroot, arch_name):

    trained_model = load_model(model_path, compile=False)

    pred = trained_model.predict_generator(generator=testGen, verbose=1, steps = len(test_gen))
    print("pred shape: ", pred.shape)
    print(pred[0])

    with open(join(OUTroot, arch_name, "scores.txt"), 'x') as f:
      for idx, c in enumerate(all_labels):
          true_label = test_df['Finding Labels'].str.contains(c).to_numpy()*1 
          aucLine ='{} {} AUC: {} '.format(c, "weighted", roc_auc_score(true_label, pred[:,idx], average='weighted'))
          prcLine = '{} {} AUPRC: {} '.format(c,  "weighted", average_precision_score(true_label, pred[:,idx], average='weighted'))
          print(aucLine)
          print(prcLine)
          f.write(aucLine)
          f.write("\n")
          f.write(prcLine)
          f.write("\n")
        
def plot_history(history, arch_name, OUTroot):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])    
    plt.title('Learning Curve')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(os.path.join(OUTroot,arch_name,'loss_curves.png'))

    plt.clf()
    plt.plot(history.history['binary_accuracy']) 
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(os.path.join(OUTroot, arch_name, 'accuracy_curves.png'))


##main ------------------------------------------------------------------------------------------------------------------------------------
arch_names = [
    "simple_densenet121_fcn",
    "whole_densenet121",
    "weighted_whole_densenet121_fcn",
    "whole_densenet121_fcn",
    ]

ratios = [0.8969051 , 0.97524081, 0.95837496, 0.97945951, 0.88122547,
0.97755976, 0.98496254, 0.99797538, 0.82256511, 0.94843025,
0.46163932, 0.94353371, 0.96980913, 0.98723689, 0.95271138]
class_weights = {k:v for k,v in enumerate(ratios)}

models = [ densenet_fcn( freezeUpto=141, onFcLayers=True),
            densenet_fcn( freezeUpto=0, onFcLayers=False),
          densenet_fcn( freezeUpto=0, onFcLayers=True),
          densenet_fcn( freezeUpto=0, onFcLayers=False),
         ]

i=3
bs = 32
lr = 0.0002
EPOCH = 10
train_gen, valid_gen, test_gen = get_data_generators(train_df, test_df, bs)

try:
    Path(join(OUTroot, arch_names[i])).mkdir(parents=True, exist_ok=True)
    if "weighted" in arch_names[i]:
       weights = ratios
    else:
       weights = None
    training(train_gen, valid_gen, model, bs, lr, EPOCH, OUTroot, arch_names[i], weights)

    jointpath = os.path.join(OUTroot,arch_names[i],'*.hdf5')
    list_of_files = glob.glob(jointpath)
    best_model_path = max(list_of_files, key=os.path.getctime)
    #last saved model is the best model
    testing(best_model_path, test_gen, OUTroot, arch_names[i])
except Exception as e:
    print(e)
    print("error encountered, now exit training ")


