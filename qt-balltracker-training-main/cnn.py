import random
import numpy as np
import tensorflow as tf

from keras import models,layers,Model
from keras import losses
from tqdm import tqdm
import glob
import json
import cv2
import keras
from unet_model import build_unet_model
from deepball import deepball
from sklearn.linear_model import RANSACRegressor,LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences


records_dir = '/Users/muck27/Downloads/qt-stuff/records/'
records = glob.glob(records_dir+'*.json')
size = 496
sample_x = np.zeros((size,size,2),np.uint8)
sample_y = np.zeros((size,size,1),np.uint8)
check_files = []
inputs,targets = [],[]
x_coords = []
x_radii = []

for rec in records:
    with open(rec) as f:
        data = json.load(f)

    coords = []
    radius = []
    vals = []
    all_recs = []
    for key in data.keys():
        x,y,r = data[key]['key']
        val = data[key]['val']
        coords.append((x,y))
        radius.append(r)
        vals.append(val)
        all_recs.append((x,y,r))
    
    inputs.append(all_recs)
    targets.append(vals)
    
    # x_coords.append(coords)
    # x_radii.append(radius)
    # sample_x_copy = sample_x.copy()
    # sample_y_copy = sample_y.copy()
    # for i,pt in enumerate(coords):
    #     if pt[0] < 496 and pt[1] < 496:

    #         sample_x_copy[pt[1]][pt[0]][0] = 255
    #         sample_x_copy[pt[1]][pt[0]][1] = radius[i]
    #         sample_y_copy[pt[1]][pt[0]][0] = 1

    # inputs.append(sample_x_copy)
    # targets.append(sample_y_copy)

total_inputs = len(inputs)
len_seq_inputs = [len(pt) for pt in inputs]

# print(total_inputs)
# print(len_seq_inputs)

mean_of_sequence = np.sum(np.array(len_seq_inputs))/total_inputs
print(mean_of_sequence)
max_of_sequence = np.max(len_seq_inputs)
print(max_of_sequence)

padded_input = pad_sequences(inputs,maxlen=max_of_sequence)
padded_targets = pad_sequences(targets,maxlen=max_of_sequence)
# print(padded_input)
# print(padded_input.shape)
# print("xxxxxxxxxxx")
# print(padded_targets)
# print(padded_targets.shape)
inputs=padded_input
targets= padded_targets


def rotated_parabola(x, a, b, c):
    return a * x**2 + b * x + c

# '''
# Debug
# '''
# for i,input in enumerate(inputs):

#     check = input[...,0]
#     merged = cv2.merge([check,check,check])

#     x = np.array([x[0] for x in x_coords[i]])
#     y = np.array([x[1] for x in x_coords[i]])

#     ransac = RANSACRegressor(make_pipeline(PolynomialFeatures(2), LinearRegression()), random_state=0,min_samples=7,max_trials=100,residual_threshold=80)
#     ransac.fit(x.reshape(-1, 1), y)
#     inliers = ransac.inlier_mask_
#     outliers = np.logical_not(inliers)
#     a, b, c = ransac.estimator_.steps[-1][1].coef_

#     # Generate points for the fitted rotated parabola
#     x_fit = np.linspace(min(x), max(x), 100)
#     y_fit = rotated_parabola(x_fit, a, b, c)

#     good_x = x[inliers]
#     good_y = y[inliers]
#     for i in range(len(good_x)):
#         x_ , y_ = good_x[i],good_y[i]
#         cv2.circle(merged,(int(x_),int(y_)),2,(255,0,0),2)

#     cv2.imshow("window",merged)
#     cv2.waitKey(0)

test_samples  = int(0.2*len(inputs))
train_samples = len(inputs) - test_samples
train_inputs,train_targets, test_inputs,test_targets = tf.convert_to_tensor(inputs[:train_samples]),tf.convert_to_tensor(targets[:train_samples]),tf.convert_to_tensor(inputs[train_samples:]),tf.convert_to_tensor(targets[train_samples:])

print("Train Input size :: " + str(train_inputs.shape) + ' Train Target size :: ' + str(train_targets.shape))
print("Test Input size :: " + str(test_inputs.shape) + ' Test Target size :: ' + str(test_targets.shape))

# model = deepball()
# print(model.summary())
# model = build_unet_model()
# print(model.summary())
# encoder = models.Sequential()
# encoder.add(layers.Conv2D(32, 3, strides=1, padding='same', input_shape=(496,496,2)))
# encoder.add(layers.BatchNormalization())
# encoder.add(layers.Activation('relu'))
# encoder.add(layers.Dropout(0.2))
# encoder.add(layers.MaxPooling2D(2, strides=2))  


# encoder.add(layers.Conv2D(64, 3, strides=1, padding='same'))
# encoder.add(layers.BatchNormalization())
# encoder.add(layers.Activation('relu'))
# encoder.add(layers.Dropout(0.2))
# encoder.add(layers.MaxPooling2D(2, strides=2))



# # encoder.add(layers.Conv2D(128, 3, strides=1, padding='same'))
# # encoder.add(layers.BatchNormalization())
# # encoder.add(layers.Activation('relu'))
# # encoder.add(layers.Dropout(0.2))
# # encoder.add(layers.MaxPooling2D(2, strides=2))
# encoder.summary()


# decoder = models.Sequential()
# decoder.add(layers.Conv2D(32, 3, strides=1, padding='same', input_shape=encoder.output.shape[1:]))
# decoder.add(layers.BatchNormalization())
# decoder.add(layers.Activation('relu'))
# decoder.add(layers.Dropout(0.2))
# decoder.add(layers.UpSampling2D(2))

# decoder.add(layers.Conv2D(16, 3, strides=1, padding='same'))
# decoder.add(layers.BatchNormalization())
# decoder.add(layers.Activation('relu'))
# decoder.add(layers.UpSampling2D(2))

# # decoder.add(layers.Dense(500, activation='softmax'))

# # decoder.add(layers.Conv2D(1, 3, strides=1, padding='same'))
# # decoder.add(layers.BatchNormalization())
# # decoder.add(layers.Activation('relu'))
# # decoder.add(layers.UpSampling2D(2))
# decoder.summary()


# def pixelwise_crossentropy_loss(y_true, y_pred, epsilon=1e-7):
#     y_true = tf.cast(y_true, tf.float32)  
#     y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
#     loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1))
#     return loss

# conv_autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.outputs))
# conv_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
#                          , loss=pixelwise_crossentropy_loss,metrics = ["accuracy"])

# # print(conv_autoencoder.summary())

# history = conv_autoencoder.fit(train_inputs, train_targets, batch_size=4, epochs=20,validation_data=(test_inputs,test_targets))



model = keras.Sequential()
# model.add(layers.Input(shape=(496,496,2)))
# model.add(layers.Reshape((496 * 496, 2), input_shape=(496, 496, 2)))
model.add(layers.LSTM(64, input_shape=(max_of_sequence, 3), return_sequences=True))
model.add(layers.Dense(1, activation='sigmoid'))
model.add(layers.Reshape((max_of_sequence, 1), input_shape=(max_of_sequence*max_of_sequence, 1)))



# Compile the model


# # Print the model summary
model = tf.keras.models.load_model('20- trained.h5')
model.summary()
# model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adagrad(learning_rate=0.001),metrics='accuracy')
# history = model.fit(train_inputs, train_targets, batch_size=8, epochs=2000,validation_data=(test_inputs,test_targets))


# model.save('20- trained.h5')
outs = model.predict(train_inputs)
# print(outs.shape)
# print(test_inputs.shape)


matt = np.zeros((446,446,3),np.uint8)
for i,pred in enumerate(outs):
    # mat = test_inputs[i].numpy()[...,0]
    # matt = cv2.merge((mat,mat,mat)).astype(np.uint8)
    mat = matt.copy()
    reshaped_pred = pred[:,0]
    # print(reshaped_pred.shape)
    # print(pred.shape)
    # print(reshaped_pred.shape)
    for j,pix_pred in enumerate(reshaped_pred):
        if pix_pred > 0.4:
            
            x,y,r = train_inputs[i][j].numpy()
            # print(x,y,r)
            cv2.circle(mat,(x,y),4,(255,0,0),-1)

    all_pts  = train_inputs[i]
    for pt in all_pts:
        x,y,r = pt
        cv2.circle(mat,(int(x),int(y)),2,(255,255,255),-1)
    # print(i,pred.reshape(1,496))
    # print(out.shape)   
    # out_ = out[...,0]
    # mat = test_inputs[i].numpy()[...,0]
    # print(mat.shape)
    # matt = cv2.merge((mat,mat,mat)).astype(np.uint8)

    # for y in range(496):
    #     for x in range(496):
    #         if (out_[y][x] >0.9):
    #             # print(y,x,out_[y][x])
    #             cv2.circle(matt,(x,y),1,(255,0,0),-1)
    #         else:
    #             cv2.circle(matt,(x,y),1,(255,255,255),-1)


    cv2.imshow('hello',mat)
    cv2.waitKey(0)

# for i,out in enumerate(outs):
#     # print(out.shape)
#     out_ = out[...,0]
#     mat = test_inputs[i].numpy()[...,0]
#     print(mat.shape)
#     matt = cv2.merge((mat,mat,mat)).astype(np.uint8)

#     for y in range(496):
#         for x in range(496):
#             if (out_[y][x] >0.9):
#                 # print(y,x,out_[y][x])
#                 cv2.circle(matt,(x,y),1,(255,0,0),-1)
#             else:
#                 cv2.circle(matt,(x,y),1,(255,255,255),-1)


#     cv2.imshow('hello',matt)
#     cv2.waitKey(0)
# print("xxxxxxx")
# print(test_targets.T)








 
# for j in tqdm(range(100)):
#     all_pts = []
#     all_radii = []
#     for i in range(30):
#         all_pts.append([random.randint(0,size-1),random.randint(0,size-1)])
#         all_radii.append(random.randint(0,10))

#     sample_x_copy = sample_x.copy()
#     sample_y_copy = sample_y.copy()
#     for i,pt in enumerate(all_pts):
#         sample_x_copy[pt[1]][pt[0]][0] = 1
#         sample_x_copy[pt[1]][pt[0]][1] = all_radii[i]
#         sample_y_copy[pt[1]][pt[0]][0] = 2*all_radii[i]*all_radii[i] + 5


#     # pos_indexes = np.where(np.array(all_radii) > 5)[0]
#     # pos_radii = [r for i,r in enumerate(all_pts) if i in pos_indexes ]
#     # for pt in pos_radii:
#     #     sample_y_copy[pt[1]][pt[0]][0] = 1
 

#     x_train.append(sample_x_copy)
#     y_train.append(sample_y_copy)


# for j in tqdm(range(50)):
#     all_pts = []
#     all_radii = []
#     for i in range(30):
#         all_pts.append([random.randint(0,size-1),random.randint(0,size-1)])
#         all_radii.append(random.randint(0,10))

#     sample_x_copy = sample_x.copy()
#     sample_y_copy = sample_y.copy()
#     for i,pt in enumerate(all_pts):
#         sample_x_copy[pt[1]][pt[0]][0] = 1
#         sample_x_copy[pt[1]][pt[0]][1] = all_radii[i]
#         sample_y_copy[pt[1]][pt[0]][0] = 2*all_radii[i]*all_radii[i] + 5


#     # pos_indexes = np.where(np.array(all_radii) > 5)[0]
#     # pos_radii = [r for i,r in enumerate(all_pts) if i in pos_indexes ]
#     # for pt in pos_radii:
#     #     sample_y_copy[pt[1]][pt[0]][0] = 1
 

#     x_test.append(sample_x_copy)
#     y_test.append(sample_y_copy)


# x_train,y_train,x_test,y_test = tf.convert_to_tensor(x_train),tf.convert_to_tensor(y_train),tf.convert_to_tensor(x_test),tf.convert_to_tensor(y_test)
# # x_train = tf.convert_to_tensor(x_train)
# print("input shape :: " + str(x_train.shape))
# print("target shape :: "+ str(y_train.shape))
