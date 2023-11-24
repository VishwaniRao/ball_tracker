
from keras.layers import Conv2D,BatchNormalization,ReLU,MaxPool2D,Input,UpSampling2D,Concatenate,Softmax
import keras


def _conv_block(inp, convs, pool=True):
    x = inp
    count = 0
    for conv in convs:
        count += 1
        x = Conv2D(conv['filter'],
                   conv['kernel'],
                   strides=conv['stride'],
                   padding='same',
                   use_bias=False)(x)
        x = BatchNormalization(epsilon=0.001)(x)
        x = ReLU()(x)
    if pool:
        x = MaxPool2D(pool_size=2)(x)
    return x

def deepball():
    #Input image
    input_image = Input(shape=(496, 496, 2))

    # Conv1
    x = _conv_block(input_image, [{'filter': 8, 'kernel': 7, 'stride': 2, 'layer_idx': 0},
                                  {'filter': 8, 'kernel': 3, 'stride': 1, 'layer_idx': 1}])
    conv1 = x
    # Conv2
    x = _conv_block(x, [{'filter': 16, 'kernel': 3, 'stride': 1, 'layer_idx': 2},
                        {'filter': 16, 'kernel': 3, 'stride': 1, 'layer_idx': 3}])
    conv2 = x

    # Conv3
    x = _conv_block(x, [{'filter': 32, 'kernel': 3, 'stride': 1, 'layer_idx': 4},
                        {'filter': 32, 'kernel': 3, 'stride': 1, 'layer_idx': 5}])
    conv3 = x

    # Upsampling Conv2
    upsampled_conv2 = UpSampling2D(size=(2,2))(conv2)

    # Upsampling Conv3
    upsampled_conv3 = UpSampling2D(size=(4,4))(conv3)

    # Concatenation along channels axis
    concat = Concatenate(axis=-1)([conv1, upsampled_conv2, upsampled_conv3])
    x = concat

    # Conv4
    x = _conv_block(x, [{'filter': 56, 'kernel': 3, 'stride': 1, 'layer_idx':6},
                        {'filter': 2, 'kernel': 3, 'stride': 1, 'layer_idx': 7}], pool=False)
    
#     conv3c = Reshape((32*30*17,))(conv3)
#     cout = Dense(1, activation='sigmoid')(Dense(200, activation='relu')(conv3c))
    
    x = Softmax(axis=-1)(x)
    model = keras.Model(inputs=input_image, outputs=x)
    
    return model



# encoder = models.Sequential()
# encoder.add(layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', input_shape=(496,496,2)))
# encoder.add(layers.Dropout(0.2))
# encoder.add(layers.MaxPooling2D(2, strides=2))
# encoder.add(layers.Conv2D(64, 3, strides=1, padding='same', activation='relu'))
# encoder.add(layers.MaxPooling2D(2, strides=2))
# encoder.add(layers.Conv2D(128, 3, strides=1, padding='same', activation='relu'))
# encoder.add(layers.MaxPooling2D(2, strides=2))
# # encoder.summary()


# decoder = models.Sequential()
# decoder.add(layers.Conv2D(32, 3, strides=1, padding='same', activation='relu', input_shape=encoder.output.shape[1:]))
# decoder.add(layers.Dropout(0.2))
# decoder.add(layers.UpSampling2D(2))
# decoder.add(layers.Conv2D(16, 3, strides=1, padding='same', activation='relu'))
# decoder.add(layers.UpSampling2D(2))
# decoder.add(layers.Conv2D(1, 3, strides=1, padding='same', activation='relu'))
# decoder.add(layers.UpSampling2D(2))
# # # decoder.summary()
