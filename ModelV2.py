
from keras import layers
from keras import models
from keras import backend as K

def myattantion(x,y,z,part):
    # =layers.

    all = layers.Concatenate(axis=3,name='mya_concat1_%d' % part)([x, y])
    all = layers.Concatenate(axis=3,name='mya_concat2_%d' % part)([all, z])

    all_w= layers.GlobalAvgPool2D(name='mya_GAP_%d' % part)(all)
    denseshape=K.int_shape(all_w)[1]
    name = 'mya_fc1_%d' % part
    all_w = layers.Dense(denseshape,name=name)(all_w)
    all_w = layers.ReLU()(all_w)
    name = 'mya_fc2_%d' % part
    all_w = layers.Dense(denseshape,name=name)(all_w)
    name = 'mya_w_out_%d' % part
    all_w= layers.Activation('sigmoid',name=name)(all_w)

    shape_skip = K.int_shape(x)
    # skip__ = layers.Conv2D(1, (1, 1), padding='same')(skip)
    # # concat = layers.Concatenate(axis=concat_axis, name=concat_name)([x__, skip__])
    # skip__ = layers.Activation('sigmoid')(skip__)
    # all_w_repeat = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
    #                           arguments={'repnum': shape_skip[3]})(all_w)

    name = 'mya_w_resh_%d' % part
    all_w_ = layers.Reshape((1, 1, denseshape),name=name)(all_w)

    name = 'mya_w_rep1_%d' % part
    all_w_repeat = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=1),
                                 arguments={'repnum': shape_skip[1]},name=name)(all_w_)

    name = 'mya_w_rep2_%d' % part
    all_w_repeat = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=2),
                                 arguments={'repnum': shape_skip[2]},name=name)(all_w_repeat)
    name = 'mya_w_mul_%d' % part
    all = layers.multiply([all, all_w_repeat],name=name)


    return all


def _conv_block(inputs, filters,kernel=(3, 3), strides=(1, 1)):

    channel_axis = -1
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.ReLU(6., name='conv1_relu')(x)

def depthwise_my(inputs, pointwise_conv_filters, block_id=111,tur='N3d',use_batchnorm=False):

    x = inputs
    strides = (1, 1)
    channel_axis=3
    if tur=='N' or tur=='N3d':
        x_1 = layers.DepthwiseConv2D((3, 3),
                                   padding='same',
                                   strides=strides,
                                   use_bias=False,
                                   name='conv_dw_%d' % block_id)(x)
        if use_batchnorm:
            x_1 = layers.BatchNormalization(
                axis=3, name='conv_dw_%d_bn' % block_id)(x_1)
            x_1 = layers.ReLU(6.,name='conv_dw_%d_relu' % block_id)(x_1)

        x_1 = layers.Conv2D(pointwise_conv_filters, (1, 1),
                          padding='same',
                          use_bias=False,
                          strides=(1, 1),
                          name='conv_pw_%d' % block_id)(x_1)
        if use_batchnorm:
            x_1 = layers.BatchNormalization(axis=channel_axis,
                                          name='conv_pw_%d_bn' % block_id)(x_1)
            x_1 =layers.ReLU(6.,name='conv_pw_%d_relu' % block_id)(x_1)

        if tur=='N':
            return x_1

    if '3do' in tur or tur=='N3d':
        my_shape = x.get_shape().as_list()
        my_shape.append(1)
        x_2 = layers.Reshape((my_shape[1],my_shape[2],my_shape[3],my_shape[4]))(x)

        x_2=layers.Conv3D(pointwise_conv_filters/(my_shape[3]), (3, 3,3),
                          padding='same',
                          use_bias=False,
                          strides=(1, 1,1),
                          name='conv3d_pw_%d' % block_id)(x_2)
        if use_batchnorm:
            x_2= layers.BatchNormalization(name='conv3d_pw_%d_bn' % block_id)(x_2)
            x_2=layers.ReLU(6.,name='conv3d_pw_%d_relu' % block_id)(x_2)
        my_shape = x_2.get_shape().as_list()
        x_2 = layers.Reshape((my_shape[1], my_shape[2], my_shape[3]*my_shape[4]))(x_2)

        if '3do' in tur:
            return x_2
    x = layers.Concatenate(axis=3, name='concat_%d_bn' % block_id)([x_1, x_2])
    return x

    # return x_1


def _depthwise_conv_block(inputs, pointwise_conv_filters, strides=(1, 1), block_id=222):

    channel_axis = 3
    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

def DecoderTransposeX2Block_mya(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 3

    def layer(input_tensor, skip=None,my_skip=None):

        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = myattantion(x, skip,my_skip,stage)
        x= _depthwise_conv_block(x, filters, block_id=stage+50)
        return x

    return layer



def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 3

    def layer(input_tensor, skip=None):

        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])
        x= _depthwise_conv_block(x, filters, block_id=stage+50)

        return x

    return layer


def model(input_shape,classes=2,activation='softmax',use_batchnorm=True,tur='3do_mya'):

    img_input = layers.Input(shape=input_shape)
    if input_shape[2]!=1:
        modelInput = layers.Conv2D(1, (1, 1),
                          padding='same',
                          use_bias=False,
                          strides=(1, 1),
                          name='RGB_Input_ReshapeToGray')(modelInput)
    else:
        modelInput = img_input

    x = _conv_block(modelInput, 32, strides=(2, 2))
    # x = _depthwise_conv_block(img_input, 32, strides=(2, 2))

    skip1 = _depthwise_conv_block(x, 32, block_id=1)

    x = _depthwise_conv_block(skip1, 64, strides=(2, 2), block_id=2)
    skip2 = _depthwise_conv_block(x, 64, block_id=3)

    x = _depthwise_conv_block(skip2, 128,strides=(2, 2), block_id=4)
    skip3 = _depthwise_conv_block(x, 128, block_id=5)

    x = _depthwise_conv_block(skip3, 256,strides=(2, 2), block_id=6)
    skip4 = _depthwise_conv_block(x, 256, block_id=7)

    x = _depthwise_conv_block(skip4, 512, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 512,block_id=13)

    # inputs_1= layers.Input(shape=(img_input.shape[1] // 2**1, img_input.shape[2] // 2**1, (2**1)**2), name='input_1__')
    # inputs_2 = layers.Input(shape=(img_input.shape[1] // 2**2, img_input.shape[2] // 2**2, (2**2)**2), name='input_2')
    # inputs_3 = layers.Input(shape=(img_input.shape[1] // 2**3, img_input.shape[2] // 2**3,  (2**3)**2), name='input_3')
    # inputs_4 = layers.Input(shape=(img_input.shape[1] // 2**4, img_input.shape[2] // 2**4, (2**4)**2), name='input_4')

    inputs_1= layers.Reshape(target_shape=(modelInput.shape[1] // 2**1, modelInput.shape[2] // 2**1, (2**1)**2), name='input_1__') (modelInput)
    inputs_2 = layers.Reshape(target_shape=(modelInput.shape[1] // 2**2, modelInput.shape[2] // 2**2, (2**2)**2), name='input_2')(modelInput)
    inputs_3 = layers.Reshape(target_shape=(modelInput.shape[1] // 2**3, modelInput.shape[2] // 2**3,  (2**3)**2), name='input_3')(modelInput)
    inputs_4 = layers.Reshape(target_shape=(modelInput.shape[1] // 2**4, modelInput.shape[2] // 2**4, (2**4)**2), name='input_4')(modelInput)

    if not tur=='org':
        sub_input_use_batchnorm = False
        inputs_1__ = depthwise_my(inputs_1, 32, block_id=100,tur=tur,use_batchnorm=sub_input_use_batchnorm)
        inputs_2__ = depthwise_my(inputs_2, 64, block_id=101,tur=tur,use_batchnorm=sub_input_use_batchnorm)
        inputs_3__ = depthwise_my(inputs_3, 128, block_id=102,tur=tur,use_batchnorm=sub_input_use_batchnorm)
        inputs_4__ = depthwise_my(inputs_4, 256, block_id=103,tur=tur,use_batchnorm=sub_input_use_batchnorm)
        my_input=[inputs_4__, inputs_3__, inputs_2__, inputs_1__ ]

    skips=[skip4,skip3,skip2,skip1]
    # (256, 128, 64, 32, 16)
    decoder_filters=(256, 128, 64, 32, 16)
    if tur=='org':
        for i in range(5):
            if i < len(skips):
                skip = skips[i]
            else:
                skip = None
            x = DecoderTransposeX2Block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)
    elif tur=='3do_mya':

        for i in range(5):
            if i < len(skips):
                skip = skips[i]
                my_skip = my_input[i]
            else:
                skip = None
            x = DecoderTransposeX2Block_mya(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip,my_skip)



    else:
        for i in range(5):
            if i < len(skips):
                skip = skips[i]
            else:
                skip = None
            concat_name='my_concat_name_%d' % i


            if not skip is  None:
                my_skip = my_input[i]
                skip = layers.Concatenate(axis=3, name=concat_name)([skip, my_skip])
            x = DecoderTransposeX2Block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)



    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    # model = models.Model([img_input,inputs_1,inputs_2,inputs_3,inputs_4], x)
    model = models.Model(img_input, x)

    return model

