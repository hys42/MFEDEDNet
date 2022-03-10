
import numpy as np




np.random.randint(0 ,25)
image_dim = (192, 192)
def to_catacorial(mask):
    array = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j]==0:
                array[i,j,0]=1
            else:
                array[i,j,1]=1

    return  array
def ReadData():
    images_img=None
    images_mask=None
    # TODO:readDataset
    return images_img, images_mask


def ShowDataExample(goster=False, X_test=None, y_test=None, Limit=100, gosterOran=6):
    import matplotlib.pyplot as plt
    if not goster:
        return 0
    rowControl = 0
    fig_2 = plt.figure(figsize=(21, 21))
    if Limit is None:
        Limit=X_test.shape[0]
    for mask_idx in range(0, Limit, 3):

        if mask_idx == 0:
            rowControl = 0
            fig_2 = plt.figure(figsize=(21, 21))
        # #
        rowControl += 1
        ax = fig_2.add_subplot(gosterOran, gosterOran, rowControl)
        ax.imshow(X_test[mask_idx]/255)
        ax.axis('off')
        ax.title.set_text('mask' + str(mask_idx))
        rowControl += 1
        ax = fig_2.add_subplot(gosterOran, gosterOran, rowControl)
        ax.imshow(y_test[mask_idx,:,:,1])
        ax.title.set_text('y_ ' + str(y_test[mask_idx].sum()))
        ax.axis('off')
        if rowControl >= gosterOran * gosterOran - 1:
            plt.show()
            fig_2 = plt.figure(figsize=(21, 21))
            rowControl = 0

from tensorflow.python.keras.callbacks import ReduceLROnPlateau
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=5,
                                         verbose=1,
                                         epsilon=0.001,
                                         cooldown=0,
                                         min_lr=0.000001)
def converNumpy(list):
    try:
        array = np.zeros((len(list), list[0].shape[0], list[0].shape[1], list[0].shape[2]), dtype=np.float32)
        for i in range(len(list)):
            array[i, :, :, :] = list[i]
    except:
        array = np.zeros((len(list), list[0].shape[0], list[0].shape[1], 1), dtype=np.float32)
        for i in range(len(list)):
            array[i, :, :, 0] = list[i]
    return array



batch_size = 16
DataSet = 'DataSet_Name'
images_img, images_mask =ReadData()


from sklearn.model_selection import KFold
kfold = KFold(n_splits=4,random_state=30, shuffle=True)
fold_no = 0
for train, test in kfold.split(images_img, images_mask):
    fold_no = fold_no + 1
    if  not (fold_no== 1):
        continue
    X_train, X_test, y_train, y_test = images_img[train], images_img[test], images_mask[train], images_mask[test]

    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('X_train   Shape=', X_train.shape)
    print('y_train   Shape=', y_train.shape)

    print('X_test   Shape=', X_test.shape)
    print('y_test   Shape=', y_test.shape)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')

    ShowDataExample(goster=False,  X_test=X_test, y_test=y_test, Limit=100, gosterOran=8)
    X_train = X_train.astype(np.float32) / 255
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32) / 255
    y_test = y_test.astype(np.float32)

    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('X_train   Shape=', X_train.shape)
    print('y_train   Shape=', y_train.shape)

    print('X_test   Shape=', X_test.shape)
    print('y_test   Shape=', y_test.shape)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    # ]

    classes = 2
    activation='softmax'

    import model as model
    import ModelV2 as model

    input_shape=(X_train.shape[1:4])

    model=model.model(input_shape=input_shape, classes=classes)
    model.summary()

    import tensorflow
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy',
                  metrics=['accuracy'])

    autoencoder_train = model.fit(X_train, y_train,
                                  batch_size=batch_size,
                                  epochs=200, verbose=2
                                  # ,callbacks=reduce_learning_rate
                                  )


    TestSonuc = model.evaluate(X_test, y_test,
                               batch_size=batch_size, verbose=0)
    sonuc=[]
    sonuc.append(DataSet)
    pred_test = model.predict(X_test, batch_size=batch_size)

    print(sonuc)
    print(sonuc)






