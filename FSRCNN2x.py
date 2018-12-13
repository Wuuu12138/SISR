from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, Input, Conv2DTranspose
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, adam
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
import prepare_data as pd
import numpy
import cv2


scale = 2

def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def model_FSRCNN():
    _input = Input(shape=(None, None, 1), name='input')
    FSRCNN = Conv2D(filters=56, kernel_size=(5, 5), padding='same', activation='relu')(_input)
#    FSRCNN = PReLU()(FSRCNN)
    FSRCNN = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(FSRCNN)
#    FSRCNN = PReLU()(FSRCNN)
    FSRCNN = Conv2D(filters=12, kernel_size=(3, 3), padding='same', activation='relu')(FSRCNN)
#    FSRCNN = PReLU()(FSRCNN)
    FSRCNN = Conv2D(filters=12, kernel_size=(3, 3), padding='same', activation='relu')(FSRCNN)
#    FSRCNN = PReLU()(FSRCNN)
    FSRCNN = Conv2D(filters=12, kernel_size=(3, 3), padding='same', activation='relu')(FSRCNN)
#    FSRCNN = PReLU()(FSRCNN)
    FSRCNN = Conv2D(filters=12, kernel_size=(3, 3), padding='same', activation='relu')(FSRCNN)
#    FSRCNN = PReLU()(FSRCNN)
    FSRCNN = Conv2D(filters=56, kernel_size=(1, 1), padding='same', activation='relu')(FSRCNN)
#    FSRCNN = PReLU()(FSRCNN)
    out = Conv2DTranspose(filters=1, kernel_size=(9, 9), strides=(2, 2), padding='same')(FSRCNN)
    model = Model(input=_input, output=out)
    return model


def FSRCNN_train():
    FSRCNN = model_FSRCNN()
    FSRCNN.compile(optimizer=adam(lr=0.0003), loss='mse',metrics=[PSNRLoss])
    print (FSRCNN.summary())

    data, label = pd.read_training_data("drive/SuperResolution/train.h5")

    val_data, val_label = pd.read_training_data("drive/SuperResolution/val.h5")

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=25, 
                                  min_lr=0.000001, verbose=1)
    checkpoint = ModelCheckpoint("drive/SuperResolution/FSRCNNx2_model.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [reduce_lr,checkpoint]
    FSRCNN.fit(data, label, batch_size=64, validation_data=(val_data, val_label),
                               callbacks=callbacks_list, shuffle=True, nb_epoch=200, verbose=1)
#    FSRCNN.save_weights("drive/SuperResolution/FSRCNN_final.h5")


def FSRCNN_predict():
    IMG_NAME = "image5/img.jpg"
    INPUT_NAME = "image5/input.jpg"
    OUTPUT_NAME = "image5/FSRCNN_pre.jpg"

    label = cv2.imread(IMG_NAME)
    shape = label.shape

    img = cv2.resize(label, (int(shape[1] / scale), int(shape[0] / scale)), cv2.INTER_CUBIC)
    cv2.imwrite(INPUT_NAME, img)

    FSRCNN = model_FSRCNN()
    FSRCNN.load_weights("result_code/FSRCNNx2_model.h5")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
    Y[0, :, :, 0] = img[:, :, 0].astype(float) / 255.
    img = cv2.cvtColor(label, cv2.COLOR_BGR2YCrCb)

    pre = FSRCNN.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre = numpy.uint8(pre)
    img[:, :, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.resize(im2, (img.shape[1], img.shape[0]))
    cv2.imwrite("Bicubic.jpg", cv2.cvtColor(im2, cv2.COLOR_YCrCb2BGR))
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)

    print ("Bicubic:")
    print (cv2.PSNR(im1[:, :, 0], im2[:, :, 0]))
    print ("FSRCNN:")
    print (cv2.PSNR(im1[:, :, 0], im3[:, :, 0]))


##*******************************************************************************************************************//
if __name__ == "__main__":
#    FSRCNN_train()
    FSRCNN_predict()
    #vilization_and_show()
