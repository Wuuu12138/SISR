from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, Input, Conv2DTranspose, merge ,UpSampling2D,BatchNormalization,Add,Activation
from keras.optimizers import SGD, adam
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.layers.merge import concatenate, add
import prepare_data as pd
import numpy
import cv2

scale = 2

def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

'''

'''
def _residual_block(ip, kernel):
    init = ip
    x = Conv2D(64, (kernel, kernel), activation='linear', padding='same')(ip)
#    x = BatchNormalization(name="sr_res_batchnorm_" + str(id) + "_1")(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (kernel, kernel), activation='linear', padding='same')(x)
#    x = BatchNormalization(name="sr_res_batchnorm_" + str(id) + "_2")(x)
#    m = Add(name="sr_res_merge_" + str(id))([x, init])
    return x



def _upscale_block(ip):
    init = ip
    x = UpSampling2D()(init)
#    x = UpSampling2D()(x)
    x = Conv2D(64, (3, 3), activation="relu", padding='same')(x)
    return x

'''

'''
def model_InceptionResNet():
    _input = Input(shape=(None, None, 1), name='input')
    x1 = Conv2D(16, (1, 1), activation='relu', padding='same')(_input)
    x2 = Conv2D(16, (3, 3), activation='relu', padding='same')(_input)
    x3 = Conv2D(16, (5, 5), activation='relu', padding='same')(_input)
    x4 = Conv2D(16, (7, 7), activation='relu', padding='same')(_input)
    x0 = concatenate(inputs=[x1,x2, x3,x4])
    
    x = _residual_block(x0,3)
    nb_residual = 5
    for i in range(nb_residual):
        x = _residual_block(x,3)
    
#    nb_residual = 5
#    for i in range(nb_residual):
#        x = _residual_block(x, i + 5)
#    x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = Add()([x, x0])
    x = _upscale_block(x)

    x = Conv2D(1, (3, 3), activation="linear", padding='same', name='sr_res_conv_final')(x)
    out = x;
    model = Model(input=_input, output=out)
    return model


def InceptionResNet_train():
    InceptionResNet = model_InceptionResNet()
    InceptionResNet.compile(optimizer=adam(lr=0.0003), loss='mse', metrics=[PSNRLoss])
    print (InceptionResNet.summary())

    data, label = pd.read_training_data("drive/SuperResolution/onetrain2x.h5")

    val_data, val_label = pd.read_training_data("drive/SuperResolution/oneval2x.h5")
#    data, label = pd.read_training_data("Nov_test/onetrain2x.h5")
#
#    val_data, val_label = pd.read_training_data("Nov_test/oneval2x.h5")
    
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=25, 
                                  min_lr=0.000001, verbose=1)
    checkpoint = ModelCheckpoint("drive/SuperResolution/oneResNetXx2_model.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
#    checkpoint = ModelCheckpoint("Nov_test/oneResNetXx2_model.h5", monitor='val_loss', verbose=1, save_best_only=True,
#                                 save_weights_only=False, mode='min')
    callbacks_list = [reduce_lr,checkpoint]

    InceptionResNet.fit(data, label, batch_size=64, validation_data=(val_data, val_label),
                               callbacks=callbacks_list, shuffle=True, nb_epoch=200, verbose=1)
#    InceptionResNet.save_weights("drive/SuperResolution/ResNetBFInce_final.h5")

def InceptionResNet_predict():
    IMG_NAME = "image5/img.jpg"
    INPUT_NAME = "image5/input.jpg"
    OUTPUT_NAME = "image5/ResNetX_pre.jpg"

    label = cv2.imread(IMG_NAME)
    shape = label.shape

    img = cv2.resize(label, (int(shape[1] / scale), int(shape[0] / scale)), cv2.INTER_CUBIC)
    cv2.imwrite(INPUT_NAME, img)

    InceptionResNet= model_InceptionResNet()
    InceptionResNet.load_weights("result_code/ResNetXx2_model.h5")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
    Y[0, :, :, 0] = img[:, :, 0].astype(float) / 255.
    img = cv2.cvtColor(label, cv2.COLOR_BGR2YCrCb)

    pre = InceptionResNet.predict(Y, batch_size=1) * 255.
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
    print ("InceptionResNet:")
    print (cv2.PSNR(im1[:, :, 0], im3[:, :, 0]))


##*******************************************************************************************************************//
if __name__ == "__main__":
#    InceptionResNet_train()
    InceptionResNet_predict()
    #vilization_and_show()
