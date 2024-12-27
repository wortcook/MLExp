import tensorflow as tf
import keras

from keras import models
from keras import layers

def build_model():

    inp_shape = (32, 32, 3)

    inp = layers.Input(shape=inp_shape)

    x2 = layers.Multiply()([inp, inp])
    x2 = layers.Conv2D(64, (7, 7), activation='relu', padding='same', name='x2')(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.BatchNormalization()(x2)

    x1 = layers.Conv2D(64, (7, 7), activation='relu', padding='same', name='x1')(inp)
    x1 = layers.Dropout(0.2)(x1)
    x1 = layers.BatchNormalization()(x1)

    #create lambda layer that returns a constant 1.0
    x0 = layers.Lambda(lambda x: tf.constant(1.0), output_shape=(32,32,64))(inp)
    x0 = layers.Dropout(0.2)(x0)

    x = layers.Subtract()([x2, x1])
    x = layers.Subtract()([x, x0])

    x = layers.Conv2D(128, (7, 7), activation='relu', padding='same', name='fy')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=x)

    print(model.summary())

    #create a visual representation of the model
    # keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    # #show the image model.png
    # from PIL import Image
    # img = Image.open('model.png')
    # img.show()


    return model

def run_experiment(train_images, train_labels, test_images, test_labels):
    model = build_model()

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    learning_rate_decay = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.618033, patience=8, min_lr=0.00005)

    callbacks = [early_stopping, learning_rate_decay]

    history = model.fit(train_images, train_labels, epochs=200, validation_data=(test_images, test_labels), callbacks=callbacks)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    #visualize the weights of layers x2, x1, and fy

    #get the weights of the layers
    layer_x2 = model.get_layer('x2')
    layer_x1 = model.get_layer('x1')
    layer_fy = model.get_layer('fy')

    #get the weights of the layers
    weights_x2 = layer_x2.get_weights()
    weights_x1 = layer_x1.get_weights()
    weights_fy = layer_fy.get_weights()

    #display the weights as images
    import matplotlib.pyplot as plt

    #show all 32 filters of layer x2, x1, and fy

    fig, axs = plt.subplots(3, 32, figsize=(32, 3))

    # for i in range(32):
    #     axs[0, i].imshow(weights_x2[0][:,:,0,i])
    #     axs[0, i].axis('off')

    #     axs[1, i].imshow(weights_x1[0][:,:,0,i])
    #     axs[1, i].axis('off')

    #     axs[2, i].imshow(weights_fy[0][:,:,0,i])
    #     axs[2, i].axis('off')

    # plt.show()
    

    return test_acc


def main():
    cifar = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar.load_data()

    train_images = 1.0 - 2.0*(x_train / 255.0)
    test_images  = 1.0 - 2.0*(x_test / 255.0)

    test_acc = run_experiment(train_images, y_train, test_images, y_test)

    print(test_acc)


if __name__ == "__main__":
    main()