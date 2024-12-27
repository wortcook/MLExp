
import numpy as np
import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

print(train_images.shape)
print(test_images.shape)

def build_model(unit_base, layer_depth):
    # Create the CNN model
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(32, 32, 3)))

    for i in range(layer_depth):
        model.add(layers.Conv2D(unit_base, (3, 3), activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D((2, 2)))

    for i in range(layer_depth):
        model.add(layers.Conv2D(2*unit_base, (3, 3), activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D((2, 2)))

    for i in range(layer_depth):
        model.add(layers.Conv2D(3*unit_base, (3, 3), activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model

def run_experiment(unit_base, layer_depth):
    model = build_model(unit_base, layer_depth)

    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
    learning_rate_decay = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=0.00005)

    callbacks = [early_stopping, learning_rate_decay]

    history = model.fit(train_images, train_labels, epochs=200, validation_data=(test_images, test_labels), callbacks=callbacks)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    return test_acc

unit_base = 32
layer_depth = 1

run_results = []

for ub in range(32, 256, 4):
    acc = run_experiment(ub, layer_depth)
    run_results.append((ub, acc))
    print('Unit base: %d, Accuracy: %.4f' % (ub, acc))

run_results = np.array(run_results)
plt.plot(run_results[:,0], run_results[:,1])
plt.xlabel('Unit base')
plt.ylabel('Accuracy')

plt.show()



