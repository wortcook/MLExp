import torch
import numpy as np
import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

def run_experiment(model, train_images, train_labels, val_images, val_labels, eval_images, eval_labels):

    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
    learning_rate_decay = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=0.00005)

    callbacks = [early_stopping, learning_rate_decay]

    freezable_layers = []

    #loop through the model
    for layer in model.layers:
        if type(layer) == layers.Conv2D:
            freezable_layers.append(layer)
        elif type(layer) == layers.Dense:
            freezable_layers.append(layer)

    #randomly freeze 1 layer
    freeze_layer = np.random.choice(freezable_layers)
    freeze_layer.trainable = False

    history = model.fit(train_images, train_labels, epochs=200, validation_data=(val_images, val_labels), callbacks=callbacks, verbose=0)
    test_loss, test_acc = model.evaluate(eval_images,  eval_labels, verbose=2)

    #unfreeze the layers
    for layer in model.layers:
        layer.trainable = True

    return test_acc

model = build_model(32, 1)

test_accs = []

for i in range(1,101):
    #get a random seed
    seed = np.random.randint(10000)

    #split the data
    run_train_images, run_val_images, run_train_labels, run_val_labels = train_test_split(train_images, train_labels, test_size=0.4, random_state=seed)

    acc = run_experiment(model, run_train_images, run_train_labels, run_val_images, run_val_labels, test_images, test_labels)

    print(f"Run: {i}")
    print(f"Accuracy: {acc}")

    #get test accuracy
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    test_accs.append(test_acc)


    #plot the results every 10 experiments
    if i % 10 == 0:
        print(f"Mean test accuracy: {np.mean(test_accs)}")
        print(f"Std test accuracy: {np.std(test_accs)}")

        #plot the results
        plt.plot(test_accs)
        plt.xlabel('Experiment')
        plt.ylabel('Test Accuracy')
        plt.show()


print(f"Mean test accuracy: {np.mean(test_accs)}")
print(f"Std test accuracy: {np.std(test_accs)}")

#plot the results
plt.plot(test_accs)
plt.xlabel('Experiment')
plt.ylabel('Test Accuracy')
plt.show()