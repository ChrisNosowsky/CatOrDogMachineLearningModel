

import os
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import matplotlib.pyplot as plt
import numpy as np
original_dataset_dir = 'C:/Users/Racec/OneDrive/Documents\MachineLearning/DogsVsCats/dogs-vs-cats/train/train' # This is where original dataset goes


base_dir = '/Users/Racec/OneDrive/Documents/MachineLearning/DogsVsCats' # This is where your test, train and validation data goes
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
# train_cats_dir = os.path.join(train_dir, 'cats')
# train_dogs_dir = os.path.join(train_dir, 'dogs')
# validation_cat_dir = os.path.join(validation_dir, 'cats')
# validation_dog_dir = os.path.join(validation_dir, 'dogs')
# test_cats_dir = os.path.join(test_dir, 'cats')
# test_dogs_dir = os.path.join(test_dir, 'dogs')

# print('total training cat images:', len(os.listdir(train_cats_dir)))
# print('total training dog images:', len(os.listdir(train_dogs_dir)))
# print('total validation cat images:', len(os.listdir(validation_cat_dir)))
# print('total validation dog images:', len(os.listdir(validation_dog_dir)))
# print('total test cat images:', len(os.listdir(test_cats_dir)))
# print('total test dog images:', len(os.listdir(test_dogs_dir)))
conv_base = VGG16(weights="imagenet",
                  include_top=False,
                  input_shape=(150,150,3))

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False





train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')



# def extract_features(directory, sample_count):
#     features = np.zeros(shape=(sample_count, 4, 4, 512))
#     labels = np.zeros(shape=(sample_count))
#     generator = datagen.flow_from_directory(
#         directory,
#         target_size=(150,150),
#         batch_size=batch_size,
#         class_mode='binary')
#     i=0
#     for inputs_batch, labels_batch in generator:
#         features_batch = conv_base.predict(inputs_batch)
#         features[i * batch_size : (i+1) * batch_size] = features_batch
#         labels[i * batch_size : (i+1) * batch_size] = labels_batch
#         i+=1
#         if i * batch_size >= sample_count:
#             break
#     return features, labels

# train_features, train_labels = extract_features(train_dir, 2000)
# validation_features, validation_labels = extract_features(validation_dir, 1000)
# test_features, test_labels = extract_features(test_dir, 1000)

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))




model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

test_loss, test_acc = model.evaluate_generator(test_generator,
                                               steps = 50)

print('test acc: ', test_acc)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data = validation_generator,
    validation_steps=50)




model.save('cats_and_dogs_small_2.h5')


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) +1)

plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()




