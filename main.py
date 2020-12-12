import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile as Base
from tensorflow.keras.layers import Dropout, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_path = "./train_public/"
model_name = "./first_try.h5"
threshold = 0.5

# необходимо добавить, чтобы программа работала на локальном компьютере
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

img_shape = (224, 224)


# неоптимально, но так сохранена модель
def make_net():
    base = Base(input_shape=img_shape + (3,), include_top=False, weights="imagenet")
    for layer in base.layers[:-11]:
        layer.trainable = False
    head = base.output
    head = MaxPooling2D(pool_size=(7, 7))(head)
    head = Flatten()(head)
    head = Dense(64, activation='relu')(head)
    head = Dropout(0.5)(head)
    head = Dense(1, activation='sigmoid')(head)
    model = Model(inputs=base.input, outputs=head)
    return model


data_path = os.path.abspath(data_path)
datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = datagen.flow_from_directory(
    os.path.dirname(data_path),
    target_size=img_shape,
    shuffle=False,
    class_mode='binary',
    classes=[os.path.basename(data_path)]
)
filenames = test_generator.filenames

model = make_net()
model.load_weights(model_name)  # неоптимально, но так сохранена модель
predict = model.predict(test_generator, verbose=True)
xy = zip(filenames, [int(x[0] > threshold) for x in predict])

df = pd.DataFrame(xy, columns=["name", "disease_flag"])
df.to_csv("OBP_result.csv")
