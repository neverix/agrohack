import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_path = "./train_public/"
model_name = "model-Xception.h5"
threshold = 0.1

# необходимо добавить, чтобы программа работала на локальном компьютере
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_path = os.path.abspath(data_path)
datagen = ImageDataGenerator(rescale=1/255)
test_generator = datagen.flow_from_directory(
    os.path.dirname(data_path),
    target_size=(331, 331),
    shuffle=False,
    class_mode='binary',
    classes=[os.path.basename(data_path)]
)
filenames = test_generator.filenames

model = load_model(model_name, custom_objects={"f1": None})
predict = model.predict(test_generator, batch_size=1, verbose=True)
xy = zip(filenames, [int(x[0] > threshold) for x in predict])

df = pd.DataFrame(xy, columns=["name", "disease_flag"])
df.to_csv("OBP_result.csv")
