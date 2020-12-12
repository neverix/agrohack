import os
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


sns.set_theme()
sns.set_style("white")
data_path = "./train_public/"
model_name = "./first_try.h5"

data_path = os.path.abspath(data_path)
datagen = ImageDataGenerator(rescale=1 / 255)
test_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    shuffle=False,
    class_mode='binary',
)

if os.path.exists("predictions.json"):
    predictions = json.load(open("predictions.json"))
else:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    # необходимо добавить, чтобы программа работала на локальном компьютере
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model = load_model("model.h5", custom_objects={'f1': metrics.f1_score})
    predictions = [float(x) for x in model.predict(test_generator, verbose=True)]
    json.dump(predictions, open("predictions.json", 'w'))

ts = []
f1s = []
best_f1 = 0
best_threshold = 0
for i in range(100):
    threshold = i / 100
    ts.append(threshold)
    y = [int(x > threshold) for x in predictions]
    f1 = metrics.f1_score(y, test_generator.classes)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
    f1s.append(f1)
plt.plot(ts, f1s)
plt.savefig("plots/threshold.png")
plt.show()

conf = metrics.confusion_matrix([int(x > best_threshold) for x in predictions], test_generator.classes)
conf = metrics.ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=[0, 1])
conf.plot()
plt.axis("off")
plt.gca().images[-1].colorbar.remove()
plt.savefig("plots/confmatrix.png")
plt.show()

model_predictions = np.array([int(x > best_threshold) for x in predictions])
indices = lambda z: [(y, i) for i, y in enumerate(z)]
unindices = lambda z: [i for x, i in z]
most_confident_positive = sorted(indices(predictions))[-5:]
most_confident_negative = sorted(indices(predictions))[:5]
wrong = model_predictions != test_generator.classes
fp = np.where(np.logical_and(test_generator.classes == 1, wrong))[:5]
fn = np.where(np.logical_and(test_generator.classes == 0, wrong))[:5]

