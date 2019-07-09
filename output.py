import copy
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 读取模型
model = keras.models.load_model("MyModel.h5")

def plot_images(images, test_labels, n):
    index = int (n - 1 / 25)
    for i in range(0, index):
        plt.figure(figsize=(10, 10))
        for j in range(25):
            plt.subplot(5, 5, j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid('off')
            plt.imshow(images[j + i * 25], cmap=plt.cm.binary)
            predictions = model.predict(images)
            predicted_label = np.argmax(predictions[j + i * 25])
            true_label = test_labels[j + i * 25]
            if predicted_label == true_label:
                color = 'blue'
            else:
                color = 'red'
            plt.xlabel("{} ({})".format(class_names[predicted_label],
                                        class_names[true_label]),
                       color=color)
        plt.show()


def get_label(index):
    return class_names[index]