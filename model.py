
import tensorflow as tf
from tensorflow import keras

'''
用于训练并生成模型 MyModel.h5
'''

def create_model():
    # 构建模型
    model = keras.Sequential([
        # 将二维数组转换成一维数组
        keras.layers.Flatten(input_shape=(28, 28)),
        # 密集连接层
        keras.layers.Dense(128, activation=tf.nn.relu),
        # 返回10个概率得分的数组，表示当前图像属于10个类别中某一个的概率
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # 编译模型
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(train_images, train_labels, epochs=25)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    return model

# 下载数据集，包含70000张灰度图像，10个类别
fashion_mnist = keras.datasets.fashion_mnist
# 60000张用于训练，10000张用于测试
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 类别标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_labels))
print(train_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0

# 评估准确率
model = create_model()
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\ntest loss:', test_loss)
print('accuracy:', test_acc)

model.save('MyModel.h5',  include_optimizer=True)



# 做出预测
# attacks = main.aiTest(test_images, (25, 28, 28, 1))
# predictions = model.predict(test_images)
# for i in range(0, 25):
#     print(predictions[i])
#     print(np.argmax(predictions[i]))
# print(predictions)
# prediction = predictions[0]
# print("You will find that the ans is same to the former res", np.argmax(prediction))
# print(test_labels[0])

# # 将预测绘制成图，查看全部10个通道
# plt.figure(figsize=(10, 10))
# def plot_image(i, predictions_array, true_label, img):
#     predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label:
#         color = 'green'
#     else:
#         color = 'red'
#
#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                          100 * np.max(predictions_array),
#                                          class_names[true_label]),
#                color=color)

#　画出每个图片的 10 个预测概率的柱状图
# def plot_value_array(i, predictions_array, true_label):
#     predictions_array, true_label = predictions_array[i], true_label[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     thisplot = plt.bar(range(10), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#
#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')

# 查看第0张图像、预测和预测数组
# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions, test_labels)

# i = 12
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions, test_labels)

# 将预测绘制成图像，正确的预测标签为蓝色，错误的预测标签为红色，数字表示预测标签的百分比
# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
# num_rows = 5
# num_cols = 3
# num_images = num_rows * num_cols
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions, test_labels)

# plt.show()

# 用模型预测单个图像
# Grab an image from the test dataset
# img = test_images[0]
# print(img.shape)
#
# # Add the image to a batch where it's the only member.
# img = (np.expand_dims(img, 0))
# print(img.shape)
#
# # 预测图像
# predictions_single = model.predict(img)
# print(predictions_single)
# plot_value_array(0, predictions_single, test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
#
# np.argmax(predictions_single[0])
