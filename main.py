# import cv2.cv as cv
from tensorflow import keras
import tensorflow as tf
import random
import numpy as np
import copy
import time
import ssim
# import output

# 读取模型
model = keras.models.load_model("MyModel.h5")

# 目标算法
def aiTest(images, shape):
    new_images = []
    size = shape[0]
    for i in range(size):
        new_image = attack(images[i])

        new_images.append(new_image)

    # 转回ndarray数组
    res = np.array(new_images)
    return res

def load_model():
    model = tf.keras.models.load_model('MyModel.h5')
    return model

# 图片中间一半涂白
def half_white(image):
    new_image = copy.deepcopy(image)
    color = 1  # 涂白
    for x in range(5, 23):
        for y in range(3, 25):
            new_image[x][y] = color
    return new_image

# 返回预测正确的标签
def get_true_label(image):
    predictions = model.predict(np.expand_dims(image, 0))
    prediction = predictions[0]
    lable = np.argmax(prediction)
    return lable

# 攻击算法,最大1500次循环
def attack(image):
    # 最大循环次数
    N = 300

    # 正确的标签
    label = get_true_label(image)

    max_image = draw_shape(image)
    max_ssim = ssim.compute_ssim(max_image, image)
    for w in range(5):   # 图形宽度
        new_image = draw_shape(image, offset=w + 1)
        index = 0

        # 选取相似度在85%以上的
        while index < N and (0.85 > ssim.compute_ssim(new_image, image) or label == get_true_label(new_image)):
            # 加大宽度继续画
            new_image = draw_shape(image, offset=w + 1)

            index = index +  1

            # 保留相似度最高的攻击成功的图片
            if (not label == get_true_label(new_image)) and max_ssim < ssim.compute_ssim(new_image, image):
                max_image = copy.deepcopy(new_image)

        if index < N:
            return new_image

    # 相似度最高的图片攻击失败，则全部涂白
    if label == get_true_label(max_image):
        return half_white(image)
    return max_image


# 随机画图形
def draw_shape(image, offset=1):
    new_image = copy.deepcopy(image)

    (width, height) = new_image.shape

    # 先随机修改15个像素对
    for k in range(10):
        x1 = random.randint(0, new_image.shape[0] - 1)
        y1 = random.randint(0, new_image.shape[1] - 1)
        x2 = random.randint(0, new_image.shape[0] - 1)
        y2 = random.randint(0, new_image.shape[1] - 1)
        temp = new_image[x1][y1]
        new_image[x1][y1] = new_image[x2][y2]
        new_image[x2][y2] = temp

    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)

    # 顺便再画一条杠
    for i in range(5, width-5):
            new_image[5][i] = 50

    # 画米字
    for i in range(0, offset):
        if x + i < height:
            for j in range(width):
                # 随机像素
                new_image[x + i][j] = random.randint(0, 255)

        if y + i < width:
            for j in range(0, height):
                new_image[j][y + i] = random.randint(0, 255)

    return new_image


if __name__ == '__main__':
    # 下载数据集
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 输出结果
    start = time.time()
    images = aiTest(test_images, (100, 28, 28, 1))
    print("攻击时间", time.time() - start) #修改传入的所有图片的花费时间

    count = 0
    ssim_sum = 0.0
    start = time.time()
    for i in range(len(images)):
        new_image = attack(images[i])
        if not get_true_label(new_image) == get_true_label(images[i]):
            count += 1
            ssim_sum += ssim.compute_ssim(new_image, images[i])
            print(i, ssim.compute_ssim(new_image, images[i]))
    end = time.time()
    print("输出时间", end - start)
    print("成功个数：", count, ",  成功率：", count/100, ", 平均相似度：", ssim_sum/count)

    # util.plot_images(images, test_labels, 100)
