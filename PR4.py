import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kMeansClustering(image_path, num_clusters):
     # Загрузка изображения
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

   # Преобразование изображения в формат, удобный для обработки
    pixels = image.reshape((-1, 3))

     # Создание объекта KMeans с указанием количества кластеров
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)
    # Получение меток кластеров для каждой точки данных
    labels = kmeans.labels_
    centers = np.uint8(kmeans.cluster_centers_)
    clustered_image = centers[labels.flatten()]
    clustered_image = clustered_image.reshape(image.shape)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Оригинальное изображение')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(clustered_image)
    plt.title('Кластеризованное изображение ({} кластера)'.format(num_clusters))
    plt.axis('off')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Цикл до сходимости: перемещение центров и пересчет меток
    for label, color in zip(range(num_clusters), centers):
        ax.scatter(pixels[labels == label][:, 0], pixels[labels == label][:, 1], pixels[labels == label][:, 2],
                   c=[color / 255])
    ax.set_xlabel('Красный')
    ax.set_ylabel('Зеленый')
    ax.set_zlabel('Синий')
    ax.set_title('Сгруппированные пиксели в трехмерном пространстве')
    plt.show()

image_path = "zuckanddana.jpg"
num_clusters = 3
kMeansClustering(image_path, num_clusters)
