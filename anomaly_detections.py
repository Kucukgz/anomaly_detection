"""
Image Processing Library.

Wow!
"""
import os
import numpy as np
from skimage.io import imread
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from operator import itemgetter
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.ensemble import IsolationForest
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from pyoptflow import HornSchunck


def _get_measurements(folder):
    # list image files
    filenames = os.listdir(folder)
    # sort the image filenames
    filenames = sorted(filenames, key=lambda v: v.upper())
    nl, bs, bs2, ai, dl, di, cl, mz = [], [], [], [], [], [], [], []
    for filename in filenames:
        print(filename)
        filename = os.path.join(folder, filename)
        im = imread(filename)
        im = np.moveaxis(im, 0, -1)
        for i in range(im.shape[2]):
            nl.append(estimate_sigma(im[:, :, i], multichannel=False, average_sigmas=True))
            imlap = laplace(im[:, :, i])
            bs.append(imlap.var())  # Blurriness Score
            im2 = gaussian_filter(im[:, :, i], sigma=3)
            bs2.append(im2.var())  # Blurriness Score with Gaussian Filter
            ai.append(im[:, :, i].mean())  # Average Intensity
            dl.append(_get_dark_light(im[:, :, i]))  # Darkness Level
            di.append(_get_dominant_intensity(im[:, :, i]))  # Dominant intensity
            imgx, imgy = np.gradient(im[:, :, i])
            img = np.sqrt(np.power(imgx, 2) + np.power(imgy, 2))
            cl.append(np.sum(img) / (im.shape[0] * im.shape[1]))  # Contrast Level
        for i in range(im.shape[2] - 1):
            _, _, m, _ = _motion_estimation(im[:, :, i], im[:, :, i + 1])
            ali = np.sum(m)
            mz.append(ali)  # Motion Estimation
    return nl, bs, bs2, ai, dl, di, cl, mz


def _motion_estimation(im1, im2, a=1.0, n=100):
    u, v = HornSchunck(im1, im2, alpha=a, Niter=n)
    m = np.sqrt(np.power(u, 2) + np.power(v, 2))  # magnitude
    a = np.arctan2(u, v)  # angle
    return m, v, u, a


def _get_dark_light(im):
    # im - grayscale [0-255]
    # intensity palette of the image
    palette = defaultdict(int)
    for pixel in np.nditer(im):
        palette[int(pixel)] += 1
    # sort the intensity present in the image
    sorted_x = sorted(palette.items(), key=itemgetter(1), reverse=True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for _, x in enumerate(sorted_x[:pixel_limit]):
        if x[0] <= 20:  # dull : too much darkness
            dark_shade += x[1]
        if x[0] >= 240:  # bright : too much whiteness
            light_shade += x[1]
        shade_count += x[1]
    light_percent = round((float(light_shade) / shade_count) * 100, 2)
    dark_percent = round((float(dark_shade) / shade_count) * 100, 2)
    return dark_percent


def _get_dominant_intensity(im):
    # k-means
    kmeans_cluster = KMeans(n_clusters=5)
    kmeans_cluster.fit(im)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    # dominant intensity
    palette = np.uint8(cluster_centers)
    dominant_intensity = palette[np.argmax(
                          np.unique(cluster_labels, return_counts=True)[1])]
    # from vector [1,...,z] - > 1 number
    dominant_intensity = np.median(dominant_intensity)
    return dominant_intensity


def figure(x):
    """To figure all features."""
    for i, j in x:
        plt.figure()
        plt.scatter(range(len(i)), np.sort(i))
        plt.xlabel('Index')
        plt.ylabel(f'{j}')
        plt.title(f"{j} Distribution")
        sns.despine()
        plt.show()
        plt.figure()
        sns.distplot(i)
        plt.title(f"Distribution of {j}")
        sns.despine()
        plt.show()


def anomaly_score(x):
    """To figure out anomaly scores."""
    # must calibrate it for all measurements
    outliers = []
    for i, j in x:
        pd_i = pd.DataFrame(i)
        isolation_forest = IsolationForest(n_estimators=100, contamination=float(.01))
        isolation_forest.fit(pd_i.values.reshape(-1, 1))
        xx = np.linspace(pd_i.min(), pd_i.max(), len(pd_i)).reshape(-1, 1)
        anomaly_score = isolation_forest.decision_function(xx)
        outlier = isolation_forest.predict(xx)
        isoF_outliers_values = pd_i[isolation_forest.predict(xx) == -1]
        outliers.append(isoF_outliers_values)
        plt.figure(figsize=(10, 4))
        plt.plot(xx, anomaly_score, label='anomaly score')
        plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score),
                         where=outlier == -1, color='r',
                         alpha=.4, label='Outlier Region')
        plt.legend()
        plt.ylabel(f'Anomaly Score')
        plt.xlabel(f'{j}')
        plt.show()
    return outliers


def skew_kurt(x):
    """To figure out distribution."""
    for i, j in x:
        pd_x = pd.DataFrame(i)
        print(f"Skewness of {j}: %f" % pd_x.skew())
        print(f"Kurtosis of {j}: %f" % pd_x.kurt())


def grabbing_outliers(x, folder):
    """To grab outliers."""
    outliers_list = []
    for i in x:
        detections = i.index.values.tolist()
        outliers_list.append(detections)
    return outliers_list


def dropping_outliers(x):
    """To drop off outliers."""
    dropped_outliers = []
    for i in x:
        # print(i)
        for j in i:
            # print(j)
            dropped_outliers.append(j)
    dropped_outliers = set(dropped_outliers)
    return dropped_outliers


if __name__ == '__main__':
    # load images
    folder = 'data'
    nl, bs, bs2, ai, dl, di, cl, mz = _get_measurements(folder)
    values = ((nl, 'Noise Level'), (bs, 'Blurriness Score'), (bs2, 'Blurriness Score with Gaussian Filter'),
              (ai, 'Average Intensity'), (dl, 'Darkness Level'), (di, 'Dominant Intensity'), (cl, 'Contrast Level'),
              (mz, 'Motion Estimation'))
    skew_kurt(values)
    # figure(values)
    outliers = anomaly_score(values)
    grapped_outliers = grabbing_outliers(outliers, folder)

    dropped_outliers = dropping_outliers(grapped_outliers)
    with open("outliers.txt", "w") as f:
        for s in dropped_outliers:
            f.write(str(s) + "\n")
