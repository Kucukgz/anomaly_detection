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

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from numpy import quantile, where, random
import numpy
import plotly.express as px
import plotly.graph_objects as go
import plotly.io


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
            nl.append(estimate_sigma(
                im[:, :, i], multichannel=False, average_sigmas=True))
            imlap = laplace(im[:, :, i])
            bs.append(imlap.var())  # Blurriness Score
            im2 = gaussian_filter(im[:, :, i], sigma=3)
            bs2.append(im2.var())  # Blurriness Score with Gaussian Filter
            ai.append(im[:, :, i].mean())  # Average Intensity
            dl.append(_get_dark_light(im[:, :, i]))  # Darkness Level
            di.append(_get_dominant_intensity(
                im[:, :, i]))  # Dominant intensity
            imgx, imgy = np.gradient(im[:, :, i])
            img = np.sqrt(np.power(imgx, 2) + np.power(imgy, 2))
            # Contrast Level
            cl.append(np.sum(img) / (im.shape[0] * im.shape[1]))
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
        isolation_forest = IsolationForest(
            n_estimators=100, contamination=float(.00001))
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
        plt.title(f'Isolation Forest Detection of {j}', fontsize=15)
        plt.ylabel(f'Anomaly Score', fontsize=15)
        plt.savefig(f'isoF_images/LOF_{j}', format='png', dpi=1200)
        plt.show()
    return outliers


def svm_anomaly_score(df_data):
    """To figure out anomaly scores."""
    # must calibrate it for all measurements
    outliers = []
    for label, content in df_data.items():
        df_data[f'{label}'] = df_data[f'{label}'].fillna(0)
        svm = OneClassSVM(kernel='rbf', gamma=0.00001, nu=0.03)
        pred = svm.fit_predict(df_data[f'{label}'].values.reshape(-1, 1))
        scores = svm.score_samples(df_data[f'{label}'].values.reshape(-1, 1))

        thresh = quantile(scores, 0.008)
        feature_score = []
        anom = []
        inliers_feature_score = []
        inliers = []
        kazim = []
        ali = []
        for i, j in enumerate(scores):
            if j <= thresh:
                outliers.append(i)
                anom.append(j)
                feature_score.append(df_data[f'{label}'][i])
                ali.append(i)
            else:
                inliers.append(j)
                inliers_feature_score.append(df_data[f'{label}'][i])
                kazim.append(i)

        inliers_pd = pd.DataFrame({'inliers': inliers, 'inliers_feature_score': inliers_feature_score,
                                  'inliers_index': kazim})
        pd_anom = pd.DataFrame({'AnomScore': anom, 'FeatureScore': feature_score, 'outlier_index': ali})

        fig = go.Figure()
        fig.update_layout(title={
            'text': f"SVM Detection of {label}", 'y': 0.97, 'x': 0.5},
            paper_bgcolor='white', plot_bgcolor="rgb(211, 216, 230)",
            # xaxis_title=" ",
            yaxis_title="Anomaly Score",
            font=dict(family="Courier New, monospace", size=50, color="rgb(10, 16, 87)"),
            title_font_color='rgb(145, 0, 0)',
            shapes=[dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=df_data[f'{label}'].min(),
                    y0=thresh,
                    x1=df_data[f'{label}'].max(),
                    y1=thresh,
                    opacity=1,
                    line=dict(color='blue', dash='dot')
                    )])

        fig.add_trace(go.Scatter(x=inliers_pd['inliers_feature_score'], y=inliers_pd['inliers'],
                                 mode='markers', marker=dict(size=6, color='rgb(0, 0, 0)'),
                                 name='Normal', marker_symbol='circle'))
        fig.add_trace(go.Scatter(x=pd_anom['FeatureScore'], y=pd_anom['AnomScore'],
                                 mode='markers', marker=dict(size=14, color='rgb(255, 0, 0)'),
                                 name='Abnormal', marker_symbol=206))

        fig.show()
        plotly.io.write_image(fig, f'SVM_images/{label}.png', width=2560, height=1440)
    return outliers


def LOF_anomaly_score(x):
    """To figure out anomaly scores."""
    # must calibrate it for all measurements
    outliers = []
    outliers_list = []
    for i, j in x:
        pd_i = pd.DataFrame(i)
        method = 1
        k = 30
        clf = LocalOutlierFactor(n_neighbors=k , algorithm='auto', contamination=0.1, n_jobs=-1)
        clf.fit(pd_i)
        # Record k neighborhood distance
        pd_i['k distances'] = clf.kneighbors(pd_i)[0].max(axis=1)
        # Record LOF factor，take negative
        pd_i['local outlier factor'] = -clf._decision_function(pd_i.iloc[:, :-1])
        # Separate group points and normal points according to the threshold
        outliers = pd_i[pd_i['local outlier factor'] > method].sort_values(by='local outlier factor')
        inliers = pd_i[pd_i['local outlier factor'] <= method].sort_values(by='local outlier factor')
        # Figure
        plt.rcParams['axes.unicode_minus'] = False  # display the negative sign
        plt.figure(figsize=(8, 4)).add_subplot(111)
        plt.scatter(pd_i[pd_i['local outlier factor'] > method].index,
                    pd_i[pd_i['local outlier factor'] > method]['local outlier factor'], c='red', s=50,
                    marker='.', alpha=None,
                    label='outliers')
        plt.scatter(pd_i[pd_i['local outlier factor'] <= method].index,
                    pd_i[pd_i['local outlier factor'] <= method]['local outlier factor'], c='black', s=50,
                    marker='.', alpha=None, label='inliers')
        plt.hlines(method, -2, 2 + max(pd_i.index), linestyles='--')
        plt.xlim(-2, 2 + max(pd_i.index))
        plt.title(f'LOF Local outlier detection of {j}', fontsize=13)
        plt.ylabel('Anamoly Score', fontsize=15)  # Local outlier Factors
        plt.legend()
        plt.savefig(f'LOF_images/LOF_{j}', format='png', dpi=1200)
        plt.show()
        outliers_list.append(list(outliers.index))
    return outliers_list


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
    return dropped_outliers


def save_values(values, path):
    """To save numpy arrays."""
    k = 0
    for i, j in values:
        npy_data = np.array(i)
        target_name = os.path.join(path, j + "_" + str(k) + ".npy")
        np.save(target_name, npy_data)
        k += 1
    return


def load_values(path):
    """To load numpy arrays."""
    names = {i: [] for i in os.listdir(path)}
    filename = []
    np_list = []
    np_data = ()
    df = ()
    name_list = []
    for key, value in names.items():
        filename.append(key)
        np_list.append(value)
    for i in filename:
        splitting_name = i.replace(".npy", "")
        # number = int(splitting_name.split('_')[1])
        name = splitting_name.split('_')[0]
        np_list = np.load(path + i)
        np_data += ((np_list.tolist(), name),)
        df += (np_list.tolist(),)
        name_list.append(name)
    df_data = pd.DataFrame(df)
    df_data = df_data.T
    df_data.columns = name_list
    return np_data, df_data


def save_txt(x, name):
    """To save anomalyies as a text file."""
    with open(f"{name}_anomalies.txt", "w") as f:
        for s in x:
            f.write(str(s) + "\n")
    return


if __name__ == '__main__':
    # load images
    folder = '/home/burak/vsc/first_part/__private__/__private__ (copy)/Data/original'
    path = "measurements/single_npy/"

    # Save measurements as numpy arrays
    # nl, bs, bs2, ai, dl, di, cl, mz = _get_measurements(folder)
    # values = ((nl, 'NoiseLevel'), (bs, 'BlurrinessScore'), (bs2, 'BlurrinessScoreWithGaussianFilter'),
    #           (ai, 'AverageIntensity'), (dl, 'DarknessLevel'), (di, 'DominantIntensity'), (cl, 'ContrastLevel'),
    #           (mz, 'MotionEstimation'))
    # save_values(values, path)

    values, df_data = load_values(path)
    skew_kurt(values)
    figure(values)

    isoF_outliers = anomaly_score(values)
    grapped_outliers = grabbing_outliers(isoF_outliers, folder)
    isoF_dropped_outliers = set(dropping_outliers(grapped_outliers))

    svm_outliers = svm_anomaly_score(df_data)
    svm_dropped_outliers = set(svm_outliers)

    lof_outliers = LOF_anomaly_score(values)
    grapped_outliers = [val for sublist in lof_outliers for val in sublist]
    lof_dropped_outliers = set(grapped_outliers)

    save_txt(sorted(isoF_dropped_outliers), name='isoF')
    save_txt(sorted(svm_dropped_outliers), name='svm')
    save_txt(sorted(lof_dropped_outliers), name='LOF')

