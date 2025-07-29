import os
import pickle

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score

from co_cluster_bank import co_training_kmeans, co_training_gmm


def read_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_to_pkl(data_dict, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def compute_class_means(X, labels, K):
    D = X.shape[1]
    means = []
    for k in range(K):
        class_mask = labels == k
        if np.any(class_mask):
            mean_vec = X[class_mask].mean(axis=0)
        else:
            mean_vec = np.zeros(D)
        means.append(mean_vec)
    return np.stack(means, axis=0)

def plot_cluster_mean_kde_grid(X_2d, other_2d, labels_1, labels_2, method, cluster_method_name, modality, save_path):
    """
    生成 2×2 子图，分别显示 labels_1 / labels_2 的 KDE + 均值图，适用于 X_2d 和 other_2d。
    """

    fig, axes = plt.subplots(2, 2, figsize=(20, 18), dpi=300)
    axes = axes.flatten()
    titles = [
        f"{cluster_method_name.upper()} Mean+KDE on {modality} (Label1)",
        f"{cluster_method_name.upper()} Mean+KDE on Other Modality (Label1)",
        f"{cluster_method_name.upper()} Mean+KDE on {modality} (Label2)",
        f"{cluster_method_name.upper()} Mean+KDE on Other Modality (Label2)"
    ]

    datasets = [X_2d, other_2d, X_2d, other_2d]
    labels_all = [labels_1, labels_1, labels_2, labels_2]

    # 提前定义调色板（支持更多颜色）
    all_labels = np.concatenate([labels_1, labels_2])
    n_classes = int(np.max(all_labels)) + 1
    palette = sns.color_palette("hls", n_classes)

    for i in range(4):
        ax = axes[i]
        data = datasets[i]
        labels = np.array(labels_all[i])

        # 绘制淡颜色散点
        for label in np.unique(labels):
            idx = labels == label
            color = palette[label]
            ax.scatter(
                data[idx, 0], data[idx, 1],
                color=color, alpha=0.8, s=30, label=None
            )

        # 绘制 KDE 等高线与中心点
        for label in np.unique(labels):
            cluster_points = data[labels == label]
            if cluster_points.shape[0] < 5:
                continue
            color = palette[label]
            sns.kdeplot(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                ax=ax,
                levels=1,
                color=color,
                linewidths=1.0,
                alpha=0.8,
                bw_adjust=0.3
            )
            mean = cluster_points.mean(axis=0)
            ax.scatter(mean[0], mean[1], color=color, edgecolors='black', marker='o', s=120, label=f"Label {label}")

        ax.set_title(titles[i])
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_cluster_visualization(
    X_valid,
    labels_1,
    method='pca',
    cluster_method_name='kmeans',
    n_clusters=5,
    save_path_prefix="../data/MIMIC3/analysis",
    modality='modality',
    other_X=None,
    labels_2=None,
    sample_label=None,
    unify_projection=False,  # 是否拼接降维器
    lambda_align=0.5,
):
    assert method in ['pca', 'tsne'], "method must be 'pca' or 'tsne'"

    # 降维器选择
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    # elif method == 'umap':
    #     reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)

    # 降维逻辑
    if other_X is not None and unify_projection:
        # 拼接训练降维器，统一坐标空间
        concat_X = np.concatenate([X_valid, other_X], axis=0)
        X_all_2d = reducer.fit_transform(concat_X)
        X_2d = X_all_2d[:len(X_valid)]
        other_2d = X_all_2d[len(X_valid):]
    else:
        # 各自独立降维
        X_2d = reducer.fit_transform(X_valid)
        other_2d = reducer.fit_transform(other_X) if other_X is not None else None

    # 创建一张双子图
    fig, axes = plt.subplots(3, 2, figsize=(35, 45), dpi=300)
    axes = axes.flatten()

    # 子图 1：主模态聚类
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels_1, palette='husl', s=60, alpha=0.8, ax=axes[0])
    axes[0].set_title(f"{cluster_method_name.upper()} on {modality}")
    axes[0].set_xlabel(f"{method.upper()} 1")
    axes[0].set_ylabel(f"{method.upper()} 2")
    axes[0].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 子图 2：主模态聚类
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels_2, palette='husl', s=60, alpha=0.8, ax=axes[2])
    axes[2].set_title(f"{cluster_method_name.upper()} on {modality}")
    axes[2].set_xlabel(f"{method.upper()} 1")
    axes[2].set_ylabel(f"{method.upper()} 2")
    axes[2].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=sample_label, palette='husl', s=60, alpha=0.8, ax=axes[4])
    axes[4].set_title(f"{cluster_method_name.upper()} on {modality} LABEL")
    axes[4].set_xlabel(f"{method.upper()} 1")
    axes[4].set_ylabel(f"{method.upper()} 2")
    axes[4].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 子图 2：other_hs 的同标签投影（不聚类）
    if other_2d is not None:
        sns.scatterplot(x=other_2d[:, 0], y=other_2d[:, 1], hue=labels_1, palette='husl', s=60, alpha=0.8, ax=axes[1])
        axes[1].set_title(f"{cluster_method_name.upper()} Labels on Other Modality")
        axes[1].set_xlabel(f"{method.upper()} 1")
        axes[1].set_ylabel(f"{method.upper()} 2")
        axes[1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

        sns.scatterplot(x=other_2d[:, 0], y=other_2d[:, 1], hue=labels_2, palette='husl', s=60, alpha=0.8, ax=axes[3])
        axes[3].set_title(f"{cluster_method_name.upper()} Labels on Other Modality")
        axes[3].set_xlabel(f"{method.upper()} 1")
        axes[3].set_ylabel(f"{method.upper()} 2")
        axes[3].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

        sns.scatterplot(x=other_2d[:, 0], y=other_2d[:, 1], hue=sample_label, palette='husl', s=60, alpha=0.8,
                        ax=axes[5])
        axes[5].set_title(f"{cluster_method_name.upper()} Labels on Other Modality LABEL")
        axes[5].set_xlabel(f"{method.upper()} 1")
        axes[5].set_ylabel(f"{method.upper()} 2")
        axes[5].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    suffix = "unified" if unify_projection else "separate"
    save_path = f"{save_path_prefix}/{cluster_method_name}_nCluster_{n_clusters}_{method}_dual_projection_{suffix}_lambda_align_{lambda_align}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    if other_2d is not None:
        save_mean_grid_path = f"{save_path_prefix}/{cluster_method_name}_nCluster_{n_clusters}_{method}_dual_projection_{suffix}_lambda_align_{lambda_align}_mean_grid.png"
        plot_cluster_mean_kde_grid(X_2d, other_2d, labels_1, labels_2, method, cluster_method_name, modality,
                                   save_mean_grid_path)


def evaluate_and_print_all_metrics(X1, X2, label1, label2):
    """
    逐项输出四种组合下的三类聚类评价指标：CH、Silhouette、DB。
    共输出 12 个指标值。
    """

    # 转换为 numpy，如果是 torch.tensor
    import numpy as np
    if hasattr(X1, 'detach'):
        X1 = X1.detach().cpu().numpy()
    if hasattr(X2, 'detach'):
        X2 = X2.detach().cpu().numpy()

    pairs = [
        ('L1 on X1', X1, label1),
        ('L1 on X2', X2, label1),
        ('L2 on X2', X2, label2),
        ('L2 on X1', X1, label2),
    ]

    for name, X, label in pairs:
        print(f"\n==== {name} ====")

        try:
            sil = silhouette_score(X, label)
            print(f"Silhouette Score       : {sil:.4f}")
        except Exception as e:
            print(f"Silhouette Score       : Error ({e})")

        try:
            ch = calinski_harabasz_score(X, label)
            print(f"Calinski-Harabasz Score: {ch:.2f}")
        except Exception as e:
            print(f"Calinski-Harabasz Score: Error ({e})")

        try:
            db = davies_bouldin_score(X, label)
            print(f"Davies-Bouldin Score   : {db:.4f}")
        except Exception as e:
            print(f"Davies-Bouldin Score   : Error ({e})")


def cluster(X, other_X, sample_label, method='co-kmeans', n_clusters=20, modality="ts_hs", lambda_align=0.5, save_path=None):
    """
    :param X: numpy array, shape (N, D)
    :param method: one of ['kmeans', 'gmm', 'dbscan', 'agglomerative', 'spectral']
    :param n_clusters: int, for methods that require it
    """
    print(f"\nUsing clustering method: {method}")


    if method == 'co-kmeans':
        label_1, label_2 = co_training_kmeans(X, other_X, n_clusters, lambda_align=lambda_align)
    elif method == 'co-gmm':
        label_1, label_2 = co_training_gmm(X, other_X, n_clusters, lambda_align=lambda_align)

    evaluate_and_print_all_metrics(X, other_X, label_1, label_2)

    # 三种降维方式都画图
    for dim_method in ['tsne']:  # ['pca', 'tsne', 'umap']
        plot_cluster_visualization(
            X,
            labels_1=label_1,
            method=dim_method,
            cluster_method_name=method,
            n_clusters=n_clusters,
            modality=modality,  # e.g. 'note' or 'ts'
            other_X=other_X,
            labels_2=label_2,
            sample_label=sample_label,
            lambda_align=lambda_align,
            save_path_prefix=save_path
        )

    mean_X_by_label_1 = compute_class_means(X, label_1, n_clusters)
    mean_otherX_by_label_1 = compute_class_means(other_X, label_1, n_clusters)
    mean_X_by_label_2 = compute_class_means(X, label_2, n_clusters)
    mean_otherX_by_label_2 = compute_class_means(other_X, label_2, n_clusters)

    mean_X_by_label_1 = torch.tensor(mean_X_by_label_1, dtype=torch.float32)
    mean_otherX_by_label_1 = torch.tensor(mean_otherX_by_label_1, dtype=torch.float32)
    mean_X_by_label_2 = torch.tensor(mean_X_by_label_2, dtype=torch.float32)
    mean_otherX_by_label_2 = torch.tensor(mean_otherX_by_label_2, dtype=torch.float32)

    return mean_X_by_label_1, mean_otherX_by_label_1, mean_X_by_label_2, mean_otherX_by_label_2


def read_hidden_state(filepath, modality, another_modality):
    data = read_pkl(filepath)

    hidden_state_list = []
    other_hs_list = []
    label_list = []
    for item in data:
        now_hs = item[modality]
        hidden_state_list.append(now_hs)
        other_hs_list.append(item[another_modality])
        label_list.append(item['label'])
        # print(item['label'])
    # print(label_list)

    return torch.stack(hidden_state_list), torch.stack(other_hs_list), label_list



if __name__ == "__main__":

    # co-kmeans, lambda_align0.8 n_clusters14
    np.random.seed(0)
    modal = "ts"  # ts note
    method = 'co-kmeans'
    lambda_align = 0.8
    n_clusters = 6
    if modal == "ts":
        modality = "ts_hs"
        another_modality = "note_hs"
        other = "note"
    elif modal == "note":
        modality = "note_hs"
        another_modality = "ts_hs"
        other = "ts"
    model_name = "128_256"
    save_path = f"../data/MIMIC3/analysis/{model_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    hidden_state, other_hs, label = read_hidden_state(filepath=f"../data/MIMIC3/analysis/merged_analysis_{model_name}.pkl", modality=modality, another_modality=another_modality)

    print(hidden_state.shape)
    print(other_hs.shape)

    for n_clusters in [11]:  # range(7, 15)
        print(f"n_clusters {n_clusters}...")
        for lambda_align in range(0, 1):
            lambda_align = float(lambda_align) / 10
            print(f"lambda align: {lambda_align}")

            mean_X_by_label_1, mean_otherX_by_label_1, mean_X_by_label_2, mean_otherX_by_label_2 = cluster(hidden_state, other_hs, label, method=method, n_clusters=n_clusters, modality=modality, lambda_align=lambda_align, save_path=save_path)

            rag = {
                f"{modal}_2_{other}_from": mean_X_by_label_1,
                f"{modal}_2_{other}_to": mean_otherX_by_label_1,
                f"{other}_2_{modal}_from": mean_otherX_by_label_2,
                f"{other}_2_{modal}_to": mean_X_by_label_2,
            }

            save_to_pkl(rag, f"{save_path}/rag_{model_name}_{method}_{n_clusters}_{lambda_align}.pkl")
