# import numpy as np
# import torch
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import matplotlib.colors as col
#
# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
#               filename: str, source_color='r', target_color='b'):
#     """
#     可视化来自不同领域的特征，使用 t-SNE 降维。
#
#     Args:
#         source_feature (tensor): 源域特征，形状为 (minibatch, F)
#         target_feature (tensor): 目标域特征，形状为 (minibatch, F)
#         filename (str): 保存 t-SNE 可视化结果的文件名
#         source_color (str): 源特征的颜色，默认是 'r'（红色）
#         target_color (str): 目标特征的颜色，默认是 'b'（蓝色）
#     """
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#
#     # 确保特征是 2D 数组，如果是 3D 数组，展平为 2D
#     if source_feature.ndim > 2:
#         source_feature = source_feature.reshape(source_feature.shape[0], -1)
#     if target_feature.ndim > 2:
#         target_feature = target_feature.reshape(target_feature.shape[0], -1)
#
#     # 合并源域和目标域的特征
#     features = np.concatenate([source_feature, target_feature], axis=0)
#
#     # 使用 t-SNE 进行降维
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
#
#     # 生成领域标签，1 代表源域，0 代表目标域
#     domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
#
#     # 使用 matplotlib 进行可视化
#     plt.figure(figsize=(10, 10))
#     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=2)
#     # plt.title("t-SNE 可视化")
#     plt.savefig(filename)
#     # plt.close()
# import numpy as np
# import torch
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import matplotlib.colors as col
#
# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
#               filename: str, source_color='r', target_color='b'):
#     """
#     可视化来自不同领域的特征，使用 t-SNE 降维。
#
#     Args:
#         source_feature (tensor): 源域特征，形状为 (minibatch, F)。
#         target_feature (tensor): 目标域特征，形状为 (minibatch, F)。
#         filename (str): 保存 t-SNE 可视化结果的文件名。
#         source_color (str): 源特征的颜色，默认是 'r'（红色）。
#         target_color (str): 目标特征的颜色，默认是 'b'（蓝色）。
#     """
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#
#     # 确保特征是 2D 数组，如果是 3D 数组，展平为 2D
#     if source_feature.ndim > 2:
#         source_feature = source_feature.reshape(source_feature.shape[0], -1)
#     if target_feature.ndim > 2:
#         target_feature = target_feature.reshape(target_feature.shape[0], -1)
#
#     # 合并源域和目标域的特征
#     features = np.concatenate([source_feature, target_feature], axis=0)
#
#     # 使用 t-SNE 进行降维
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
#
#     # 生成领域标签，1 代表源域，0 代表目标域
#     domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
#
#     # 使用 matplotlib 进行可视化
#     plt.figure(figsize=(10, 10))
#
#     # 绘制散点图
#     scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains,
#                           cmap=col.ListedColormap([target_color, source_color]), s=8)
#
#     # 添加图例
#     plt.legend(handles=scatter.legend_elements()[0], labels=['Target Domain', 'Source Domain'], loc='best')
#
#     # 调整坐标轴字体大小
#     plt.xlabel('t-SNE Dimension 1', fontsize=18)
#     plt.ylabel('t-SNE Dimension 2', fontsize=18)
#
#     # 调整坐标轴刻度字体大小
#     plt.tick_params(axis='both', which='major', labelsize=16)
#
#     # 保存图像
#     plt.savefig(filename)
#     plt.close()


"""保存坐标"""
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as col


def visualize(
        source_feature: torch.Tensor,
        target_feature: torch.Tensor,
        filename: str,
        source_color='r',
        target_color='b',
        save_coords: bool = True,  
        coord_format: str = "npy"  # 新增保存格式参数 (npy/txt)
):
    """
    可视化来自不同领域的特征，使用 t-SNE 降维，并保存坐标数据。

    Args:
        source_feature (tensor): 源域特征，形状为 (minibatch, F)
        target_feature (tensor): 目标域特征，形状为 (minibatch, F)
        filename (str): 保存 t-SNE 可视化结果的文件名（不含扩展名）
        source_color (str): 源特征的颜色，默认是 'r'（红色）
        target_color (str): 目标特征的颜色，默认是 'b'（蓝色）
        save_coords (bool): 是否保存坐标数据，默认开启
        coord_format (str): 坐标保存格式，可选 'npy' 或 'txt'
    """
    # 特征预处理
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()

    if source_feature.ndim > 2:
        source_feature = source_feature.reshape(source_feature.shape[0], -1)
    if target_feature.ndim > 2:
        target_feature = target_feature.reshape(target_feature.shape[0], -1)

    # 合并特征并执行 t-SNE
    features = np.concatenate([source_feature, target_feature], axis=0)
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # 构建带标签的坐标数据矩阵
    domains = np.concatenate([np.ones(len(source_feature)), np.zeros(len(target_feature))])
    coord_data = np.column_stack([X_tsne, domains])  # 形状：(N_samples, 3)

   
    if save_coords:
        coord_filename = f"{filename}_coords"
        if coord_format == "npy":
            np.save(f"{coord_filename}.npy", coord_data)
        elif coord_format == "txt":
            header = "t-SNE-Dim1\tt-SNE-Dim2\tDomain(1=Source/0=Target)"
            np.savetxt(
                f"{coord_filename}.txt", coord_data,
                fmt="%.6f", delimiter="\t", header=header
            )
        else:
            raise ValueError("coord_format 必须是 'npy' 或 'txt'")


    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=domains,
        cmap=col.ListedColormap([target_color, source_color]),
        s=8,
        alpha=0.6  # 添加透明度提升重叠区域可见性
    )

    # 优化图例和坐标轴
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=['Target Domain', 'Source Domain'],
        loc='upper right',
        fontsize=14,
        markerscale=2  # 放大图例标记
    )
    plt.xlabel('t-SNE Dimension 1', fontsize=16)
    plt.ylabel('t-SNE Dimension 2', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 保存图像
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()