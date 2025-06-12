"""自适应分布 全部数据"""
# import torch
# from torch.utils.data import DataLoader
# import tqdm


# def collect_feature(data_loader: DataLoader, feature_extractor: torch.nn.Module,
#                     device: torch.device, max_num_features=None) -> torch.Tensor:
#     """
#     Fetch data from `data_loader`, and then use `feature_extractor` to collect features.
#
#     Args:
#         data_loader (torch.utils.data.DataLoader): Data loader.
#         feature_extractor (torch.nn.Module): A feature extractor.
#         device (torch.device): The device (CPU or GPU) to run the model on.
#         max_num_features (int): The max number of features to return (optional).
#
#     Returns:
#         torch.Tensor: Collected features in shape (min(len(data_loader), max_num_features), F).
#     """
#     feature_extractor.eval()
#     all_features = []
#     with torch.no_grad():
#         for i, (images, target, _) in enumerate(tqdm.tqdm(data_loader)):
#             images = images.to(device)
#             target = target.to(device)
#             feature = feature_extractor(images, target)
#             # feature = feature_extractor.backbone(images)
#             logits_all, boxes_all = feature['logits_all'], feature['boxes_all']
#             feature = logits_all[-1].cpu()
#             all_features.append(feature)
#             if max_num_features is not None and i >= max_num_features:
#                 break
#     return torch.cat(all_features, dim=0)

"""自适应分布 随机选择数据"""
#
import torch
from torch.utils.data import DataLoader
import tqdm
def collect_feature(data_loader: DataLoader, feature_extractor: torch.nn.Module,
                    device: torch.device, max_images=1000) -> torch.Tensor:

    feature_extractor.eval()
    all_features = []
    total_images = 0  # 已处理的图像数量

    with torch.no_grad():
        for i, (images, target, _) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(device)

            # 如果 target 是字典，将其值移动到设备
            # if isinstance(target, dict):
            #     target = {key: value.to(device) for key, value in target.items()}
            # else:
            target = target.to(device)

            # 提取特征
            feature = feature_extractor(images, target)
            logits_all, boxes_all = feature['logits_all'], feature['boxes_all']
            feature = logits_all[-1].cpu()  # 提取最后一层的 logits

            all_features.append(feature)
            total_images += images.size(0)

            # 达到最大图像限制时退出
            if total_images >= max_images:
                break

    # 如果提取的图像多于 max_images，则截取前 max_images
    all_features = torch.cat(all_features, dim=0)

    if all_features.size(0) > max_images:
        all_features = all_features[:max_images]

    return all_features



"""最初分布"""
# #
# import random
# import torch
# from torch.utils.data import DataLoader, Subset
# import tqdm
#
#
# def collect_feature(data_loader: DataLoader, feature_extractor: torch.nn.Module,
#                     device: torch.device, sample_size=1000) -> torch.Tensor:
#     """
#     从数据集中随机选择指定数量的图像，并提取其特征。
#
#     Args:
#         data_loader (torch.utils.data.DataLoader): 数据加载器。
#         device (torch.device): 计算设备 (CPU 或 GPU)。
#         sample_size (int): 要随机选择的图像数量。
#
#     Returns:
#         torch.Tensor: 提取的特征，形状为 (sample_size, F)。
#     """
#     # 获取数据集的总大小
#     dataset = data_loader.dataset
#     total_samples = len(dataset)
#
#     # 确保 sample_size 不超过数据集大小
#     if sample_size > total_samples:
#         raise ValueError(f"Requested sample_size ({sample_size}) exceeds dataset size ({total_samples}).")
#
#     # 随机选择样本索引
#     selected_indices = random.sample(range(total_samples), sample_size)
#
#     # 创建一个包含随机选择的样本的子集
#     subset_loader = DataLoader(Subset(dataset, selected_indices), batch_size=data_loader.batch_size, shuffle=False)
#
#     # 提取特征
#     all_features = []
#     with torch.no_grad():
#         for batch in tqdm.tqdm(subset_loader):
#             if len(batch) == 2:  # 数据加载器返回 (images, target)
#                 images, target = batch
#             elif len(batch) == 3:  # 数据加载器返回 (images, target, meta)
#                 images, target, _ = batch
#             else:
#                 raise ValueError(f"Unexpected data format with {len(batch)} elements")
#
#             images = images.to(device)
#             features = images.view(images.size(0), -1).cpu()  # 展平每张图像
#             all_features.append(features)
#
#     return torch.cat(all_features, dim=0)