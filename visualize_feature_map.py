# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 00:58:20 2022

@author: Chen'hong'yu
"""

import os
import matplotlib.pyplot as plt
import torchvision
import torch
from PIL import Image
# import utils
from build_modules import *
from main import *
def normalize_feature_map(feature_map):
    """归一化特征图到 [0, 1] 范围"""
    min_val = feature_map.min()
    max_val = feature_map.max()
    normalized = (feature_map - min_val) / (max_val - min_val + 1e-5)  # 加上小值防止除零
    return normalized

def get_row_col(num_pic):
    """计算显示特征图的最优网格大小。"""
    row = int(num_pic ** 0.5)
    col = row if num_pic == row * row else row + 1
    return row, col


def visualize_feature_map(feature_map , save_dir, img_name, selected_channels=None):
    """可视化并保存特征图，支持展示多个通道。"""
    # 将特征图转为 CPU，并转换为 NumPy 数组
    feature_map = feature_map.detach().cpu().numpy()

    # 如果未指定特定通道，则默认展示所有通道
    if selected_channels is None:
        selected_channels = list(range(feature_map.shape[1]))

    # 检查所选通道是否在特征图范围内
    for ch in selected_channels:
        if ch >= feature_map.shape[1]:
            raise ValueError(f"通道索引 {ch} 超出范围，最多可选择 {feature_map.shape[1] - 1} 通道")

    num_channels = len(selected_channels)
    rows = int(num_channels ** 0.5)
    cols = (num_channels // rows) + (num_channels % rows > 0)
    save_path = os.path.join(save_dir, img_name)  # 保存路径
    plt.figure(figsize=(8,6))

    # for i, ch in enumerate(selected_channels):
    #     plt.subplot(rows, cols, i + 1)
    #     plt.imshow(feature_map[0, ch, :, :], cmap="viridis")# viridis cubehelix" inferno
    #     # plt.title(f"Channel {ch + 1}")
    #     plt.axis('off')
    #     # 保存图像
    # plt.savefig(os.path.join("./ch_vis/bg/", img_name"), bbox_inches='tight', pad_inches=0)
    # plt.show()
    '''viridis 适合展示去雨后图像中的亮度变化和对比，特别是强调从低频到高频细节的过渡。
       inferno 当需要展示图像中的高频细节或需要强调图像的某些特定区域时，使用此映射效果较好
       cubehelix 适合展示图像中的层次感和细节复杂度，尤其在视觉上比较清晰且区分度高的场景。
    '''
    # 将所有通道特征图叠加
    feature_map_sum = feature_map.sum(axis=1)
    plt.imshow(feature_map_sum[0, :, :], cmap="jet") # viridis cubehelix" inferno
    # plt.title("Summed Channels")


    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0,dpi=300)
    plt.close()  # 关闭图像以释放内存
    # plt.show()




def load_and_process_image(image_path, size=(800, 800)):
    """加载并预处理图像。"""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # 获取原始尺寸 (宽度, 高度)
    if img.size != size:
        img = img.resize(size)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),

    ])
    return transform(img).unsqueeze(0),original_size


# def main():
#     img_dir = r"F:\Deraining\our_try\vis_add_Images\image/"
#     save_dir = r"F:\Deraining\our_try\vis_add_Images\ablation_study/EGF/"  # 保存图像的文件夹
#     model_path = r'F:\Deraining\our_try\checkpoints\models/TDA_200L/model_best.pth'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     device = torch.device(args.device)
#     model = build_model(args, device)
#     utils.load_checkpoint(model, model_path)
#     model.eval()
#
#     for img_name in os.listdir(img_dir):
#         img_path = os.path.join(img_dir, img_name)
#         img_tensor,original_size = load_and_process_image(img_path)
#         img_tensor = img_tensor.cuda()
#         with torch.no_grad():
#             conv_img = model(img_tensor)
#
#
#         if isinstance(conv_img, (list, tuple)):
#             conv_img = torch.nn.functional.interpolate(conv_img[-1], size=original_size[::-1], mode='bilinear', align_corners=False)
#         normalized_conv_img = normalize_feature_map(conv_img)
#
#         # 选择特定通道
#         visualize_feature_map(normalized_conv_img, save_dir, img_name, selected_channels=None)

#
# if __name__ == "__main__":
#     main()
