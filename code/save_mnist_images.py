import torch
import torchvision
import os
from PIL import Image
import numpy as np

def save_mnist_images(dataset, save_dir, prefix):
    """
    将MNIST数据集保存为单独的图片文件
    dataset: MNIST数据集
    save_dir: 保存目录
    prefix: 文件名前缀（'train' 或 'test'）
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 为每个数字类别创建子目录
    for i in range(10):
        digit_dir = os.path.join(save_dir, str(i))
        if not os.path.exists(digit_dir):
            os.makedirs(digit_dir)
    
    # 遍历数据集并保存图片
    for idx, (image, label) in enumerate(dataset):
        # 将图片数据转换为numpy数组
        image_np = image.numpy()[0] * 255  # 取消归一化，转回0-255范围
        image_np = image_np.astype(np.uint8)
        
        # 创建PIL图片
        img = Image.fromarray(image_np)
        
        # 构建保存路径：/保存目录/数字类别/前缀_索引.png
        save_path = os.path.join(
            save_dir, 
            str(label),
            f"{prefix}_{idx}.png"
        )
        
        # 保存图片
        img.save(save_path)
        
        # 打印进度
        if (idx + 1) % 1000 == 0:
            print(f"已保存 {idx + 1} 张图片...")

def main():
    # 数据预处理（与训练代码相同）
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    
    # 加载训练集和测试集
    print("正在加载训练集...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    print("正在加载测试集...")
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 设置保存目录
    save_dir = './mnist_images'
    
    # 保存训练集图片
    print("正在保存训练集图片...")
    save_mnist_images(train_dataset, save_dir, 'train')
    
    # 保存测试集图片
    print("正在保存测试集图片...")
    save_mnist_images(test_dataset, save_dir, 'test')
    
    print("所有图片保存完成！")
    print(f"图片保存在: {os.path.abspath(save_dir)}")

if __name__ == '__main__':
    main() 