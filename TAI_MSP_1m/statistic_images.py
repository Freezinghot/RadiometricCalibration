# -*- coding: utf-8 -*-
# @File  : statistic_images.py
# @Author: Freezinghot
# @Date  : 2026/7/24
# @Desc  :
import numpy as np
from PIL import Image
import os
import csv
from pathlib import Path
from typing import List, Union
# import pandas as pd
import glob
import argparse
import sys


def calculate_column_means(image_paths: List[Union[str, Path]]) -> np.ndarray:
    """
    计算多个TIF图像在列方向的均值

    Args:
        image_paths: TIF文件路径列表

    Returns:
        np.ndarray: 列方向均值数组，长度等于图像的列数

    Raises:
        ValueError: 如果图像列数不一致或没有图像
    """
    if not image_paths:
        raise ValueError("图像列表为空")

    # 存储所有图像数据
    all_images = []
    total_rows = 0
    image_width = None

    # 第一遍：读取所有图像并验证尺寸
    print(f"正在读取 {len(image_paths)} 个图像文件...")

    for idx, img_path in enumerate(image_paths, 1):
        try:
            # 读取图像
            img = Image.open(img_path)
            # 转换为numpy数组
            img_array = np.array(img, dtype=np.float64)

            # 如果是灰度图，保持2D；如果是彩色图，转换为灰度
            if len(img_array.shape) == 3:
                # 转为灰度（加权平均）
                img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

            # 验证列数是否一致
            if image_width is None:
                image_width = img_array.shape[1]
            elif img_array.shape[1] != image_width:
                raise ValueError(
                    f"图像 {img_path} 的列数 ({img_array.shape[1]}) "
                    f"与之前的列数 ({image_width}) 不一致"
                )

            all_images.append(img_array)
            total_rows += img_array.shape[0]
            print(f"  [{idx}/{len(image_paths)}] 已读取: {os.path.basename(img_path)} "
                  f"(行数: {img_array.shape[0]}, 列数: {img_array.shape[1]})")

        except Exception as e:
            print(f"读取图像 {img_path} 时出错: {e}")
            continue

    if not all_images:
        raise ValueError("没有成功读取任何图像")

    # 计算列方向均值
    print(f"\n总行数: {total_rows}, 列数: {image_width}")
    print("正在计算列方向均值...")

    # 方法1：逐图像累积求和（内存友好）
    column_sum = np.zeros(image_width, dtype=np.float64)
    total_rows_actual = 0

    for img_array in all_images:
        # 列方向求和（对每一列求和）
        column_sum += np.sum(img_array, axis=0)
        total_rows_actual += img_array.shape[0]

    # 计算均值
    column_means = column_sum / total_rows_actual

    print(f"计算完成，列方向均值数组长度: {len(column_means)}")

    return column_means


def save_means_to_csv(means: np.ndarray, output_path: Union[str, Path],
                      include_index: bool = True, add_header: bool = True):
    """
    将列方向均值保存为CSV文件

    Args:
        means: 列方向均值数组
        output_path: 输出CSV文件路径
        include_index: 是否包含列索引列
        add_header: 是否添加表头
    """
    output_path = Path(output_path)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n正在保存结果到: {output_path}")

    # 准备数据
    data = []
    if add_header:
        if include_index:
            header = ['Column_Index', 'Mean_Value']
        else:
            header = ['Mean_Value']
        data.append(header)

    # 添加数据行
    for idx, value in enumerate(means):
        if include_index:
            data.append([idx, value])
        else:
            data.append([value])

    # 写入CSV文件
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        print(f"CSV文件已成功保存: {output_path}")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
        raise


def save_means_to_csv_pandas(means: np.ndarray, output_path: Union[str, Path]):
    """
    使用pandas保存列方向均值为CSV文件（备选方法）

    Args:
        means: 列方向均值数组
        output_path: 输出CSV文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建DataFrame
    df = pd.DataFrame({
        'Column_Index': range(len(means)),
        'Mean_Value': means
    })

    # 保存为CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"CSV文件已成功保存: {output_path}")


def process_tif_files(image_paths: List[Union[str, Path]],
                      output_csv: Union[str, Path],
                      use_pandas: bool = False):
    """
    完整的处理流程：读取TIF文件，计算列方向均值，保存为CSV

    Args:
        image_paths: TIF文件路径列表
        output_csv: 输出CSV文件路径
        use_pandas: 是否使用pandas保存（默认使用csv模块）
    """
    print("=" * 60)
    print("开始处理TIF图像列方向均值统计")
    print("=" * 60)

    # 计算列方向均值
    means = calculate_column_means(image_paths)

    # 保存为CSV
    if use_pandas:
        save_means_to_csv_pandas(means, output_csv)
    else:
        save_means_to_csv(means, output_csv, include_index=True, add_header=True)

    # 显示统计信息
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"统计结果概览:")
    print(f"  - 处理的图像数量: {len(image_paths)}")
    print(f"  - 列方向均值范围: [{means.min():.2f}, {means.max():.2f}]")
    print(f"  - 均值数组长度: {len(means)}")
    print(f"  - 结果已保存到: {output_csv}")
    print("=" * 60)


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description='统计TIF图像列方向均值工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 基本使用
  python statistic_images.py -i "E:\\RSDATA\\0723测试\\split_raw" -o "E:\\RSDATA\\0723测试\\statistic\\B1.csv" -p "B1"

  # 使用文件列表
  python statistic_images.py -f filelist.txt -o "output.csv"
        '''
    )

    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        help='输入图像目录路径'
    )

    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='*.tif',
        help='文件名匹配模式，默认为 "*.tif"'
    )

    parser.add_argument(
        '-f', '--file-list',
        type=str,
        help='包含图像文件路径列表的文本文件（每行一个路径）'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出CSV文件路径'
    )

    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='递归搜索子目录'
    )

    args = parser.parse_args()

    # 获取图像文件列表
    tif_files = []

    if args.file_list:
        # 从文件列表读取
        try:
            with open(args.file_list, 'r', encoding='utf-8') as f:
                tif_files = [line.strip() for line in f if line.strip()]
            print(f"从文件列表读取了 {len(tif_files)} 个图像路径")
        except Exception as e:
            print(f"读取文件列表失败: {e}")
            sys.exit(1)
    elif args.input_dir:
        # 从目录搜索
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"错误：目录不存在 {args.input_dir}")
            sys.exit(1)

        if args.recursive:
            tif_files = list(input_dir.glob(f"**/{args.pattern}"))
        else:
            tif_files = list(input_dir.glob(args.pattern))

        print(f"在目录 {args.input_dir} 中找到 {len(tif_files)} 个匹配文件")
    else:
        print("错误：请指定 -i (输入目录) 或 -f (文件列表)")
        parser.print_help()
        sys.exit(1)

    if not tif_files:
        print("错误：没有找到任何TIF文件")
        sys.exit(1)

    # 处理图像
    try:
        process_tif_files(tif_files, args.output)
    except Exception as e:
        print(f"处理过程中出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 如果命令行运行，使用命令行参数
    # if len(sys.argv) > 1:
    main()
    # else:
    #     # 否则使用默认配置（方便调试）
    #     input_dir = r'E:\RSDATA\0723测试\split_raw'
    #     output_csv = r'E:\RSDATA\0723测试\statistic\B1.csv'
    #     label = 'B1'
    #
    #     tif_files = list(Path(input_dir).glob(f"{label}*.tif"))
    #
    #     if tif_files:
    #         process_tif_files(
    #             image_paths=tif_files,
    #             output_csv=output_csv
    #         )
    #     else:
    #         print("没有找到TIF文件，请指定正确的路径")