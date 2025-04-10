import pandas as pd
from typing import List
import numpy as np


def read_excel_save_to_csv(file_path: str, save_name: str, sheet_name=None) -> None:
    """
    读取excel文件并保存为csv文件，下次读取csv文件可以加快读取速度
    :param file_path: excel文件地址
    :param save_name: 需要保存的csv文件名，保存在当前同级目录下
    :param sheet_name: excel文件中的表名
    :return: None
    """
    if sheet_name is None:
        sheet_name = 0
    # pandas.read_excel('file path')读取Excel文件
    # pandas.read_csv('file path')读取csv文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # 保存成csv文件,同时不保存索引
    df.to_csv(save_name, index=False)


def merge_appendix1_csv(file_list: List[str], save_name: str) -> None:
    """
    读取多个csv文件并合并数据后保存
    :param file_list: 需要合并csv的文件列表
    :param save_name: 需要保存csv的文件名
    :return: None
    """
    df = None
    column_name_switch = {"温度，oC": "温度",
                          "频率，Hz": "频率",
                          "磁芯损耗，w/m3": "磁芯损耗",
                          "0（磁通密度B，T）": "0",
                          "0（磁通密度，T）": "0"}

    for i, file_path in enumerate(file_list):
        if df is None:
            # 第一次先初始化 df 作为主存储变量
            df = pd.read_csv(file_path)
            df.insert(loc=4, column='材料类别', value=i + 1)
            # 统一列名，材料2的列名 "0（磁通密度，T）" 不统一
            df.rename(columns=column_name_switch, inplace=True)
        else:
            # 后面使用临时变量，合并到主变量中
            temp_df = pd.read_csv(file_path)
            temp_df.insert(loc=4, column='材料类别', value=i + 1)
            # 统一列名
            temp_df.rename(columns=column_name_switch, inplace=True)
            df = pd.concat([df, temp_df], ignore_index=True)
    # # 筛选出包含NaN的行, 查看合并是否产生NaN值
    # nan_rows = df[df.isna().any(axis=1)]
    # print(nan_rows)
    df.to_csv(save_name, index=False)

def curve_fit_SE(f, Bm):
    k_1 = 1.49974538
    alpha_1 = 1.4296344
    beta_1 = 2.47125407
    return k_1 * np.power(f, alpha_1) * np.power(Bm, beta_1)

def temp_SE(f, Bm, T):
    dict = {25: 0, 50: 1, 70: 2, 90: 3}
    i = dict[T]
    k_1 = [3.99479398, 0.74065738, 0.23393301, 0.22204269]
    alpha_1 = [1.34718585, 1.49181571, 1.59104354, 1.60157565]
    beta_1 = [2.30477083, 2.48215826, 2.57619652, 2.65488024]
    return k_1[i] * np.power(f, alpha_1[i]) * np.power(Bm, beta_1[i])

if __name__ == '__main__':
#     read_excel_save_to_csv("附件三（测试集）.xlsx", 'appendix3.csv')
#     read_excel_save_to_csv("附件二（测试集）.xlsx", 'appendix2.csv')
    merge_appendix1_csv(['appendix1_m1.csv', 'appendix1_m2.csv', 'appendix1_m3.csv', 'appendix1_m4.csv'],
                         'appendix1_all.csv')
#     read_excel_save_to_csv("附件一（训练集）.xlsx", 'appendix1_m1.csv', "材料1")
#     read_excel_save_to_csv("附件一（训练集）.xlsx", 'appendix1_m2.csv', "材料2")
#     read_excel_save_to_csv("附件一（训练集）.xlsx", 'appendix1_m3.csv', "材料3")
#     read_excel_save_to_csv("附件一（训练集）.xlsx", 'appendix1_m4.csv', "材料4")
