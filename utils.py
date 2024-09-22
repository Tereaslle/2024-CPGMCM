import pandas as pd


def read_excel_save_to_csv(file_path: str, save_name: int, sheet_name=None) -> None:
    """
    读取excel文件并保存为csv文件，下次读取csv文件可以加快读取速度
    :param file_path: excel文件地址
    :param save_name: 需要保存的csv文件名，保存在当前同级目录下
    :return: None
    """
    # pandas.read_excel('file path')读取Excel文件
    # pandas.read_csv('file path')读取csv文件
    df = pd.read_excel(file_path, sheet_name)
    # 保存成csv文件,同时不保存索引
    df.to_csv(save_name, index=False)


# if __name__ == '__main__':
#     read_excel_save_to_csv("附件一（训练集）.xlsx", 'appendix_1_m1.csv', "材料1")
#     read_excel_save_to_csv("附件一（训练集）.xlsx", 'appendix_1_m2.csv', "材料2")
#     read_excel_save_to_csv("附件一（训练集）.xlsx", 'appendix_1_m3.csv', "材料3")
#     read_excel_save_to_csv("附件一（训练集）.xlsx", 'appendix_1_m4.csv', "材料4")
