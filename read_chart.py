import pandas as pd

def transform_csv_to_xlsx():
    df = pd.read_csv('AI_Text_Detection/data/UCAS_AISAD_TEXT-test1.csv', encoding='utf-8')  # 如果 CSV 有中文，建议指定编码
    df.to_excel('AI_Text_Detection/data/UCAS_AISAD_TEXT-test1.xlsx', index=False, engine='openpyxl')  # index=False 不保存行索引


def cc():

    df_a = pd.read_csv('AI_Text_Detection/data/test_data.csv')
    df_b = pd.read_csv('AI_Text_Detection/winner_model/output_res.csv')
    df_a.insert(1, 'result', df_b.iloc[:, 0])  # 列名可自定义
    # 保存修改后的a文件（可选覆盖原文件或另存为新文件）
    df_a.to_csv('AI_Text_Detection/winner_model/result.csv', index=False)

cc()
