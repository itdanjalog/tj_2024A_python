# day24 > 2_군집분석.py

# [1] 데이터 수집
import pandas as pd
retail_df = pd.read_excel('Online Retail.xlsx')
print( retail_df.head() )

# [2] 데이터 준비 및 탐색
# 1.정제하기