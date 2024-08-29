import pandas as pd
# p.124 연습문제
# 1. 제시된 표의 샘플 데이터
day= [2015,2016,2017,2018,2019,2020] # 행 레이블
data = [[500,450,520,610],  # 데이터
        [690,700,820,900],
        [1100,1030,1200,1380],
        [1500,1650,1700,1850],
        [1990,2020,2300,2420],
        [1020,1600,2200,2550]]
# 2. 샘플 데이터를 판다스 객체 생성
dt_tbl = pd.DataFrame(data, index=day ,columns=['1분기','2분기','3분기','4분기'])
dt_tbl.to_csv("dbtbl.csv", encoding='utf-8', mode='w', index=True)
# 3. 맷플롯립
import matplotlib.pyplot as plt
for line in data:
    plt.plot( ['first', 'second', 'third', 'fourth'] , line)
        # ['first', 'second', 'third', 'fourth'] , line y축 값
plt.legend( day ) # 범례
plt.title('2015~2020 Quarterly sales')
plt.ylabel('sales')
plt.xlabel('Quarters')
plt.show()
