
from scipy import stats

# A팀과 B팀의 경기 득점 데이터 (가상의 데이터)
A_team_scores = [82, 88, 85, 90, 87]
B_team_scores = [78, 79, 85, 83, 80]

# t검정 수행 (독립표본 t검정, 두 집단의 데이터가 독립적일 때 사용)
t_stat, p_value = stats.ttest_ind(A_team_scores, B_team_scores)

# t검정 결과 출력
print(f"t-검정 통계량: {t_stat}")
print(f"p-값: {p_value}")

# p-값 해석 (일반적으로 p-값이 0.05보다 작으면 유의미한 차이가 있다고 판단)
if p_value < 0.05:
    print("두 팀의 평균 득점에 유의미한 차이가 있습니다.")
else:
    print("두 팀의 평균 득점에 유의미한 차이가 없습니다.")



