'''
        예를 들어, 다음과 같은 데이터를 수집했다고 가정합니다:

        성별	        스포츠 선호	드라마 선호
        남성	        50명	    30명
        여성	        20명	    40명

귀무 가설 (H₀): 성별과 선호하는 콘텐츠 유형(스포츠, 드라마)은 독립적이다. (성별에 따른 선호도에 차이가 없다)
대립 가설 (H₁): 성별과 선호하는 콘텐츠 유형(스포츠, 드라마)은 독립적이지 않다. (성별에 따른 선호도에 차이가 있다)
'''

import scipy.stats as stats

# 관찰된 데이터 (성별에 따른 스포츠와 드라마 선호도)
# 행: 남성, 여성 / 열: 스포츠 선호, 드라마 선호
observed = [[ 50, 30 ],
            [ 20, 40 ]]

# 카이 검증 수행
chi2, p, dof, expected = stats.chi2_contingency(observed)

# 결과 출력
print(f"Chi2 통계량: {chi2}") # 관찰된 빈도와 기대 빈도 간의 차이를 나타내는 값입니다.
print(f"p-값: {p}") # p-값이 0.05보다 작으면, 귀무 가설을 기각하고 성별과 선호하는 콘텐츠 유형 간에 통계적으로 유의미한 연관성이 있다고 판단할 수 있습니다.
print(f"자유도: {dof}") # 카이 검증에서의 자유도.
print("기대 빈도표:")
print(expected) # 성별과 콘텐츠 선호도가 독립적일 때 예상되는 빈도입니다.

'''
여기서 자유도(dof)는 두 변수(성별과 콘텐츠 선호도) 사이의 독립성을 검토할 때 사용하는 값입니다. 이 경우, 독립성 검정의 자유도 계산 방법은 다음과 같습니다:

행의 개수 (r): 2 (남성, 여성)
열의 개수 (c): 2 (스포츠 선호, 드라마 선호)
자유도는 (r - 1) * (c - 1)로 계산됩니다.

따라서, 자유도는 (2 - 1) * (2 - 1) = 1 * 1 = 1입니다.
'''

'''
    - 용어 
        명목형 변수 : 1.여자 2.남자 ( 구분해주는 분류하는 변수 ) / 숫자에 의미가 없다
        연속형 변수 : 점수화 가능한 변수 / 숫자에 있믜가 있다.
    - 집단 비교 
        남녀 , 실험 전후 , 서울 부산 대전 등등
    - 가설
        남자는 더 소주를 좋아하고 , 여자는 맥주를 더 좋아한다.
    - 카이검증 
        - 독립변수 ( 성별 )
        - 종속변수 ( 주류 선호 )
            - 원인 , 결과가 모두 명목형 변수/자료 
    - p값 0.05 ? 
        - 5% 버리고 보편적인걸 결론 낸다. 일반적으로 통계학에서 사용되는 방법
        - 일반적으로 p-값이 0.05보다 작으면 "귀무 가설을 기각할 충분한 근거가 있다"고 판단합니다.
        
    - 가설 -> 주제 -> 분석방법(카이검증) -> 결론 및 제언 -> 한계점 
        - 성별에 따른 주류 선호 종류의 차이연구
        - ~~에 따른 ~~ 비교 연구 ( 차이 연구 )
'''






