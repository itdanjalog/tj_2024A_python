# day17 > 2_텍스트빈도분석.py

# 1. 분석할 텍스트 준비
textData = """Big data refers to the large volume of data – both structured and unstructured – that inundates a business on a day-to-day basis. 
But it’s not the amount of data that’s important. It’s what organizations do with the data that matters. 
Big data can be analyzed for insights that lead to better decisions and strategic business moves."""
# 2. 다양한 전처리
    # (1) 정규화 : 모든 영대소문자를 소문자 변환
textData = textData.lower()
# "문자열".lower() : 소문자 로 변환 함수
# "문자열".upper() : 대문자 로 변환 함수
print( textData )
    # (2) 정규화2 : 구두점( . , ? ! : ; ' " () 등 )과 불필요한 특수문자/기호 제거 # 정규표현식
import re # 정규표현식 # 파이썬 내장 라이브러리
textData = re.sub( r'[^\w\s]' , '' , textData )
    # \w : 문자 혹은 숫자 의미   # \s : 공백(스페이바,탭) 의미  # ^ : 부정 의미
    # [^\w\s] : 문자, 숫자 , 공백 이 아닌 것 찾기
    # re.sub(  r'정규표현식' , '대체할문자' ,  '기존문자열' )
    # 기존문자열에서 정규표현식을 이용한 문자를 찾아 대체 하는 함수
print( textData )
    # (3) 문자열을 토큰(단어)화
words = textData.split(" ") # 띄어쓰기 기준으로 문자 구분
print( words )

# 3. 문자 개수 세기
from collections import Counter # 컬렉션(리스트/딕셔너리/집합)
wordCount = Counter( words ) # 중복된 요소들의 개수를 튜플형식으로 반환 해주는 함수
print( wordCount )  # [ (단어,개수) , (단어,개수) , (단어,개수) , (단어,개수) ]

# 4. 빈도가 높은 상위 n개 만큼 출력 # .most_common( N ) : 상위n개 반환 해주는 함수
print( wordCount.most_common(10) )

# 5. 시각화 : 워드클라우드( )
from wordcloud import WordCloud # 모듈 호출
import matplotlib.pyplot as plt
# (1) 워드클라우드 객체 생성 # WordCloud( width = 가로사이즈 , height = 세로사이즈 , background_color='배경색상'  ).generate( 시각화할문자열 )
wordcloud = WordCloud( width=800 , height=800 , background_color="white" ).generate( textData )
# (2)
plt.imshow( wordcloud ) # 워드클라우드 객체를 맷플롭립 에 적용
plt.axis("off")  # 축 숨기기
plt.show()





