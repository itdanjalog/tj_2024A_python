
# 텍스트 전처리 : 1.정제( 불필요한 기호 제거 ) 2.정규화(

text_data = """
Big data refers to the large volume of data – both structured and unstructured – that inundates a business on a day-to-day basis. 
But it’s not the amount of data that’s important. It’s what organizations do with the data that matters. 
Big data can be analyzed for insights that lead to better decisions and strategic business moves.
"""

import re
ㅈ
# 텍스트를 소문자로 변환
text_data = text_data.lower()

# 구두점과 불필요한 기호 제거
text_data = re.sub(r'[^\w\s]', '', text_data)

print(text_data)

from collections import Counter

# 텍스트를 단어 단위로 분할
words = text_data.split()

# 단어 빈도 계산
word_freq = Counter(words)

# 상위 10개 빈도수 높은 단어 출력
print(word_freq.most_common(10))
########################################################################
# https://youngwoos.github.io/Doit_textmining/01-wordFrequency.html#4
