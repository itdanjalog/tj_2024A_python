
# 텍스트 전처리 : 1.정제( 불필요한 기호 제거 ) 2.정규화(

text_data = """
Big data refers to the large volume of data – both structured and unstructured – that inundates a business on a day-to-day basis. 
But it’s not the amount of data that’s important. It’s what organizations do with the data that matters. 
Big data can be analyzed for insights that lead to better decisions and strategic business moves.
"""

import re

# 텍스트를 소문자로 변환
text_data = text_data.lower()
# text_data에 저장된 텍스트를 모두 소문자로 변환합니다. 이 단계는 동일한 단어가 대소문자로 인해 중복 계산되는 것을 방지합니다.


# 구두점과 불필요한 기호 제거
text_data = re.sub(r'[^\w\s]', '', text_data)

print(text_data)

from collections import Counter

# 텍스트를 단어 단위로 분할
words = text_data.split()
print( words )

# 단어 빈도 계산
word_freq = Counter(words)

# 상위 10개 빈도수 높은 단어 출력
print(word_freq.most_common(10))
########################################################################
# https://youngwoos.github.io/Doit_textmining/01-wordFrequency.html#4


'''

구두점 (Punctuation Marks):
    마침표 (.): 문장의 끝을 나타냅니다.
    쉼표 (,): 문장에서 일부분을 구분하거나 나열할 때 사용됩니다.
    물음표 (?): 질문을 나타내는 데 사용됩니다.
    느낌표 (!): 감정이나 강한 표현을 나타낼 때 사용됩니다.
    콜론 (:): 설명을 시작하거나 목록을 열기 위해 사용됩니다.
    세미콜론 (;): 서로 밀접하게 관련된 문장이나 문장 부분을 구분할 때 사용됩니다.
    작은따옴표 ('): 인용구를 표시하거나 소유격을 나타낼 때 사용됩니다.
    큰따옴표 ("): 직접 인용문을 나타낼 때 사용됩니다.
    괄호 (()): 추가적인 설명이나 정보를 제공할 때 사용됩니다.
특수 기호 (Special Characters):
    @: 이메일 주소나 소셜 미디어에서 사용자 이름을 언급할 때 사용됩니다.
    #: 해시태그로 사용되거나 번호를 나타낼 때 사용됩니다.
    $: 화폐 단위를 나타낼 때 사용됩니다.
    %: 백분율을 나타낼 때 사용됩니다.
    &: 그리고를 의미하며, 보통 연결할 때 사용됩니다.
    *: 별표 기호로, 강조하거나 중요성을 나타낼 때 사용됩니다.
    - (하이픈): 단어를 연결하거나 구분할 때 사용됩니다.
    _ (언더스코어): 공백 대신 사용되거나 이름을 구분할 때 사용됩니다.
    
'''