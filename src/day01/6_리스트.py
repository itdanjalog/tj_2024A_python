# 6_리스트.py
# 여러 자료들을 하나의 자료로 묶음
# 즉) 리스트 안에는 어떠한 자료형도 포함할 수 있다.

# (1) 리스트 타입 형태
# 변수명 = [ 요소1, 요소2, 요소3, 요소4 ]
odd = [ 1 , 3 , 5 , 7 , 9 ]
print( odd )
a = []
print( a )
b = [1,2,3]
print( b )
c = [ 'Life' , 'is', 'too' , 'short']
print( c )
d = [ 1, 2, 'Life' , 'is' ]
print( d )
e = [ 1 , 2 , ['Life','is'] ]
print( e )

# (2) 리스트의 인덱싱과 슬라이싱
studentList = ['유재석','강호동','신동엽','서장훈']
# 인덱싱 : 인덱스 이용한 요소 추출
    # '유재석'[0],'강호동'[1],'신동엽'[2],'서장훈'[3]
    # '유재석'[-4],'강호동'[-3],'신동엽'[-2],'서장훈'[-1]
print( studentList ) # ['유재석', '강호동', '신동엽', '서장훈']
print( studentList[0] ) # 유재석
print( studentList[1] ) # 강호동
print( studentList[3] ) # 서장훈
# print( studentList[4] ) # 예외발생, IndexError: list index out of range
print( studentList[-1] )    # 서장훈
print( studentList[-3] )    # 강호동
print( studentList[-4] )    # 유재석

# 슬라이싱 : 인덱싱 를 이용한 요소들 추출 , [ 시작인덱스(포함0) : 끝인덱스(포함x) : 증감단위 ]
print( studentList[ 0 : 2 ] )   # ['유재석', '강호동'] , 0 ~ 2미만
print( studentList[ 0 : 3 ] )   # ['유재석', '강호동', '신동엽'] , 0~3미만
print( studentList[ : 4 ] )     # ['유재석', '강호동', '신동엽', '서장훈'] , 생략시 0 ~ 4미만
print( studentList[ 2 : ] )     # ['신동엽', '서장훈'] , 2 ~ 생략시마지막인덱스 까지
print( studentList[ : ] )       # ['유재석', '강호동', '신동엽', '서장훈'] , 생략시 0 ~ 마지막 인덱스
print( studentList[ 0 : 4 : 1 ] ) # ['유재석', '강호동', '신동엽', '서장훈']
print( studentList[ 0 : 4 : 2 ] ) # ['유재석', '신동엽']
print( studentList[ -1 : -5 : -1 ] )    # ['서장훈', '신동엽', '강호동', '유재석']
print( studentList[ -1 : -5 : -2 ] )    # ['서장훈', '강호동']

#(3) 리스트 연산
# [리스트1] + [리스트2] = [리스트]
a = [ 1, 2, 3 ]
b = [ 4, 5, 6 ]
c = a + b
print( c ) # [1, 2, 3, 4, 5, 6]
# [리스트1] * 반복수 = [리스트]
c = a * 3
print( c )  # [1, 2, 3, 1, 2, 3, 1, 2, 3]
# len( 리스트 ) : 리스트의 요소 개수 반환
print( len(a) ) # 3

# (4) 리스트내 요소의 값 수정    , 리스트명[수정할인덱스] = 새로운값
a[1] = 4
print( a )  # [1, 4, 3]
# (5) 리스트내 요소 삭제        , del 리스트명[삭제할인덱스]
del a[1]
print( a )  # [1, 3]
del b[ 1: ]        # 슬라이싱 활용한 요소들 삭제  , 1 인덱스 부터 마지막인덱스까지 삭제
print( b )         # [4]

# (6) 리스트 관련 함수들
# 1. .append( 요소값 ) : 리스트의 맨 마지막에 요소값 추가
a = [ 1,2,3 ]
a.append( 4 )
print( a )      # [1, 2, 3, 4]
# 2. .insert( 인덱스 , 요소값 ) : 리스트의 특정 인덱스에 요소값 추가
a.insert( 0 , 6 )   # 0번 인덱스의 6 삽입 , 삽입된 위치 뒤로 한칸씩 밀림.
print( a )      # [6, 1, 2, 3, 4]
# 3. .remove( 삭제할요소값 ) : 리스트의 삭제할 요소값이 존재하면 삭제
a.remove( 1 )
print( a )      # [6, 2, 3, 4]
# a.remove( 1 )   # 존재 하지 않는 값이면 예외 발생 ,ValueError: list.remove(x): x not in list
# 4.    .pop( )          : 리스트의 맨 마지막 요소를 삭제
#       .pop(삭제할인덱스)  : 리스트의 삭제할인덱스 요소를 삭제
a.pop()
print( a )      # [6, 2, 3]
a.pop( 1 )
print( a )      # [6, 3]
# 5. .count( 찾을요소값 )      : 리스트의 해당 찾을요소값이 존재하는 개수 반환
a = [ 1 , 2 , 3 , 1 ]
print( a.count( 1 ) )   # 2 , 리스트내 찾는 요소의 개수가 2 반환
# 6. .index( 찾을요소값 )      : 리스트의 해당 찾을요소값이 존재하는 인덱스 반환
print( a.index( 2 ) )   # 1 , 리스트내 찾는 요소의 위치 1 반환
# 7. .extend( 리스트 )         : 리스트의 해당 리스트를 더 해준다.
b = [ 4 , 5 ]
a.extend( b )
print( a )              # [1, 2, 3, 1, 4, 5]
# 8. .sort()                : 리스트의 요소들을 오름차순 정렬
a.sort()
print( a )              # [1, 1, 2, 3, 4, 5]
# 9. .reverse()              : 리스트의 요소들을 내림차순 정렬 , sort() -> reverse()
a.reverse()
print( a )              # [5, 4, 3, 2, 1, 1]



















































