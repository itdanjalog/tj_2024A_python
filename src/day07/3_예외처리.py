# day07 > 3_예외처리.py
# 예외 처리 란? 프로그램 실행 도중에 발생하는 오류를 다른 실행문으로 처리해주는 역할

list = [ 1 , 2 , 3 ]
# list[3] # 예외발생 : 존재하지 않는 인덱스 이므로 예외 발생
# [1] 예외 처리 방법1
try:
    # 예외가 발생 할것 같은 코드들
    list[3]
except :
    # 만약에 예외가 발생 했을때 실행되는 코드들
    print('존재하지 않는 인덱스 입니다.')
# [2] 예외 처리 방법2
try :
    list[3]
except IndexError : # 특정 예외를 처리 할때는 예외클래스명 작성한다.
    print('존재하지 않는 인덱스 입니다. ')
# [3] 예외 처리 방법3
try :
    list[3]
except IndexError as e : # 특정 예외 가 발생한 사유를 보고싶을때 as 예외변수명 작성한다. # list index out of range
    print( e )

# [4] finally , 예외 발생 여부와 상관없이 무조건 실행되는 구역
try:
    list[3]
except IndexError as e :
    print( e )
finally:
    print('예외 여부와 상관없이 실행')

# [5] 다중 except , 다중 except 중 1번 또는 0번 실행된다.
try :
    list[3]     # try 안에서 예외가 발생하면 아래코드는 실행되지 않고 except 이동 하므로 예외처리는 1번 또는 0번 실행한다.
    print( 4/0 )
    int( "a" )
except ZeroDivisionError as e : print( e )  # ZeroDivisionError 예외가 발생했을때 실행
except IndexError as e : print( e )         # IndexError 예외가 발생했을때 실행
except Exception as e : print( e )          # 그외 모든 예외가 발생했을때 실행 # 다중예외 에서는 마지막에 사용한다.














