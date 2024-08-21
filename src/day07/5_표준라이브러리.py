import json
with open('myinfo.json' , encoding='utf-8' ) as f:
    data = json.load(f)

print( type(data) )
print( data )

data = {'name': '홍길동', 'birth': '0525', 'age': 30}
with open('myinfo2.json', 'w'  ) as f:
    json.dump(data, f , indent=2 ,  ensure_ascii=False )



from faker import Faker
fake = Faker()
data = fake.name()
print( data )

fake = Faker('ko-KR')
data = fake.name()
print( data )

data = fake.address()
print( data )

test_data = [(fake.name(), fake.address()) for i in range(30)]
print( test_data )











