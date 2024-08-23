# day08 > task10 > region.py
class Region :
    def __init__(self, name, total, male, female, house): # 생성자
        self.name = name
        self.total = total
        self.male = male
        self.female = female
        self.house = house
        self.malepercent = round( (male/total ) * 100 , 1 ) # 남성비율
        self.femalepercent = round( (female / total) * 100, 1 ) # 여성비율