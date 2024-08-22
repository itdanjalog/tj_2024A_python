# day07 > task9 > region.py
class Region :
    def __init__(self , name , total , male , female , house ):
        self.name = name
        self.total = total
        self.male = male
        self.female = female
        self.house = house
        self.maleRate = round( (self.male / self.total) * 100 , 1 )
        self.femaleRate = round( (self.female / self.total) * 100 , 1 )

    def maleRate(self ):
        return  round( (self.male / self.total) * 100 , 1 )

    def femaleRate(self):
        return round( (self.female / self.total) * 100 , 1 )

    # def info(self):
    #     return f'{self.name:<5}{self.total:>10}{self.male:>10}{self.female:>10}{self.house:>10}{self.maleRate():>10}%{self.femaleRate():>10}%'
