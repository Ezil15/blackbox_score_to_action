import numpy as np
from average_method import Average_ActionGiver
from ml_method import ML_ActionGiver
import csv

#Читаем данные из csv файла
with open('table.csv', newline='') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data,None)
    data = list(data)

av = Average_ActionGiver(data)


print("Параметры для действий Поиска по среднему:",av.actions_average)
print("Score = 15, Action =",av.get_action(15))
print("Score = 60, Action =",av.get_action(60))
print("Score = 90, Action =",av.get_action(90))
print("Эффективность Поиска по среднему:",av.get_effectiveness(data))

ml = ML_ActionGiver(data)
print("Score = 15, Action=",ml.get_action(15))
print("Score = 60, Action=",ml.get_action(60))
print("Score = 90, Action=",ml.get_action(90))
print("Эффективность машинного обучения:",ml.get_effectiveness(data))