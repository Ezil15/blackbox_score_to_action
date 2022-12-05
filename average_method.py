import numpy as np
import csv


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

class AverageActionGiver:
    '''Класс реализующий алгоритм поиска необходимого действия по переданному score.
    Принцип поиска необходимых параметров для каждого действия - Поиск среднего значения из переданного датасета'''
    def __init__(self,data) -> None:
        self.actions_average = {}
        self.data = data

        #Сбор всех возможных действий Actions из переданного датасета data
        actions = np.unique(np.array(data)[:, 2]).astype(str)

        for action in actions:
            #Подсчет среднего параметра для каждого действия Action
            av = self.get_average_for_action(action)
            self.actions_average[action] = av
        
        #Сортировка всех действий action и их средних значений параметров в порядке "по убыванию"
        self.actions_average = {k: v for k, v in sorted(self.actions_average.items(), key=lambda item: item[1], reverse=True)}
        
    def get_average_for_action(self, action: str) -> float:
        '''Функция поиска среднего значения из переданной таблицы данных для определенного действия action'''
        a = np.asarray(self.data)

        #Берем все записи где действие равно переданному action 
        all_with_action = a[np.in1d(a[:, 2], np.asarray([action]))][:, 1].astype(float)
        
        #Избавляемся от аномалий
        all_with_action = reject_outliers(all_with_action,2)
        all_with_action = reject_outliers(all_with_action,1) #По правде... не то чтобы повторное взятие медианы было нужно, но она дает большую эффективность

        #Подсчет среднего
        av = sum(all_with_action)/len(all_with_action)

        return av

    def get_action(self, score: float) -> str:
        '''Функция возвращающее предполагаемое действие исходя из переданного score'''

        params = list(self.actions_average.values())
        actions = list(self.actions_average.keys())

        #Проходимся по всем параметрам каждого действия от большего к меньшему
        for param_id in range(len(params)):
            #Если переданный score больше параметра определенного action...
            if score > params[param_id]:
                #...то возвращаем action для переданного score
                return actions[param_id]

        #В случае если не одно из действий не подошло под параметры, то возвращаем действие с наименьшим параметром
        return actions[-1]

    def get_effectiveness(self,data: list) -> float:
        '''Функция высчитывающая эффективность алгоритма к переданной таблице данных'''
        successed_rows = [row for row in data if row[2] == self.get_action(float(row[1]))]
        return  len(successed_rows)/len(data)

