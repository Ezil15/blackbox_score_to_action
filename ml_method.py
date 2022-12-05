import numpy as np
from pandas import read_csv as read
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

class ML_ActionGiver:
    '''Класс реализующий алгоритм поиска необходимого действия по переданному score.
    В класс передается датасет, на котором алгоритм обучается выдавать предполагаемое значение на основе алгоритма Random Forest'''
    def __init__(self, data) -> None:
        self.data = np.array(data)

        #Берем список всех возможных действий actions
        actions = np.unique(np.array(data)[:, 2]).astype(str)
        
        #Создаем пустой список для будущего пересобранного дата сета без аномалий
        new_data_set = np.empty((0,2))
        for action in actions:
            #Берем все значения score с определенным действием action
            all_with_action = self.data[np.in1d(self.data[:, 2], np.asarray([action]))][:, 1].astype(float)
            all_with_action = reject_outliers(all_with_action,2)
            all_with_action = reject_outliers(all_with_action,1)
            new_array = []
            for value in all_with_action:
                new_array.append([value,action])
            new_data_set = np.vstack([new_data_set, np.array(new_array)])
        self.data = new_data_set
        #Вычлением необходимые столбцы, где X - score, Y = action
        X = np.array(self.data)[:, 0]
        y = np.array(self.data)[:, 1]

        '''Разбиваем переданный датасет на данные на которых будем оценивать эффективность
        и параметры на котором будет проводиться обучение
        '''
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.6, random_state=1)

        #Превращаем одномерный массив в двумерный
        X_train = X_train.reshape(-1, 1)
        self.X_test = self.X_test.reshape(-1, 1)

        #Создаем экземпляр классификатора, который будет обучаться на нашем датасете
        self.clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

        #Запускаем обучение
        self.clf.fit(X_train, y_train)

    def get_action(self,score: float) -> float:
        '''Функция возвращающее предполагаемое действие исходя из переданного score'''
        return self.clf.predict([[score]])[0]

    def get_effectiveness(self,data: list) -> float:
        '''Функция высчитывающая эффективность алгоритма к переданной таблице данных'''
        successed_rows = [row for row in data if row[2] == self.get_action(float(row[1]))]
        return  len(successed_rows)/len(data)

    def get_effectiveness_on_dataset(self) -> float:
        '''Функция высчитывающая эффективность алгоритма основываясь 
        на тестовой выборке из переданного датасета'''
        return self.clf.score(self.X_test, self.y_test)

