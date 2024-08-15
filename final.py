import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import module as ml

df = pd.read_csv(r'C:\\Users\User\PycharmProjects\finalproject\heart-disease.csv')

# Смотрим первые 5 строк
print(df.head())

"""
age: Возраст пациента (в годах)
sex: Пол пациента (1 = мужской, 0 = женский)
cp: Тип боли в груди (1-4)
trestbps: Артериальное давление в состоянии покоя (в мм рт. ст. при поступлении в больницу)
chol: Уровень холестерина в сыворотке в мг/дл
fbs: Уровень сахара в крови натощак > 120 мг/дл (1 = верно; 0 = неверно)
restecg: Результаты электрокардиографии в состоянии покоя (0-2)
thalach: Максимальная достигнутая частота сердечных сокращений
exang: Стенокардия, вызванная физической нагрузкой (1 = да; 0 = нет)
oldpeak: Депрессия ST, вызванная физической нагрузкой, относительно состояния покоя
slope: Максимальная нагрузка
ca: Кол-во сосудов, окрашенных флурозопией
thal: Результат таллийного стресс-теста
"""

# Проверка пустых значений
print(df.isnull().sum())

# Проверка типов данных
print(df.dtypes)

# Описание датасета
print(df.describe().T)
print('-' * 110)
# Визуализация данных
sns.pairplot(df, hue='target')
plt.show()

# Определение числовых и категориальных признаков
num = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Определение целевой переменной
y = df['target']


# Визуализация данных
sns.countplot(x=y)
plt.show()

fig, ax = plt.subplots(ncols=5, figsize=(13, 5))
for i, j in enumerate(num):
    sns.histplot(df, x=j, ax=ax[i])
    ax[i].set_ylabel("")
    ax[i].set_xlabel("")
    ax[i].set_title(j)
plt.show()

sns.pairplot(df, vars=num, hue='target')
plt.show()

fig2, ax2 = plt.subplots(ncols=8, figsize=(15, 4))
for i, j in enumerate(cat):
    sns.countplot(df, x=j, ax=ax2[i])
    ax2[i].set_ylabel("")
    ax2[i].set_xlabel("")
    ax2[i].set_title(j)
plt.tight_layout()
plt.show()

# Визуализация корреляционной матрицы
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, linewidth=0.5)
plt.title('Корреляционная матрица')
plt.show()

# Предобработка данных
X, preprocessor = ml.preprocessor(df, num, cat)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=60)

# Определение моделей
models = {
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
    'ExtraTrees': ExtraTreesClassifier(),
    'KNeighbors': KNeighborsClassifier(),
    'SVM': SVC(),
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier()
}

# Оценка моделей, вывод отчета о классификации, вывод матрицы ошибок
res = ml.report(models, X_train, X_test, y_train, y_test)

# Вывод лучшей модели
ml.best_model(res)

# Определение гиперпараметров
param_gr = {
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 1, 10],
    },
    'ExtraTrees': {
        'n_estimators': [100, 150, 300],
        'max_depth': [2, 4, 8],
        'min_samples_split': [2, 3, 7],
        'min_samples_leaf': [1, 2, 4]
    },
    'KNeighbors': {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 4]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf'],
    },
    'LogisticRegression': {
        'tol': [0.00001, 0.0001, 0.001],
        'fit_intercept': [True, False],
        'C': [0.1, 1, 10, 100]
    },
    'RandomForest': {
        'n_estimators': [100, 175, 250],
        'max_depth': [2, 4, 8],
        'min_samples_split': [2, 3, 7],
        'min_samples_leaf': [1, 2, 4]
    },
}

# Определение лучших параметров и точности моделей
best_estimators = {}
for name, mod in models.items():
    grid = GridSearchCV(mod, param_gr[name], cv=5, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_estimators[name] = grid.best_estimator_
    print(f"Best parameters for {name}: {grid.best_params_}")
    print(f"Best accuracy for {name}: {grid.best_score_:.4f}")
    print('-' * 110)

# Оценка моделей с использованием кросс-валидации, вывод отчета о классификации, вывод матрицы ошибок
res2 = ml.report(best_estimators, X_train, X_test, y_train, y_test)

# Вывод лучшей модели с гиперпараметрами
ml.best_model(res2)

# Объединение результатов
res3 = ml.mergdict(res, res2)

# Создание и вывод таблицы наших результатов
accuracy = pd.DataFrame.from_dict(res3, orient='index', columns=['default', 'hyper'])
print(accuracy)

# График сравнения точности моделей
ml.com_chart(accuracy)