from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt


# Масштабирование признаков
def preprocessor(data, numeric_features, categorical_features):
    X = data.drop(['target'], axis=1)

    # Создание препроцессора
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features),
            ('cat', cat_transformer, categorical_features)
        ])

    # Применение препроцессора к данным
    X_processed = preprocessor.fit_transform(X)
    return X_processed, preprocessor

# Оценка моделей с использованием кросс-валидации, вывод отчета о классификации, вывод матрицы ошибок
def report(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, mod in models.items():
        scores = cross_val_score(mod, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = scores.mean()
        mod.fit(X_train, y_train)
        y_pred = mod.predict(X_test)
        print(f'{name}: {scores.mean():.4f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('*' * 55)
    return results

# Определение лучшей модели
def best_model(res):
    best_model_name = max(res, key=res.get)
    best_model_score = res[best_model_name]
    print(f'\nBest model: {best_model_name} with accuracy {best_model_score * 100:.2f}%')
    print('-' * 100)

# Функция объединения словарей
def mergdict(dict1, dict2):
    for k, v in dict2.items():
        if dict1.get(k):
            dict1[k] = [dict1[k], v]
        else:
            dict1[k] = v
    return dict1

def com_chart(df):
    df.plot.bar(color=['yellow', 'green'])
    plt.title('Сравнение точности моделей')
    plt.xlabel('Методы')
    plt.ylabel('Точность моделей')
    plt.tight_layout()
    plt.ylim(0, 1.2)
    plt.legend(loc='best')
    plt.show()