# Project films analysis 
анализ dataset о фильмах и сериалах и их рейтингах на IMBD :shipit:

Сделано: команда Гуманитарии (Павловская Екатерина, Асонкова Диана)
***

# Разделы
[Примеры полученных данных](https://github.com/ditayson/project-films/tree/main#%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D1%80%D1%8B-%D0%BF%D0%BE%D0%BB%D1%83%D1%87%D0%B5%D0%BD%D0%BD%D1%8B%D1%85-%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85)

[Корреляция](https://github.com/ditayson/project-films/tree/main#%D0%BA%D0%BE%D1%80%D1%80%D0%B5%D0%BB%D1%8F%D1%86%D0%B8%D0%B8)

[Гипотеза](https://github.com/ditayson/project-films/tree/main#%D0%B3%D0%B8%D0%BF%D0%BE%D1%82%D0%B5%D0%B7%D0%B0)

[Машинное обучение](https://github.com/ditayson/project-films#%D0%BC%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5-%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5)

***

# Файлы
[код](https://github.com/ditayson/project-films/blob/main/movie.ipynb)

[данные](https://github.com/ditayson/project-films/blob/main/movies.csv)

***

# Примеры полученных данных

## Анализ рейтингов фильмов

![image](https://github.com/ditayson/project-films/assets/133684422/f85294e5-93bb-49be-a05e-4880c07a6c99)

**ВЫВОД: наибольшее число фильмов подразумевают просмотр лицам до 17 лет обязательно в присутствие взрослого, то есть содержат неприемлемый контент для детей.**

## Точечный график зависимости бюджета от cборов

![image](https://github.com/ditayson/project-films/assets/133684422/001e7c5c-6b51-4380-9dbb-fd30716a63d0)

**ВЫВОД: наблюдается прямопропорциональная зависимость: чем больше бюджет фильма, тем больший доход от него получит производитель).**

## Посмотрим, какая компания получила больше прибыли

![image](https://github.com/ditayson/project-films/assets/133684422/88f26c35-32fc-4efa-bbc7-dcedc7bbf857)

**ВЫВОД: наибольший кассовый сбор получила компания Twentieth Century Fox**

## Выявим наиболее популярные жанры

![image](https://github.com/ditayson/project-films/assets/133684422/b1ae077b-a6c3-478a-b20e-acde869e937d)

**ВЫВОД: Можно увидеть, что наибольшее количество конттента на платформе - комедии. Драммы и боевики, тоже популярны. Самые редко встречающиесяся жанры - спрортивный, вестерн, музыкальный.**

## Посмотрим, фильмы какого жанра собрали больше всего выручки

![image](https://github.com/ditayson/project-films/assets/133684422/a495b061-cd2d-4af4-95f4-591e27023d42)

**ВЫВОД:наиболее прибыльными оказались боевики (что, безусловно, говорит о прогрессируещей тенденции популяризации и романтизации насилия)**

## Проанализируем каст: Самые популярные актеры фильмов 

![image](https://github.com/ditayson/project-films/assets/133684422/74211804-61c2-4344-bb24-efa0653f5b43)

**ВЫВОД: наибольшее число фильмов, в которых принимали участие Николас Кейдж и Том Хэнкс.**

## Посмотрим, фильм с участием какого актера получил больше прибыли

![image](https://github.com/ditayson/project-films/assets/133684422/4e922d06-b5be-4fed-907c-012ceaaafe01)

**ВЫВОД: наиболее прибыльным, оказался фильм с участием Сэма Уортингтона, но скорее всего, это Аватар, и дело вовсе не в Сэме..Зато целых 5 фильмов с участием с Роберта Дауни Мл.вошли в топ 20 (потому что он обаяшка), а значит, его участие в фильме может влиять на кассовые сборы.**

## Построим график распредедения контента разных жанров по годам

![image](https://github.com/ditayson/project-films/assets/133684422/7bcafe08-8f4e-4ae5-b3ed-c5d593e253a7)

**ВЫВОД: Можно увидеть, что на всем временном промежутке наиболее популярными жанрами были комедии и вестерны, наименее популярными: ужастики и музыкальные фильмы.**

## Для наглядности, построим круговую диаграмму по жанрам.

![image](https://github.com/ditayson/project-films/assets/133684422/b51d5fd8-48b5-4978-9ddd-2f553ab81ee9)

## Выявим Top 10 стран-производителей

![image](https://github.com/ditayson/project-films/assets/133684422/78e6b544-7928-4b66-848b-6a6a29fea1ac)

**ВЫВОД: наибольшие кассовые сборы наблюдаются среди фильмов, выпущенных США и Великобританией.**

## Проанализируем показатель IMDB оценки. Выявим контент с самым высоким рейтингом - это контент с оценкой более 9,0

![image](https://github.com/ditayson/project-films/assets/133684422/7d321326-cd1c-4c7d-ba28-2e5befa9d68c)

**ВЫВОД: фильм "Побег из Шоушенка" №1 в рейтинге (уверена, что из-за Моргана Фримена, потому что он шикарен)**

## Выявим контент с самым низким рейтингом - это контент с оценкой менее 2,0

![image](https://github.com/ditayson/project-films/assets/133684422/dfed3703-5f7e-4a6d-b445-6d079e2a58e4)

**ВЫВОД: фильмы, получившие самую низкую оценку - Супердетки: Вундеркинды 2 (потому что там играют дети), Нереальный блокбастер (с их безумной пародией на Марвел) и Красавица и уродина (там и по обложке всё понятно..)**

***
  
# Корреляции

## Посмотрим на корреляции числовых данных и определим факторы, имеющие наибольшую корреляцию с валовым доходом

![image](https://github.com/ditayson/project-films/assets/133684422/fa43fc09-9d1b-415e-aaff-089a67211dbd)

**Отдельно рассмотрим корреляцию между бюджетом и кассовыми сборами, поскольку коэффициент данной корреляции близок к 1 (0.746712),и значит можно сделать предположение, что между переменными наблюдается положительная корреляция.**

## Визуализируем матрицу корреляции между бюджетом и cборами в виде Heatmap

![image](https://github.com/ditayson/project-films/assets/133684422/b01433b3-8450-425c-a460-7f55deda5122)

**ВЫВОД: Очевидно, что бюджет, сборы и голоса избирателей взаимосвязаны.**

## Создадим Heatmap

![image](https://github.com/ditayson/project-films/assets/133684422/3f9a9e18-5e3a-49f8-bef7-6ae18c02e0a8)

**ВЫВОД:не заметна какая-либо сильная корреляция между нечисловыми характеристиками. Однако видно, что все же есть зависимость score от votes, year, runtime, genre и друими характеристиками в меньшей степени, а значит есть смысл предиктить оценку фильма в зависимости от них.**

## Построим набор гистограмм

![image](https://github.com/ditayson/project-films/assets/133684422/b6c2540a-73f3-41b9-9f20-67cd76ed2e0d)

**Среди числовых признаков нет константных, но есть такие, у которых доминирует одно значение.**

## Построим матрицу графиков

![image](https://github.com/ditayson/project-films/assets/133684422/0d35b258-b61a-4752-9cdb-d987997a3ff2)


## Детально рассмотрим график score

![image](https://github.com/ditayson/project-films/assets/133684422/e273bfbd-3ecb-4be1-b5b0-f9c5cf234155)

**ВЫВОД: График напоминает нормальное распределение.**

## Построим BoxPlot для значений score

![image](https://github.com/ditayson/project-films/assets/133684422/cf9fa125-9a72-4b2b-be4f-c5d877b0818e)

**ВЫВОД: BoxPlot деманстрирует, что медиана распределения находится где-то около 6,5. Верхний (0.75) квантиль расположен около 7, а нижний (0,25) квантиль - между 5 и 6. Большое количествро выбросов ниже min, и одно значение выше max.**




***

# Гипотеза

**№1 Гипотеза: Проверим, что распределение нормальное, с помощью критерия согласия.**

Нулевая гипотеза: X принадлежин нормальному распределению
Альтернативная гипотеза: X не принадлежин нормальному распределению
Для проверки гипотезы воспользуемся тестом Колмогорова-Смирнова

```
# Создадим новый датасет и удалим лишние колонки
#define subsetted DataFrame
score_data = m_data. copy ()
score_data.drop(["name","rating","genre","year","released","votes","director","writer","star","country","budget","gross","company","runtime"], axis = 1, inplace = True)
  
#Выведем
print(score_data)

round_data = np.round(score_data['score'],decimals = 0) 

from scipy.stats import kstest
from scipy.stats import lognorm
from scipy import stats
stats.kstest(round_data, 'norm', args=(round_data.mean(), round_data.std(ddof=1)))
```

**KstestResult(statistic=0.1974903237511067, pvalue=2.3906665907588564e-259)**

### Видим, что p-value очень низкое, значит гипотеза о нормальности отвергается. Делаем вывод о том, что выборка из x (значений score) не имеет нормального распределения.


***

**№2 Гипотеза о положительной корреляции между продолжительностью фильма и его бюджетом.**

Нулевая гипотеза: средние бюджеты коротких и длинных фильмов равны.
Альтернативная гипотеза: средние бюджеты коротких и длинных фильмов различаются.

Для проверки этой гипотезы через среднее используем t-test Стьюдента. Для этого разделим выборку на две группы: фильмы с продолжительностью менее 120 минут и фильмы с продолжительностью более 120 минут.

```
from scipy.stats import t
from math import sqrt

# разделение выборки на две группы
short_films = m_data[m_data['runtime'] < 120]['budget']
long_films = m_data[m_data['runtime'] >= 120]['budget']

# вычисление среднего значения для каждой группы
mean_short = short_films.mean()
mean_long = long_films.mean()

# вычисление стандартного отклонения для каждой группы
std_short = short_films.std()
std_long = long_films.std()

# вычисление t-статистики
n1 = len(short_films)
n2 = len(long_films)
t_stat = (mean_long - mean_short) / (sqrt((std_long**2/n2) + (std_short**2/n1)))

# определение уровня значимости и степени свободы
alpha = 0.05
df = n1 + n2 - 2

# вычисление критического значения
t_crit = t.ppf(1 - alpha/2, df)

# сравнение t-статистики с критическим значением
if abs(t_stat) > t_crit:
    print("Гипотеза подтверждается")
else:
    print("Гипотеза не подтверждается")
```

**Гипотеза подтверждается**

### Из проведенного анализа данных следует, что существует положительная корреляция между продолжительностью фильма и его бюджетом. Это означает, что чем больше продолжительность фильма, тем выше вероятность того, что его бюджет будет больше.



***

# Машинное обучение

**score - наша объясняемая переменная**

```
m_data.head()
numeric = ['votes', 'budget', 'gross', 'runtime']
X = m_data[numeric]
y = m_data['score']

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression # подгрузили модель
scaler = MinMaxScaler()
df_scaled_train = pd.DataFrame(scaler.fit_transform(X), columns=numeric,index=X.index)

X_train, X_test, y_train, y_test = train_test_split(df_scaled_train, y, test_size=0.2, random_state=42, shuffle=True)
X_train
```

![image](https://github.com/ditayson/project-films/assets/133684422/1047a378-7844-4035-94ec-195e857c2e6e)

```
linreg = LinearRegression()

# Обучили модель на тренировочной выборке
linreg.fit(X_train,y_train)

# Сделали прогнозы на тестовой выборке 
predicted = linreg.predict(X_test)

# Посчитаем метрики регрессии
from sklearn import metrics  # подгружаем метрики

def print_metrics(predicted, y_test):
    print('MAE:', "%.4f" % metrics.mean_absolute_error( predicted, y_test)) #Средняя абсолютная ошибка, MAE
    print('RMSE:', "%.4f" % np.sqrt(metrics.mean_squared_error(predicted, y_test))) #Средняя квадратичная ошибка, RMSE
    print('MAPE:', "%.4f" % metrics.mean_absolute_percentage_error(predicted, y_test)) #Средняя абсолютная процентная ошибка, MAPE

print_metrics(predicted, y_test)
```

**MAE: 0.6323
RMSE: 0.8178
MAPE: 0.1006**

```
# Линейная регрессия 
model = LinearRegression()
print('Lineer Regression')
model.fit(X_train,y_train)
predicted = model.predict(X_test)

print('Test Score: ',metrics.mean_squared_error(y_test,predicted))
predicted = model.predict(X_train)
print('Train Score: ',metrics.mean_squared_error(y_train,predicted))
```

**Lineer Regression
Test Score:  0.6688703822809126
Train Score:  0.6682895828525601**

___Разница между Test Score и Train Score невелика, значит модель обучилась.___

```
# Random Forest
model = RandomForestRegressor(n_estimators = 500, max_depth=3)
print('Random Forest')
model.fit(X_train,y_train)
predicted = model.predict(X_test)
print('метрики')
print_metrics(y_test, predicted)
print('')
print('Test Score: ',metrics.mean_squared_error(y_test,predicted))
predicted = model.predict(X_train)
print('Train Score: ',metrics.mean_squared_error(y_train,predicted))
```

**Random Forest
метрики
MAE: 0.6337
RMSE: 0.8121
MAPE: 0.1106
Test Score:  0.6594748682208071
Train Score:  0.6382946390499364**

___Разница между Test Score и Train Score невелика, значит модель обучилась.___

```
#KNN-5 (Метод ближайших соседей)
from sklearn.neighbors import KNeighborsRegressor
print('KNN-5')
model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train, y_train)

predict_knn = model_knn.predict(X_test)
#Посчитаем метрики для KNN 
print('метрики')
print_metrics(y_test, predict_knn)
print('')
print('Test Score: ',metrics.mean_squared_error(y_test,predict_knn))
predict_knn = model_knn.predict(X_train)
print('Train Score: ',metrics.mean_squared_error(y_train,predict_knn))
```

**KNN-5
метрики
MAE: 0.5967
RMSE: 0.7941
MAPE: 0.1047
Test Score:  0.630555657292348
Train Score:  0.4157408570493948

**Gосмотрим, как будет вести себя алгоритм при увеличении числа соседей**

```
model_knn_10 = neighbors.KNeighborsRegressor(n_neighbors = 10)
print('KNN-10')
model_knn_10.fit(X_train,y_train)
predict_knn_10 = model_knn_10.predict(X_test)
#Посчитаем метрики для KNN 
print('метрики')
print_metrics(y_test, predict_knn_10)
print('')
print('Test Score: ',metrics.mean_squared_error(y_test,predict_knn_10))
predict_knn_10 = model_knn_10.predict(X_train)
print('Train Score: ',metrics.mean_squared_error(y_train,predict_knn_10))
predict_knn_10 = model_knn_10.predict(X_test)
```

**KNN-10
метрики
MAE: 0.5738
RMSE: 0.7627
MAPE: 0.1009
Test Score:  0.5817855461085677
Train Score:  0.47692976774615636**

___Мы знаем, что качество kNN при увеличении К должно сначала расти (приближаясь к числу объектов в выборке), а потом падать, и оптимум будем где-то посередине. С учетом, того, что наша выборка большая, можно увеличить К.___

```
model_knn_n = neighbors.KNeighborsRegressor(n_neighbors = 500)
print('KNN-500')
model_knn_n.fit(X_train,y_train)
predict_knn_n = model_knn_n.predict(X_test)
#Посчитаем метрики для KNN 
print('метрики')
print_metrics(y_test, predict_knn_n)
print('')
print('Test Score: ',metrics.mean_squared_error(y_test,predict_knn_n))
predict_knn_n = model_knn_n.predict(X_train)
print('Train Score: ',metrics.mean_squared_error(y_train,predict_knn_n))
predict_knn_n = model_knn_n.predict(X_test)
```

**KNN-500
метрики
MAE: 0.6327
RMSE: 0.8060
MAPE: 0.1088
Test Score:  0.6495738641465009
Train Score:  0.6458407197579327**


### ВЫВОД: У модели значения метрик больше остальных моделей, значит, мы можем сказать, что она обучилась лучше остальных. (что, в целом, и объясняет ее появление в нашем проекте)
