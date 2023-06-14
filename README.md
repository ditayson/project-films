# Project films analysis 
анализ dataset о фильмах и сериалах и их рейтингах на IMBD :shipit:

***

# Разделы
[Примеры полученных данных](https://github.com/ditayson/project-films/tree/main#%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D1%80%D1%8B-%D0%BF%D0%BE%D0%BB%D1%83%D1%87%D0%B5%D0%BD%D0%BD%D1%8B%D1%85-%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85)

[Корреляция](https://github.com/ditayson/project-films/tree/main#%D0%BA%D0%BE%D1%80%D1%80%D0%B5%D0%BB%D1%8F%D1%86%D0%B8%D0%B8)

[Гипотеза](https://github.com/ditayson/project-films/tree/main#%D0%B3%D0%B8%D0%BF%D0%BE%D1%82%D0%B5%D0%B7%D0%B0)

***

# Файлы
[код](https://github.com/ditayson/project-films/blob/main/netflix.ipynb)

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

# Визуализируем матрицу корреляции между бюджетом и cборами в виде Heatmap

![image](https://github.com/ditayson/project-films/assets/133684422/b01433b3-8450-425c-a460-7f55deda5122)

**ВЫВОД: Очевидно, что бюджет, сборы и голоса избирателей взаимосвязаны.**

## Столбцы с типом данных object мы преобразуем в тип данных категории, чтобы тоже включить их в корреляцию.

![image](https://github.com/ditayson/project-films/assets/133684422/bfc2fe03-9695-46f4-be84-1e4a3fb4af20)

## Создадим Heatmap

![image](https://github.com/ditayson/project-films/assets/133684422/3f9a9e18-5e3a-49f8-bef7-6ae18c02e0a8)

**ВЫВОД:не заметна какая-либо выдающаяся корреляция между нечисловыми характеристиками. Однако видно, что все же есть зависимость score от votes, year, runtime, genre и друими характеристиками в меньшей степени, а значит есть смысл предиктить оценку фильма в зависимости от них.**

## Посмотрим распределение оценок

![image](https://github.com/ditayson/project-films/assets/133684422/e273bfbd-3ecb-4be1-b5b0-f9c5cf234155)

**График напоминает нормальное распределение.**

## Проверим гипотезу о распределение нормальное

![image](https://github.com/ditayson/project-films/assets/133684422/cf9fa125-9a72-4b2b-be4f-c5d877b0818e)

**ВЫВОД: видно, что набор данных не сбалансирован**



## Для наглядности, построим круговую диаграмму по жанрам.

![image](https://github.com/ditayson/project-films/assets/133684422/b51d5fd8-48b5-4978-9ddd-2f553ab81ee9)



![image](https://github.com/ditayson/project-films/assets/133684422/b6c2540a-73f3-41b9-9f20-67cd76ed2e0d)

**Среди числовых признаков нет константных, но есть такие, у которых доминирует одно значение.**


## Построим матрицу графиков, чтобы исследовать корреляцию

![image](https://github.com/ditayson/project-films/assets/133684422/0d35b258-b61a-4752-9cdb-d987997a3ff2)

***

# Гипотеза
После анализа данных о фильмах и рейтингах фильмов и сериалов Netflix в различных странах, мы заметили, что преобладающее число актеров является индийской национальности, поэтому можно выдвинуть данную гипотезу:

Включение индийских актеров в фильмы и сериалы Netflix увеличивает интерес к ним со стороны аудитории из Индии и других стран, где популярна индийская культура и киноиндустрия, что в свою очередь положительно сказывается на рейтинге этих произведений.

Для подверждения стоит проанализировать, как влияют актеры других стран на рейтинг фильмов и сериалов.

Для проверки данной гипотезы можно использовать следующие методы математической статистики:

1. Анализ дисперсии (ANOVA). Этот метод позволяет определить, есть ли статистически значимая разница в рейтингах фильмов и сериалов с участием индийских актеров и без них в разных странах. Если разница будет статистически значимой, то можно сделать вывод о том, что наличие индийских актеров действительно влияет на рейтинг произведения.

2. Корреляционный анализ. Этот метод позволяет определить, есть ли связь между наличием индийских актеров и рейтингом произведения в разных странах. Если будет обнаружена положительная корреляция, то это будет подтверждением гипотезы о том, что наличие индийских актеров увеличивает интерес к произведению у аудитории из стран, где популярна индийская культура и киноиндустрия.

3. Множественная регрессия. Этот метод позволяет определить, какие факторы (в том числе наличие индийских актеров) влияют на рейтинг произведения в разных странах. Если будет обнаружено, что наличие индийских актеров является статистически значимым фактором, то это будет подтверждением гипотезы о том, что наличие индийских актеров увеличивает интерес к произведению у аудитории из стран, где популярна индийская культура и киноиндустрия.
