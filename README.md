# project-films
анализ dataset о фильмах и сериалах Netflix и их рейтингах на IMBD
# Анализ
#Проанализируем рейтинги фильмов

plt.figure(figsize=(10,8))
sns.set(style="white")

# Сделаем переход от самого темного фиолетового к самому светлому
ax = sns.countplot(x="Rating", data=netflix_movies, palette="Purples_r", order=netflix_movies['Rating'].value_counts().index[0:10])

