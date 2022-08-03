import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('./movie/02_movie_review_concat/2016_movie_review_concat.csv')
df.info()

okt = Okt()

df_stopwords = pd.read_csv('./movie/stopwords.csv')
stopwords = list(df_stopwords['stopword'])
stopwords = stopwords + ['영화', '연출', '관객', '개봉', '개봉일', '주인공', '출연', '배우', '리뷰',
                         '각본', '감독', '극장', '촬영', '네이버', '박스', '오피스', '박스오피스', '장면',
                         '씬', '신', '연기', '작품', '되어다']

cleaned_sentences = []

for review in df.reviews:
    review = re.sub('[^가-힣 ]', ' ', review)

    token = okt.pos(review, stem = True)
    df_token = pd.DataFrame(token, columns=['word', 'class'])
    df_token = df_token[(df_token['class'] == 'Noun') | (df_token['class'] == 'Verb') | (df_token['class'] == 'Adjective') | (df_token['class'] == 'Adverb')]


    words = []
    for word in df_token.word:
        if len(word) > 1:
            if word not in stopwords:
              words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)


df['cleaned_sentences'] = cleaned_sentences
df = df[['title', 'cleaned_sentences']]
df.dropna(inplace=True)

one_sentences = []
for title in df['title'].unique():
    temp = df[df['title'] == title]
    if len(temp) > 30:
        temp = temp.iloc[:30, :]
    one_sentence = ' '.join(temp['cleaned_sentences'])
    one_sentences.append(one_sentence)
df_one = pd.DataFrame({'titles':df['title'].unique(), 'reviews':one_sentences})
print(df_one.head())

df_one.to_csv('./movie/03_movie_cleaned_reviews/cleaned_review_one_2016.csv', index=False)
df.info()