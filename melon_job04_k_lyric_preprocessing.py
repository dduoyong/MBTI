from konlpy.tag import Okt
import pandas as pd
import re

music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(4, 5):

    gn = music_genre_lst[i]

    df = pd.read_csv('./melon/03_melon_lyric_concat_data/{}_lyric_concat.csv'.format(gn))
    df.info()

    df_stopwords = pd.read_csv('./melon/kor_stopwords.csv')
    df_stopwords.info()

    stopwords = list(df_stopwords['stopwords'])
    stopwords = stopwords + ['라라라라', '라라']

    okt = Okt()
    Korean_lyric = []

    for lyrics in df.lyric:
        lyrics = re.sub('[^가-힣 ]', ' ', lyrics)  #문자열 review에서 [가-힣]빼고 공백으로 대체
        token = okt.pos(lyrics, stem=True)

        df_token = pd.DataFrame(token, columns=['word', 'class'])  #튜플형태 >> 컬럼 두개짜리 데이터프레임 으로 변환
        df_token = df_token[(df_token['class']=='Noun') | (df_token['class']=='Verb') | (df_token['class']=='Adjective') | (df_token['class']=='Adverb')]
        df_token.info()
        print(df_token)

        words = []
        for word in df_token.word:
            if len(word) > 1:
                if word not in stopwords :
                    words.append(word)
        cleaned_sentence = ' '.join(words)
        Korean_lyric.append(cleaned_sentence)

df['Korean_clean_lyric'] = Korean_lyric
df = df[['artist', 'title', 'Korean_clean_lyric']]
df.dropna(inplace=True)

df.to_csv('./melon/04_melon_clear_lyric/{}_clean_kor_lyric.csv'.format(gn), index=False)
df.info()