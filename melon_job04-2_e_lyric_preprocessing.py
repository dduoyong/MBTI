import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize  #띄어쓰기를 기준으로 토큰화
from nltk.tag import pos_tag
import pandas as pd
import re

# nltk.download()

music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(6, 10):

    gn = music_genre_lst[i]
    df = pd.read_csv('./melon/03_melon_lyric_concat_data/{}_lyric_concat.csv'.format(gn))
    # df.info()


    for lyrics in df.lyric:
        # -- 영어 문자 제외하고 모두 공백으로 대체 --
        lyrics = re.sub('[^a-zA-Z]', ' ', lyrics)
        #  -- 모든 영어 텍스트 소문자로 대체 --
        small_lyrics = lyrics.lower()

        # -- 토큰화 --
        nltk.download('tagsets')
        nltk.download('averaged_perceptron_tagger')
        token_words = word_tokenize(str(small_lyrics))
        # print(token_words)
        # -- 품사 tagging --
        tag_words = pos_tag(token_words)
        df_token = pd.DataFrame(tag_words, columns=['word', 'class'])
        # -- 명사(단복수)/고유명사(단복수)/동사(원형,과거,현재분사,과거분사)/형용사(비교급,최고급)/부사(비교급,최고급) 의 품사만 살림 --
        df_token = df_token[(df_token['class'] == 'NN') | (df_token['class'] == 'NNS') | (df_token['class'] == 'NNP') | (df_token['class'] == 'NNPS') |
                            (df_token['class'] == 'VB') | (df_token['class'] == 'VBD') | (df_token['class'] == 'VBG') | (df_token['class'] == 'VBN') |
                            (df_token['class'] == 'JJ') | (df_token['class'] == 'JJR') | (df_token['class'] == 'JJS') |
                            (df_token['class'] == 'RB') | (df_token['class'] == 'RBR') | (df_token['class'] == 'RBS')]
        # df_token.info()
        # print(df_token)

        # -- nltk stopwords & 불용어 제거 --
        stop_words = set(stopwords.words('english'))

        English_lyric = []
        result = []
        for token in token_words:
            if token not in stop_words:
                result.append(token)

        print(token_words)
        print(result)
        cleaned_sentence = ' '.join(result)
        English_lyric.append(cleaned_sentence)

    df['English_clean_lyric'] = English_lyric
    df = df[['artist', 'title', 'English_clean_lyric']]
    df.dropna(inplace=True)

    df.to_csv('./melon/04_melon_clear_lyric/{}_clean_eng_lyric.csv'.format(gn), index=False)
    df.info()

