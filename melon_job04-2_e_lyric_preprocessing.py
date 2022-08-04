import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize  #단어 단위를 기준으로 토큰화
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import pandas as pd
import re

# nltk.download()

music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(6, 7):

    gn = music_genre_lst[i]
    df = pd.read_csv('./melon/03_melon_lyric_concat_data/{}_lyric_concat.csv'.format(gn))
    # df.info()


    English_lyric = []

    for lyrics in df.lyric[:10]:
        # -- 영어 문자 제외하고 모두 공백으로 대체 --
        lyrics = re.sub('[^a-zA-Z]', ' ', lyrics)
        #  -- 모든 영어 텍스트 소문자로 대체 --
        small_lyrics = lyrics.lower()
        print('small_lyrics :', small_lyrics)

        # -- 토큰화 --
        # nltk.download('tagsets')
        # nltk.download('averaged_perceptron_tagger')
        # nltk.download('wordnet')
        token_words = word_tokenize(small_lyrics)
        print('token_words :', token_words)
        # exit()

        # lemm = WordNetLemmatizer()
        # # for word in token_words:
        # print([lemm.lemmatize(word, pos='v') for word in token_words])
        #
        # exit()

        # print(token_words)
        # -- 품사 tagging --
        tag_words = pos_tag(token_words)
        # print(tag_words)
        # exit()
        df_token = pd.DataFrame(tag_words, columns=['word', 'class'])
        # -- 명사(단복수)/고유명사(단복수)/동사(원형,과거,현재분사,과거분사)/형용사(비교급,최고급)/부사(비교급,최고급) 의 품사만 살림 --
        # df_token = df_token[(df_token['class'] == 'NN') | (df_token['class'] == 'NNS') |
        #                     (df_token['class'] == 'VB') | (df_token['class'] == 'VBD') | (df_token['class'] == 'VBG') | (df_token['class'] == 'VBN') |
        #                     (df_token['class'] == 'JJ') | (df_token['class'] == 'JJR') | (df_token['class'] == 'JJS') |
        #                     (df_token['class'] == 'RB') | (df_token['class'] == 'RBR') | (df_token['class'] == 'RBS')]
        # df_token.info()
        # print(df_token)
        # exit()

        lemmatizer = WordNetLemmatizer()
        df_noun = df_token[(df_token['class'] == 'NN') | (df_token['class'] == 'NNS')]
        df_verb = df_token[(df_token['class'] == 'VB') | (df_token['class'] == 'VBD') | (df_token['class'] == 'VBG') | (df_token['class'] == 'VBN')]
        df_adjective = df_token[(df_token['class'] == 'JJ') | (df_token['class'] == 'JJR') | (df_token['class'] == 'JJS')]
        df_adverb = df_token[(df_token['class'] == 'RB') | (df_token['class'] == 'RBR') | (df_token['class'] == 'RBS')]
        # df_verb.info()
        # print(df_verb)
        # print('표제어 추출 전 :', token_words)
        # print('표제어 추출 후 :', [lemmatizer.lemmatize(word) for word in df_verb['word']])

        print([lemmatizer.lemmatize(w, pos='n') for w in df_noun['word']])   #명사
        print([lemmatizer.lemmatize(w, pos='v') for w in df_verb['word']])    #동사
        print([lemmatizer.lemmatize(w, pos='a') for w in df_adjective['word']])   #형용사
        print([lemmatizer.lemmatize(w, pos='r') for w in df_adverb['word']])   #부사
        # exit()


        # -- nltk stopwords & 불용어 제거 --
        stop_words = set(stopwords.words('english'))


        result = []
        for token in token_words:
            if token not in stop_words:
                result.append(token)

        # print(token_words)
        # print(result)

        lemm = WordNetLemmatizer()
        # for word in token_words:
        # print([lemm.lemmatize(word, pos='v') for word in token_words])

        # exit()


        cleaned_sentence = ' '.join(result)
        English_lyric.append(cleaned_sentence)
        print(English_lyric)

    # df['English_clean_lyric'] = English_lyric
    # df = df[['artist', 'title', 'English_clean_lyric']]
    # df.dropna(inplace=True)
    #
    # df.to_csv('./melon/04_melon_clear_lyric/{}_clean_eng_lyric_delete_고유명사.csv'.format(gn), index=False)
    # df.info()

