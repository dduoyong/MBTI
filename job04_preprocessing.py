import pandas as pd
from konlpy.tag import Kkma
from konlpy.tag import Okt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # 시각화 지원 라이브러리
import tqdm as tq
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm as tq
import re

okt = Okt()

df_stopwords = pd.read_csv('melon/stopwords.csv')
# stopwords = list(df_stopwords['stopword'])

#한국 가사만 처리하므로 pop 장르 우선 제외
music_genre_lst = ['Adultpop', 'Ballad', 'Classic', 'Dance', 'FandB', 'Idol', 'Indie', 'Jazz', 'Jpop', 'NewAge', 'RandB_S', 'RandH', 'RandM']

for i in music_genre_lst:

    df = pd.read_csv('./melon/03_melon_lyric_concat_data/{}_lyric_concat.csv'.format(i))
    df.info()


    #전처리 진행과정 표시
    cnt = 0


    lyr = df['lyric']

    for lyric in df.lyric:
        cnt += 1
        if cnt % 10 == 0:
            print('.', end='')
        if cnt % 100 == 0:
            print()


def clean_text(text):
    cleaned_text = re.sub('[a-zA-z]', '', text)
    cleaned_text = re.sub('[0-9]', '', cleaned_text)
    cleaned_text = re.sub('[\{\}\[�\]\/?.,;:|©\“”‘)\u2005*~’`!^\-_+<>@\#$%&\\\=\(\'\"\♥\♡\ㅋ\ㅠ\ㅜ\ㄱ\ㅎ\ㄲ\ㅡ\접기]', '',
                          cleaned_text)
    return cleaned_text

    kor_lyr = []

    for i in range(len(lyr)):
        kor_lyr.append(clean_text(lyr[i]))

    data['kor_lyr'] = kor_lyr



    print(kor_lyr)

    exit()


    token = okt.pos(lyr, stem= True)
    df_token = pd.DataFrame(token, columns=['word', 'class'])
    df_token = df_token[(df_token['class'] == 'Noun') |
                        (df_token['class'] == 'Verb') |
                        (df_token['class'] == 'Adjective')]

    print(df_token)




    print(lyr)
    exit()

    kor_lyr = []
    for j in range(len(lyr)):

        cleaned_lyrics = re.sub('[^가-힣 ]', ' ', lyr)
        kor_lyr.append(cleaned_lyrics[j])

    df['kor_lyr'] = kor_lyr
    kor_lyr.to_csv('./cleaned_lyrics_data/cleaned_lyrics_{}'.format(i))

kor_lyr.info()