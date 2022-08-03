import pandas as pd
import glob

# ---- 크롤링 된 가사 데이터 확인 후 최종 10개 장르 선정 ----
music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in music_genre_lst:
    data_paths = glob.glob('./melon/02_melon_lyric_crawling_data/{}/*'.format(i))
    df = pd.DataFrame()
    # df.info()

    for path in data_paths:
        df_temp = pd.read_csv(path)
        df_temp.dropna(inplace=True)
        df_temp.drop_duplicates(inplace=True)
        df = pd.concat([df, df_temp], ignore_index=True)

    df.drop_duplicates(inplace=True)
    df.info()

    df.to_csv('./melon/03_melon_lyric_concat_data/{}_lyric_concat.csv'.format(i), index = False)