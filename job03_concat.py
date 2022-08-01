import pandas as pd
import glob

#노래 장르별 리스트
music_genre_lst = ['Adultpop', 'Ballad', 'Classic', 'Dance', 'FandB', 'Idol', 'Indie', 'Jazz', 'Jpop', 'NewAge', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in music_genre_lst:

    #각 장르별 폴더의 크롤링 된 파일 모두 불러오기
    data_paths = glob.glob('melon_lyrics_crawling_data/{}/*'.format(i))
    df = pd.DataFrame()
    df.info()

    for path in data_paths:
        df_temp = pd.read_csv(path)
        df_temp.dropna(inplace=True)
        df_temp.drop_duplicates(inplace=True)
        df = pd.concat([df, df_temp], ignore_index=True)

    df.drop_duplicates(inplace=True)
    df.info()

    #'concat_lyrics_crawling_data'폴더에 저장
    df.to_csv('./concat_lyrics_crawling_data/crawling_data.lyrics_{}.csv'.format(i), index = False)
