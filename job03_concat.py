import pandas as pd
import glob


music_genre_lst = ['Adultpop', 'Ballad', 'Classic', 'Dance', 'FandB', 'Idol', 'Indie', 'Jazz', 'Jpop', 'NewAge', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in music_genre_lst:

    data_paths = glob.glob('melon_lyrics_crawling_data/{}/*'.format(i))
    df = pd.DataFrame()
    df.info()

    for path in data_paths:
        df_temp = pd.read_csv(path)
        df_temp.dropna(inplace=True)
        df_temp.drop_duplicates(inplace=True)
        df = pd.concat([df, df_temp], ignore_index=True)


# for i in range(1, 38):
#     df_temp = pd.read_csv('./crawling_data/reviews_2020_{}page.csv'.format(i))
#     df_temp.dropna(inplace=True)
#     df_temp.drop_duplicates(inplace=True)
#     df = pd.concat([df, df_temp], ignore_index=True)

    df.drop_duplicates(inplace=True)
    df.info()

    df.to_csv('./concat_lyrics_crawling_data/crawling_data.lyrics_{}.csv'.format(i), index = False)

# df.to_csv('./melon_lyric_data/reviews_2017_2022.csv', index = False)