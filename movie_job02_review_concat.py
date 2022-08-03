import pandas as pd
import glob
#경로에 있는 파일명 list 값으로 받아올 수 있음

#nan값 제거, 중복 제거

df = pd.DataFrame()
data_paths = glob.glob('./movie/01_movie_reviews_crawling_data/movie_review_2015/*')
for path in data_paths:
    df_temp = pd.read_csv(path)
    df_temp.dropna(inplace=True)
    df_temp.drop_duplicates(inplace=True)
    df = pd.concat([df, df_temp], ignore_index=True)


# df = pd.DataFrame()
# data_paths = glob.glob('./crawling_data/cleaned_review_one/*')

# for path in data_paths:
#     df_temp = pd.read_csv(path)
#     df_temp.dropna(inplace=True)
#     df_temp.drop_duplicates(inplace=True)
#     df = pd.concat([df, df_temp], ignore_index=True)

for i in range(1, 72):
    df_temp = pd.read_csv('./movie/01_movie_reviews_crawling_data/movie_review_2015/reviews_2015_1page.csv'.format(i))
    df_temp.dropna(inplace=True)
    df_temp.drop_duplicates(inplace=True)
    df = pd.concat([df, df_temp], ignore_index=True)


df.drop_duplicates(inplace=True)
df.info()

my_year = 2015
df.to_csv('./movie/02_movie_reviews_concat_data/reviews_concat_data_{}.csv'.format(my_year), index = False)
#
# df.to_csv('./crawling_data/reviews_2017_2022.csv', index = False)