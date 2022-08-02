from selenium import webdriver
import pandas as pd
from selenium.common.exceptions import NoSuchElementException
import time

options = webdriver.ChromeOptions()

options.add_argument('lang=ko_KR')
driver = webdriver.Chrome('./chromedriver.exe', options=options)

# https://movie.naver.com/movie/sdb/browsing/bmovie.naver?open=2020&page=1 ~37p
# title >>  //*[@id="old_content"]/ul/li[3]/a
# //*[@id="old_content"]/ul/li[2]/a
# //*[@id="old_content"]/ul/li[3]/a
# review >>  //*[@id="movieEndTabMenu"]/li[6]/a
# review건수 >>   //*[@id="reviewTab"]/div/div/div[2]/span/em
# review제목 >> //*[@id="reviewTab"]/div/div/ul/li[1]/a
# review내용 >>  //*[@id="content"]/div[1]/div[4]/div[1]/div[4]
# review페이지넘버 >>  //*[@id="pagerTagAnchor1"]

your_year = 2016

review_button_xpath = '//*[@id="movieEndTabMenu"]/li[6]/a'
review_number_xpath = '//*[@id="reviewTab"]/div/div/div[2]/span/em'

review_xpath = '//*[@id="content"]/div[1]/div[4]/div[1]/div[4]'

#네이버영화랭킹-디렉토리-개봉년도(2016) 페이지
for i in range(1, 60):
    url = 'https://movie.naver.com/movie/sdb/browsing/bmovie.naver?open={}&page={}'.format(your_year, i)
    titles = []
    reviews = []
    try:
        #driver.get(url)
        time.sleep(0.5)
        for j in range(1, 21): # ~21
            driver.get(url)
            movie_title_xpath = '//*[@id="old_content"]/ul/li[{}]/a'.format(j)  #영화제목(한페이지에 20개씩 리스트업)
            try:
                title = driver.find_element("xpath", movie_title_xpath).text
                driver.find_element("xpath", movie_title_xpath).click() #영화제목클릭
                time.sleep(0.5)
                driver.find_element("xpath", review_button_xpath).click() #리뷰 탭 클릭
                time.sleep(0.5)
                review_range = driver.find_element("xpath", review_number_xpath).text #총 리뷰수
                review_range = review_range.replace(',','')
                review_range = (int(review_range)-1) // 10 + 2 #한 페이지에 리뷰 10개씩 리스트업
                for k in range(1, review_range): # ~review_range
                    review_page_button_xpath = '//*[@id="pagerTagAnchor{}"]'.format(k)
                    try:
                        try:
                            driver.find_element("xpath", review_page_button_xpath).click()
                        except:
                            driver.find_element('xpath', '//*[@id="movieEndTabMenu"]/li[5]/a').click()
                        for l in range(1, 11): # ~11
                            back_flag = False
                            review_title_xpath = '//*[@id="reviewTab"]/div/div/ul/li[{}]/a'.format(l)
                            try:
                                review = driver.find_element("xpath", review_title_xpath).click()
                                back_flag = True
                                time.sleep(0.5)
                                review = driver.find_element("xpath", review_xpath).text
                                titles.append(title)
                                reviews.append(review)
                                driver.back()
                            except:
                                if back_flag :
                                    driver.back()  #back_flag=True일때, drive.back()
                                print('review', i, j, k, l)
                        driver.back()
                    except:
                        print('review page', i, j ,k)
            except:
                print('movie', i, j)
        df = pd.DataFrame({'title':titles, 'reviews':reviews})
        df.to_csv('./movie/01_movie_review_crawling_data/reviews_{}_{}page.csv'.format(your_year, i), index=False)
    except:
        print('page', i)

driver.close()