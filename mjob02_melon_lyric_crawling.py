from selenium import webdriver as wb
from selenium.webdriver.common.keys import Keys
from time import sleep
import time
import os
from urllib.request import urlretrieve
from tqdm.notebook import tqdm
import pandas as pd
import pprint
from random import randint


data = pd.read_csv('./melon_data/Ballad0100.csv', index_col=0)

songName = data['song_name']
singerName = data['artist_name_basket']    
trs = str.maketrans("['']", '    ')
for i in tqdm(range(singerName.size)):
    a = data.iloc[i,8].translate(trs)
    a.replace('  ','')
    data.iloc[i,8] = a
# print(songName[:11]+singerName[:11])
# exit()


# MAX_SLEEP_TIME=5
# rand_value = randint(1, MAX_SLEEP_TIME)

options = wb.ChromeOptions()
#크롬 창 띄우기
# options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = wb.Chrome('./chromedriver.exe', options=options)

#멜론 창 접속
driver.get("https://www.melon.com/")
time.sleep(1)

##로그인
# sleep(rand_value)
# div = driver.find_element("css selector", "#gnbLoginDiv > div > button > span").click()
# sleep(rand_value)
# div = driver.find_element("css selector", "#conts_section > div > div > div:nth-child(3) > button").click()
# sleep(rand_value)
#
# driver.find_element("css selector", "#id").send_keys('0620julie')
# driver.find_element("css selector", "#pwd").send_keys('dlgusrud1128!')
# sleep(rand_value)
# div = driver.find_element("css selector", "#btnLogin").click()
# sleep(rand_value)

lyric2 = []

#가수 id의 시작 값과 끝 값
start = 9
end = 707686

for i in range(start,end):
    # rand_value = randint(1, MAX_SLEEP_TIME)
    time.sleep(3)

    if i == start:
        driver.find_element("css selector", "#top_search").send_keys(songName[i]+', '+singerName[i])
        time.sleep(1)
        div = driver.find_element("css selector", "#gnb > fieldset > button.btn_icon.search_m > span").click()
        time.sleep(1)

        print(songName[i]+', '+singerName[i])

        try:
            #노래 가사 버튼 클릭
            driver.find_element("css selector", '#frm_songList > div > table > tbody > tr:nth-child(1) > td:nth-child(3) > div > div > a.btn.btn_icon_detail').click()
            time.sleep(1)

            #스크롤 다운
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                #스크롤 올리기
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)

                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            print('스크롤 완료')
            time.sleep(4)

            # 가사 펼치기 버튼 찾아서 클릭
            try:
                driver.find_element("css selector", '#lyricArea > button').click()
                sleep(1)
                lyric = driver.find_element("css selector", '#d_video_summary')
                lyric = lyric.text.replace("\n", " ")
                print(lyric)
                lyric2.append(lyric)

            #가사 펼치기 버튼 없으면 없으면 패스
            except:
                pass

            #뒤로 가기
            driver.back()
            print('debug01')

        except:
            pass

    else:

        try:
            driver.find_element("css selector", "#top_search").send_keys(songName[i] + ', ' + singerName[i])
            time.sleep(1)
            div = driver.find_element("css selector", "#header_wrap > div.wrap_search_field > fieldset > button.btn_icon.search_m > span").click()
            time.sleep(1)
            print(songName[i] + ', ' + singerName[i])

            try:
                # 노래 가사 버튼 클릭
                driver.find_element("css selector",
                                    '#frm_songList > div > table > tbody > tr:nth-child(1) > td:nth-child(3) > div > div > a.btn.btn_icon_detail').click()
                time.sleep(1)

                # 스크롤 다운
                last_height = driver.execute_script("return document.body.scrollHeight")
                while True:
                    # 스크롤 올리기
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1)

                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height

                print('스크롤 완료')
                time.sleep(4)

                # 가사 펼치기 버튼 찾아서 클릭
                try:
                    driver.find_element("css selector", '#lyricArea > button').click()
                    sleep(1)
                    lyric = driver.find_element("css selector", '#d_video_summary')
                    lyric = lyric.text.replace("\n", " ")
                    print(lyric)
                    lyric2.append(lyric)

                # 가사 펼치기 버튼 없으면 없으면 패스
                except:
                    pass

                # 뒤로 가기
                driver.back()
                print('debug02')

            except:
                pass
            print('debug03')

        except:
            lyric = '검색실패'
            lyric2.append(lyric)
            time.sleep(1)

    print('debug04')
    time.sleep(1)
    driver.find_element("css selector", "#top_search").clear()

data1 = data.drop(data.index[:36765], axis=0)
data1 = data1.drop(data.index[41801:], axis=0)
data1['lyric'] = lyric2
data1.to_csv('test01.csv', index=False)