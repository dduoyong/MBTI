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


data = pd.read_csv('./melon_data/Dance0200.csv', index_col=0)

songName = data['song_name']
singerName = data['artist_name_basket']    
trs = str.maketrans("['']", '    ')
for i in tqdm(range(singerName.size)):
    a = data.iloc[i,8].translate(trs)
    a.replace('  ','')
    data.iloc[i,8] = a
# print(songName[:11]+singerName[:11])
# exit()


MAX_SLEEP_TIME=5
rand_value = randint(1, MAX_SLEEP_TIME)

options = wb.ChromeOptions()
# options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = wb.Chrome('./chromedriver.exe', options=options)

driver.get("https://www.melon.com/")
sleep(rand_value)
div = driver.find_element("css selector", "#gnbLoginDiv > div > button > span").click()
sleep(rand_value)
div = driver.find_element("css selector", "#conts_section > div > div > div:nth-child(3) > button").click()
sleep(rand_value)

driver.find_element("css selector", "#id").send_keys('0620julie')
driver.find_element("css selector", "#pwd").send_keys('dlgusrud1128!')
sleep(rand_value)
div = driver.find_element("css selector", "#btnLogin").click()
sleep(rand_value)

lyric2 = []
start = 17
end = 707956
for i in tqdm(range(start,end)):
    rand_value = randint(1, MAX_SLEEP_TIME)

    if i == start:
        driver.find_element("css selector", "#top_search").send_keys(songName[i]+', '+singerName[i])
        div = driver.find_element("css selector", "#gnb > fieldset > button.btn_icon.search_m > span").click()
        sleep(rand_value)

        driver.find_element("css selector", '#frm_songList > div > table > tbody > tr:nth-child(1) > td:nth-child(3) > div > div > a.btn.btn_icon_detail').click()
        sleep(rand_value)

        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        try:
            driver.find_element("css selector", '#lyricArea > button').click()
            sleep(1)
        except:
            pass

        lyric = driver.find_element("css selector", '#d_video_summary')
        lyric = lyric.text.replace("\n", " ")
        print(lyric)
        lyric2.append(lyric)

        driver.back()

    else:
        try:
            driver.find_element("css selector", "#top_search").send_keys(songName[i] + ', ' + singerName[i])
            div = driver.find_element("css selector", "#header_wrap > div.wrap_search_field > fieldset > button.btn_icon.search_m > span").click()
            sleep(rand_value)
            div = driver.find_element("css selector", '#frm_songList > div > table > tbody > tr:nth-child(1) > td:nth-child(3) > div > div > a.btn.btn_icon_detail').click()
            sleep(rand_value)

            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            try:
                div = driver.find_element("css selector", '#lyricArea > button').click()
                sleep(1)
            except:
                pass

            lyric = driver.find_element("css selector", '#d_video_summary')
            lyric = lyric.text.replace("\n", " ")
            print(lyric)
            lyric2.append(lyric)

            driver.back()

        except:
            lyric = '검색실패'
            lyric2.append(lyric)

    sleep(rand_value)
    driver.find_element("css selector", "#top_search").clear()

data1 = data.drop(data.index[:36765], axis=0)
data1 = data1.drop(data.index[41801:], axis=0)
data1['lyric'] = lyric2
data1.to_csv('test01.csv', index=False)