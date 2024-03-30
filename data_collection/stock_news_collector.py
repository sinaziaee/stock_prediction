
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import utils as utils
from datetime import datetime as dt
from newspaper import Article
import os
from tqdm import tqdm

df = pd.read_csv('../datasets/stocks.csv')
df.head()

news_list_path = utils.create_path('../datasets/news_links')
BASE_URL = 'https://www.investing.com'
MAX_NUM_PAGES =20

def extract_news_links(df, news_list_path,  max_num_pages=1):
    for inx, (stock_name, stock_ticker, link) in enumerate(tqdm(df[['stock', 'ticker', 'link']].values)):
        try:
            full_link = f'{link}-news'
            with open(f'{news_list_path}/{stock_ticker}.txt', 'w') as file:
                for page in range(1, max_num_pages + 1):
                    full_link = f'{link}-news/{page}'
                    request = requests.get(full_link).text
                    bs4 = BeautifulSoup(request, 'html.parser')
                    news_table = bs4.find('ul', {'data-test': 'news-list'})
                    news_list = news_table.find_all('article', {'data-test': 'article-item'})
                    for news_data in news_list:
                        if str(news_data).find('mt-2.5') == -1:
                            news_link = news_data.findAll('a')[1]['href']
                            full_link = f'{BASE_URL}{news_link}'
                            file.write(f'{full_link}\n')
        except Exception as e:
            print(f'Error for stock {stock_name}: {e}')
# if the links exist, don't extract them again
if len(os.listdir(news_list_path)) == 0:
    extract_news_links(df, news_list_path, max_num_pages=MAX_NUM_PAGES)
print("finished extracting news links")

def create_dict_of_links(news_list_path):
    news_dict = {}
    for file_name in os.listdir(news_list_path):
        with open(f'{news_list_path}/{file_name}', 'r') as file:
            lines = file.readlines()
            lines = list(set(lines))
        stock_name = file_name.replace('.txt', '')
        for line in lines:
            if stock_name in news_dict:
                news_dict[stock_name].append(line.replace('\n', ''))
            else:
                news_dict[stock_name] = [line.replace('\n', '')]
    return news_dict
news_dict = create_dict_of_links(news_list_path)

def extract_news(news_dict):
    df = pd.DataFrame(columns=['stock', 'title', 'text', 'date', 'time', 'am_pm'])
    stock_list = []
    title_list = []
    date_list = []
    time_list = []
    am_pm_list = []
    text_list = []
    for inx, stock_name in enumerate(tqdm(news_dict)):
        for link in news_dict[stock_name]:
            stock_list.append(stock_name)
            request = requests.get(link).text
            bs4 = BeautifulSoup(request, 'html.parser')
            # parsing the title of the article
            try:
                header = bs4.find('h1', {'id': 'articleTitle'}).text
                title_list.append(header)
            except Exception as e:
                title_list.append(None)
                print(f'Error in parsing ""Title(header)"" in stock: {stock_name} is: {e}')
            # parsing the date and time of the article
            try:
                datetime = bs4.findAll('div', {'class': 'flex flex-row items-center'})[1].find('span').text
                datetime = datetime.replace('Published ', '')[:]
                datetime = dt.strptime(datetime, '%m/%d/%Y, %I:%M %p')
                time = datetime.strftime('%H:%M')
                date = datetime.strftime('%Y-%m-%d')
                am_pm = datetime.strftime('%p')
                date_list.append(date)
                time_list.append(time)
                am_pm_list.append(am_pm)
            except Exception as e:
                date_list.append(None)
                time_list.append(None)
                am_pm_list.append(None)
                print(f'Error in parsing ""datetime"" in stock: {stock_name} is: {e}')
                
            try:
                text = bs4.find('div', {'class': 'article_WYSIWYG__O0uhw article_articlePage__UMz3q text-[18px] leading-8'})
                all_ps = text.findAll('p')
                text = ''
                for each_p in all_ps:
                    text = text + each_p.text 
                    
                if text == '':
                    print(f'Error in parsing ""article body"" in stock: {stock_name} is: {e}')
                
                text = text.replace('Position added successfully to:', '')
                text = text.replace('\n', ' ')    
                text_list.append(text) 
            except Exception as e:
                print(f'Error in parsing ""article body"" in stock: {stock_name} is: {e}')
                text_list.append(None)
                
    df['stock'], df['title'], df['text'] = stock_list, title_list, text_list
    df['date'], df['time'], df['am_pm'] = date_list, time_list, am_pm_list   
    return df
df = extract_news(news_dict)
df.to_csv('../datasets/stock_news.csv', index=False)
