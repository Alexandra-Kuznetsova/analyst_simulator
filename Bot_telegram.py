# coding=utf-8

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from io import StringIO
import requests
import pandahouse
import telegram
import io
import seaborn as sns
import matplotlib.pyplot as plt

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

# Функция для CH
connection_simulator = {'host': '********',
                        'password': '*******',
                        'user':'******',
                        'database':'********'}

# Дефолтные параметры, которые прокидываются в таски
default_args = {
    'owner': 'a-rybakova-8',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2022, 7, 11),
}

my_token = '*************' # токен моего бота
bot = telegram.Bot(token=my_token) # получаем доступ

chat_id = '***********'  #id чата

# Интервал запуска DAG
schedule_interval = '0 11 * * *'

#функция для графиков
def plots(title, y_label, x_label):
    plt.title(title, pad = 15, fontsize = 15)
    plt.ylabel(y_label, fontsize = 15)
    plt.xlabel(x_label, fontsize = 15)
    plt.xticks(rotation = 30, fontsize = 10)
    plt.yticks(fontsize = 15)
    plot_object = io.BytesIO()
    plt.savefig(plot_object)
    plot_object.seek(0)
    plot_object.name = f'title.png'
    return plot_object

@dag(default_args=default_args, schedule_interval=schedule_interval, catchup=False)
def dag_report_rybakova():
    
    #таск с текстом о значении метрик за предыдущий день
    @task()
    def make_report():
        q_yest = """
                   select 
                        toDate(time) as date,
                        countDistinct(user_id) as users,
                        countIf(action = 'like') as likes, 
                        countIf(action = 'view') as views,
                        likes/views as CTR
                   from 
                        simulator_20220620.feed_actions 
                   where toDate(time) = yesterday()
                   group by date
                   
"""

        df = pandahouse.read_clickhouse(q_yest, connection = connection_simulator)

        day = df.date[0].date()
        dau = df.users[0]
        likes = df.likes[0]
        views = df.views[0]
        CTR = round(df.CTR[0], 2)

        msg = f'Метрики за {day}: \nDAU: {dau} \nЛайки: {likes} \nПросмотры: {views} \nCTR: {CTR}'
        bot.sendMessage(chat_id=chat_id, text=msg)
    
    #график со значениями метрик за предудщие 7 дней
    @task()
    def make_plots():
        
        #выгружаем данные
        q_week = """
                   select 
                        toDate(time) as date,
                        countDistinct(user_id) as users,
                        countIf(action = 'like') as likes, 
                        countIf(action = 'view') as views,
                        likes/views as CTR
                   from 
                        simulator_20220620.feed_actions 
                   where toDate(time) >= today() - 7 and toDate(time) != today()
                   group by date
                   
                  """
        
        df = pandahouse.read_clickhouse(q_week, connection = connection_simulator)
        
        #графики
        sns.set_theme(style="whitegrid")
        fig = plt.subplots(figsize = (15, 15))

        plt.suptitle('Основные метрики работы приложения', x=0.5, y=0.94, fontsize = 20)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        plt.subplot(2, 2, 1)
        sns.lineplot(df.date, df.users/1000, color = 'red', marker="o")
        plot_object = plots(title = 'DAU за предыдущие 7 дней', \
            y_label = 'тыс. пользователей', \
            x_label = ' ')

        plt.subplot(2, 2, 2)
        sns.lineplot(df.date, df.likes/1000, color = 'green', marker="o")
        plot_object = plots(title = 'Лайки за предыдущие 7 дней', \
            y_label = 'тыс. лайков', \
            x_label = ' ')

        plt.subplot(2, 2, 3)
        sns.lineplot(df.date, df.views/1000, color = 'black', marker="o")
        plot_object = plots(title = 'Просмотры за предыдущие 7 дней', \
            y_label = 'тыс. просмотров', \
            x_label = 'Дата')

        plt.subplot(2, 2, 4)
        sns.lineplot(df.date, df.CTR, marker="o")
        current_values = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['{:,.3f}'.format(x) for x in current_values])
        plot_object = plots(title = 'CTR за предыдущие 7 дней', \
            y_label = 'CTR', \
            x_label = 'Дата')
       
        plt.close()

        bot.sendPhoto(chat_id=chat_id, photo=plot_object)
    
    
    make_report()
    make_plots()
    
dag_report_rybakova = dag_report_rybakova()
