# coding=utf-8

from datetime import date, datetime, timedelta
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
connection_simulator = {'host': '*********',
                        'password': '*******',
                        'user':'*********',
                        'database':'********'}

# Дефолтные параметры, которые прокидываются в таски
default_args = {
    'owner': 'a-rybakova-8',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2022, 7, 11),
}

my_token = '*********' # токен моего бота
bot = telegram.Bot(token=my_token) # получаем доступ

chat_id = '*********'

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

#запросы в clickhouse

#кол-во действий
q_actions = """    
                   select date, actions, messages from

                   (select count(action)/countDistinct(user_id) as actions, 
                            toDate(time) as date
                   from simulator_20220620.feed_actions 
                   group by date)t1

                   JOIN 

                   (select count(user_id)/countDistinct(user_id) as messages, 
                           toDate(time) as date
                   from simulator_20220620.message_actions 
                   group by date)t2

                   using date
                   where date > today() - 15 and date != today()
"""

#stickness
q_stickness = """    
                   select days, round(day_user/month_user, 3) as Stickness
                   from 
                   
                    (select countDistinct(user_id) as day_user, 
                            toDate(time) as days
                    FROM simulator_20220620.feed_actions 
                    group by toDate(time)) t1
                    
                    join 
                    
                    (select countDistinct(user_id) as month_user, 
                            toMonth(time) as months
                    FROM simulator_20220620.feed_actions 
                    group by toMonth(time)) t2
                    
                    on toMonth(t1.days) = t2.months
                    where days > today() - 15 and days != today()
                   
"""


#Пользователи ленты и мессенджера
q_users = """    
                with get_message as 
                    (select user_id, toDate(time) as date
                    from simulator_20220620.message_actions),

                    get_feed as 
                    (select user_id, toDate(time) as date
                    from simulator_20220620.feed_actions)

                select countDistinct(user_id) as users, date
                from get_feed
                     join get_message using (user_id)
                where date > today() - 15 and date != today()
                      and user_id in 
                      (select user_id from get_message)
                group by date
                order by date desc

"""

#Пользователи ленты без мессенджера
q_users_wo = """    
                with get_message as 
                    (select user_id, toDate(time) as date
                    from simulator_20220620.message_actions),

                    get_feed as 
                    (select user_id, toDate(time) as date
                    from simulator_20220620.feed_actions)

                    select countDistinct(user_id) as user_feed, date
                    from get_feed
                    left join get_message using (user_id, date)
                    where date > today() - 15 and date != today()
                          and user_id not in 
                          (select user_id from get_message)
                    group by date
                    order by date desc
"""

#подсчитываем число сообщений
q_mes = """    select count(user_id) as messages
               from simulator_20220620.message_actions 
               where toDate(time) = yesterday()

"""

#retention
q_retention = """select toString(date) as date, 
                        toString(start_date) as start_date, 
                        count(user_id) as active_users FROM 

                (select user_id, min(toDate(time)) as start_date
                 from simulator_20220620.feed_actions 
                 group by user_id
                 having start_date >= yesterday() - 1) t1

                 JOIN 

                 (select distinct user_id, toDate(time) as date
                  from simulator_20220620.feed_actions) t2
  
                using user_id

                group by date, start_date
                order by start_date

"""

#аудиторные метрики
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


#аудиторные метрики за неделю
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
                   order by date
                   
"""

#загружаем данные
df_stickness = pandahouse.read_clickhouse(q_stickness, connection = connection_simulator)
df_users = pandahouse.read_clickhouse(q_users, connection = connection_simulator)
df_users_wo = pandahouse.read_clickhouse(q_users_wo, connection = connection_simulator)
df_mes = pandahouse.read_clickhouse(q_mes, connection = connection_simulator)
df_retention = pandahouse.read_clickhouse(q_retention, connection = connection_simulator)
df_yest = pandahouse.read_clickhouse(q_yest, connection = connection_simulator)
df_actions = pandahouse.read_clickhouse(q_actions, connection = connection_simulator)
df = pandahouse.read_clickhouse(q_week, connection = connection_simulator)

@dag(default_args=default_args, schedule_interval=schedule_interval, catchup=False)
def dag_feed_mes_report_rybakova():
    
    #таск с текстом о значении метрик за предыдущий день
    @task()
    def make_report(df_yest, df_users_wo, df_users, df_mes, df_retention):
        
        #запросы в  clickhouse

        yesterday = str(date.today() - timedelta(days=1))
        day_before_yest = str(date.today() - timedelta(days=2))

        day = yesterday

        #аудиторные метрики
        dau = df_yest.users[0]
        likes = df_yest.likes[0]
        views = df_yest.views[0]
        CTR = round(df_yest.CTR[0]*100, 2)

        #пользователи ленты без мессенджера за вчера
        users_wo = df_users_wo[df_users_wo['date'] == yesterday].user_feed[0]

        #пользователи и ленты, и мессенджера за вчера
        users = df_users[df_users['date'] == yesterday].users[0]

        #% людей, использовавших мессенджер
        ratio = round(users/users_wo*100,1)

        #число отправленных сообщений за вчера
        mes = df_mes.messages[0]

        #число новых пользователей приложении за вчера
        new = df_retention[(df_retention['start_date'] == yesterday) & \
                     ((df_retention['date'] == yesterday))].active_users.max()

        #retention
        new_day_before_yest = df_retention[(df_retention['start_date'] == day_before_yest) & \
                             ((df_retention['date'] == day_before_yest))].active_users.max()

        stayed = df_retention[(df_retention['start_date'] == day_before_yest) & \
                             ((df_retention['date'] == yesterday))].active_users.max()

        retention = round(stayed / new_day_before_yest, 3) * 100

        msg = (f'Метрики за {day}\n  \nПользователи \nDAU: {dau} \nновые: {new} \nтолько лента: {users_wo}'
              f'\nлента и сообщения: {users} \nratio: {ratio}% \nretention: {retention}% \n'
              f'\nДействия \nСообщения: {mes} \nЛайки: {likes} \nПросмотры: {views} \nCTR: {CTR}%') 
        
        bot.sendMessage(chat_id=chat_id, text=msg)
    
    #график со значениями всех метрик
    @task()
    def make_plots(df, df_actions, df_stickness, df_users, df_users_wo):
        
        sns.set_theme(style="whitegrid")
        sns.set_context("notebook",font_scale=1.5, \
                        rc={"lines.linewidth":2.0, 'lines.markersize': 7})

        fig = plt.subplots(figsize = (20, 20))
        plt.suptitle('Основные метрики работы приложения', x=0.5, y=0.94, fontsize = 20)
        plt.subplots_adjust(wspace=0.2, hspace=0.5)

        plt.subplot(4, 2, 1)
        sns.lineplot(df_actions.date, df_actions['actions'], label = 'Лента', color = 'green', marker="o")
        sns.lineplot(df_actions.date, df_actions['messages'], label = 'Мессенджер', marker="o")
        plt.legend(fontsize = 13)
        plot_object = plots(title = 'Кол-во действий на пользователя за предыдущие 10 дней', \
              y_label = 'Кол-во действий', \
              x_label = ' ')

        plt.subplot(4, 2, 2)
        sns.lineplot(df_stickness.days, df_stickness['Stickness'], color = "#D42227", marker="o")
        plot_object = plots(title = 'Stickness за предыдущие 10 дней', \
              y_label = 'DAU / MAU', \
              x_label = ' ')

        plt.subplot(4, 2, 3)
        sns.lineplot(df_users.date, df_users.users/1000, color = 'black', marker="o")
        plot_object = plots(title = 'Пользователи ленты и мессенджера за предыдущие 10 дней', \
              y_label = 'тыс. пользователей', \
              x_label = ' ')

        plt.subplot(4, 2, 4)
        sns.lineplot(df_users_wo.date, df_users_wo.user_feed/1000, marker="o", color = "#0188A8")
        plot_object = plots(title = 'Пользователи ленты без мессенджера за предыдущие 10 дней', \
              y_label = 'тыс. пользователей', \
              x_label = ' ')

        plt.subplot(4, 2, 5)
        sns.lineplot(df.date, df.users/1000, color = "#D42227", marker="o")
        plot_object = plots(title = 'DAU за предыдущие 10 дней', \
              y_label = 'тыс. пользователей', \
              x_label = ' ')

        plt.subplot(4, 2, 6)
        sns.lineplot(df.date, df.likes/1000, color = 'green', marker="o")
        plot_object = plots(title = 'Лайки за предыдущие 10 дней', \
              y_label = 'тыс. лайков', \
              x_label = ' ')

        plt.subplot(4, 2, 7)
        sns.lineplot(df.date, df.views/1000, color = 'black', marker="o")
        plot_object = plots(title = 'Просмотры за предыдущие 10 дней', \
              y_label = 'тыс. просмотров', \
              x_label = 'Дата')

        plt.subplot(4, 2, 8)
        sns.lineplot(df.date, df.CTR, marker="o", color = "#0188A8")
        plot_object = plots(title = 'CTR за предыдущие 10 дней', \
              y_label = 'CTR', \
              x_label = 'Дата')

        plt.close()

        bot.sendPhoto(chat_id=chat_id, photo=plot_object)
              
    make_report(df_yest, df_users_wo, df_users, df_mes, df_retention)
    make_plots(df, df_actions, df_stickness, df_users, df_users_wo)
    
dag_feed_mes_report_rybakova = dag_feed_mes_report_rybakova()