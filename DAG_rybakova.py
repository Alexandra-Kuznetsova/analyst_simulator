# coding=utf-8

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from io import StringIO
import requests
import pandahouse

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context


# Функция для CH
connection_simulator = {'host': '*******',
                        'password': '********',
                        'user':'*********',
                        'database':'***********'}

#задаем параметры для записи итоговой таблицы в Clickhouse
connection_to = {'host':'*********', 
                           'database':'*******', 
                           'user': '********', 
                           'password':'********'} 

# Дефолтные параметры, которые прокидываются в таски
default_args = {
    'owner': 'a-rybakova-8',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2022, 7, 7),
}

def ch_get_df(query='Select 1', connection = connection_simulator):
    df = pandahouse.read_clickhouse(query, connection = connection)
    return df

# Интервал запуска DAG
schedule_interval = '0 11 * * *'

@dag(default_args=default_args, schedule_interval=schedule_interval, catchup=False)
def dag_a_rybakova():

    @task()
    def extract_feed():
        query = """select 
                        toDate(time) as event_date,
                        user_id, gender, age, os,
                        countIf(action = 'like') as likes, 
                        countIf(action = 'view') as views
                   from 
                        simulator_20220620.feed_actions 
                   where toDate(time) = yesterday()
                   group by event_date, user_id, gender, age, os
                   """
        df_cube_feed = ch_get_df(query=query)
        return df_cube_feed
    
    @task()
    def extract_messages():
        query = """with get_received as
                         (select toDate(time) as event_date, 
                                 reciever_id,
                                 count(user_id) as messages_received, 
                                 countDistinct(user_id) as users_sent
                          from simulator_20220620.message_actions
                          where toDate(time) = yesterday()
                          group by reciever_id, event_date 
                          order by reciever_id),

                          get_sent AS 
                               (select toDate(time) as event_date, 
                                       user_id,  
                                       count(reciever_id) as messages_sent, 
                                       countDistinct(reciever_id) as users_received
                                       from simulator_20220620.message_actions
                               where toDate(time) = yesterday()
                               group by event_date, user_id
                               order by user_id)


                    select event_date, user_id, 
                           messages_received, 
                           messages_sent, 
                           users_received, 
                           users_sent
                    from 
                        get_received join get_sent on get_received.reciever_id = get_sent.user_id 
                                                   and get_received.event_date = get_sent.event_date
                                 
                   """
        df_cube_messages = ch_get_df(query=query)
        return df_cube_messages

    @task
    def transfrom_merge(df_cube_feed, df_cube_messages):
        df_cube = df_cube_feed.merge(df_cube_messages, on = ['user_id', 'event_date'], how = 'outer')
        
        df_cube['age_grouped'] = np.where(df_cube['age']<=20, '10-20',
                    np.where((df_cube['age']>20) & (df_cube['age']<=30), '20-30',  
                    np.where((df_cube['age']>30) & (df_cube['age']<=40), '30-40',
                    np.where((df_cube['age']>40) & (df_cube['age']<=50), '40-50', 
                    np.where((df_cube['age']>50) & (df_cube['age']<=100), '50+',
                    pd.NaT)))))

        df_cube['metrics'] = np.nan
        df_cube['slice'] = np.nan
        
        return df_cube
    
    #Для этой таблицы считаем все эти метрики в разрезе по полу, возрасту и ос. Делаем три разных таска на каждый срез.
    
    @task
    def transfrom_gender(df_cube):
        df_cube_gender = df_cube[['event_date', 'metrics', 'slice', 'gender', 'likes','views',\
                                  'messages_received', 'messages_sent', 'users_received', 'users_sent']]\
            .groupby(['event_date', 'gender'])\
            .sum()\
            .reset_index()

        df_cube_gender.metrics = 'gender'
        df_cube_gender.slice = df_cube_gender['gender']

        return df_cube_gender
    
    @task
    def transfrom_age(df_cube):

        df_cube_age = df_cube[['event_date', 'metrics', 'slice', 'age_grouped', 'likes','views',\
                               'messages_received', 'messages_sent', 'users_received', 'users_sent']]\
            .groupby(['event_date', 'age_grouped'])\
            .sum()\
            .reset_index()

        df_cube_age.metrics = 'age'
        df_cube_age.slice = df_cube_age['age_grouped']
        df_cube_age = df_cube_age.dropna()

        return df_cube_age
    
    
    @task
    def transfrom_os(df_cube):
        df_cube_os = df_cube[['event_date', 'metrics', 'slice','os','likes','views',\
                              'messages_received', 'messages_sent', 'users_received', 'users_sent']]\
            .groupby(['event_date', 'os'])\
            .sum()\
            .reset_index()

        df_cube_os.metrics = 'os'
        df_cube_os.slice = df_cube_os['os']

        return df_cube_os

    @task
    def load(df_cube_gender, df_cube_age, df_cube_os, connection_to):
        
        #текст запроса в Clickhouse для создания таблицы
        
        query_result = """create table if not exists test.arybakova_test1_etl
        (event_date Date,
         metrics String,
         slice String,
         likes UInt64,
         views UInt64,
         messages_received UInt64,
         messages_sent UInt64,
         users_received UInt64,
         users_sent UInt64) 
         Engine = MergeTree()
         order by event_date
        """
        
        df_final = pd.concat([df_cube_gender, df_cube_age, df_cube_os], ignore_index = True)
        df_final = df_final.drop(columns={'gender', 'age_grouped', 'os'})

        df_final['event_date'] = pd.to_datetime(df_final['event_date'])
        df_final = df_final.astype({'metrics':'object', 'slice':'object', 'likes': int, 'views': int, 'messages_received':int, \
               'messages_sent': int, 'users_received': int, 'users_sent':int})
        
        #сохраняем таблицу в Clickhouse
        pandahouse.execute(query = query_result, connection = connection_to)
        pandahouse.to_clickhouse(df_final, 'arybakova_test1_etl', index = False, connection = connection_to)



    df_cube_feed = extract_feed()    
    df_cube_messages = extract_messages()
    
    df_cube = transfrom_merge(df_cube_feed, df_cube_messages)
    df_cube_gender = transfrom_gender(df_cube)
    df_cube_age = transfrom_age(df_cube)
    df_cube_os = transfrom_os(df_cube)
    
    load(df_cube_gender, df_cube_age, df_cube_os, connection_to)

dag_a_rybakova = dag_a_rybakova()
