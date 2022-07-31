import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from CH import Getch
import io
from datetime import date
import telegram
import pandahouse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Функция для CH
connection_simulator = {'host': '********',
                        'password': '********',
                        'user':'*******',
                        'database':'********'}

my_token = '*********'

chat_id = '*******'
bot = telegram.Bot(token=my_token)

def check_anomaly(df, metric, group, threshold = 0.3):
    
    df = df[df['os']== group]
    
    #сравниваем метрику со значением сутки назад
    current_ts = df['ts'].max() #max 15 минутка из df
    day_ago_ts = current_ts - pd.DateOffset(days = 1)  #та же 15-ти минутка, но сутки назад
    
    current_value = df[(df['ts'] == current_ts)][metric].iloc[0] #значение метрики
    day_ago_value = df[(df['ts'] == day_ago_ts)][metric].iloc[0]
    
    #вычисляем отклонение
    if current_value <= day_ago_value: 
        diff = abs(current_value / day_ago_value - 1)
    else:
        diff = abs(day_ago_value / current_value - 1)
        
        
    #сравниваем с пороговым значением
    if diff > threshold:
        is_alert = 1
    else:
        is_alert = 0
        
    return is_alert, current_value, round(diff*100, 2), df


def check_anomaly_iq(df, metric, group, threshold = 0.3):
    #межквартильный размах
    df = df[df['os']== group]
    
    #сравниваем метрику со значением сутки назад
    current_ts = df['ts'].max() #max 15 минутка из df
    two_hours_ago_ts = current_ts - timedelta(hours=2)  #та же 15-ти минутка, но 2 часа назад
    
    current_value = df[(df['ts'] == current_ts)][metric].iloc[0] #значение метрики
    two_hours_ago_value = df[(df['ts'] == two_hours_ago_ts)][metric].iloc[0]
    
    #нахождение персентилей
    q25 = df[(df['ts'] <= current_ts) & (df['ts'] >= two_hours_ago_ts)]['users_lenta'].quantile(q = 0.25)
    q75 = df[(df['ts'] <= current_ts) & (df['ts'] >= two_hours_ago_ts)]['users_lenta'].quantile(q = 0.75)
    
    #коэфф. определяющий ширину интервала
    a = 1.5
    
    #межквартильный размах
    IQR = q75 - q25
    
    #вычисляем отклонение
    if current_value <= two_hours_ago_value: 
        diff = abs(current_value / two_hours_ago_value - 1)
    else:
        diff = abs(two_hours_ago_value / current_value - 1)
    
    #вычисляем отклонение
    if (current_value < (q25 - a*IQR)) | (current_value > (q75 + a*IQR)): 
        is_alert = 1
    else:
        is_alert = 0
        
    return is_alert, current_value, round(diff*100, 2), df


def check_anomaly_rolling(df, metric, group, threshold = 0.3):
    #простое сглаживание
    #считаем скользящее окно
    df = df[df['os']== group]
    df['rolling'] = df[metric].rolling(window=6).mean().fillna(df[metric].mean())

    current_ts = df['ts'].max()  #max 15 минутка из df
    
    current_value = df[(df['ts'] == current_ts)][metric].iloc[0]
    predicted_value = df[(df['ts'] == current_ts)]['rolling'].iloc[0]

    if current_value <= predicted_value: 
        diff = abs(current_value / predicted_value - 1)
    else:
        diff = abs(predicted_value / current_value - 1)
    
    #сравниваем отклонение с пороговым значением
    if diff > threshold:
        is_alert = 1
    else:
        is_alert = 0
        
    return is_alert, current_value, round(diff*100, 2), df

#### экспоненциальное сглаживание

def exponential_smoothing(series, alpha):
    result = series.iloc[:] # first value is same as series
    for n in range(1, len(series)):
        result.iloc[n] = (alpha * series.iloc[n] + (1 - alpha) * result.iloc[n-1])
    return result

def check_anomaly_exponential(df, metric, group, threshold = 0.3):
    
    df = df[df['os']== group]
    df['exp'] = exponential_smoothing(df[metric], alpha = 0.4)
    current_ts = df['ts'].max() #max 15 минутка из df
    
    current_value = df[(df['ts'] == current_ts)][metric].iloc[0] #значение метрики
    predicted_value = df[(df['ts'] == current_ts)]['exp'].iloc[0]
    
    #вычисляем отклонение
    if current_value <= predicted_value: 
        diff = abs(current_value / predicted_value - 1)
    else:
        diff = abs(predicted_value / current_value - 1)
    
    #сравниваем отклонение с пороговым значением
    if diff > threshold:
        is_alert = 1
    else:
        is_alert = 0
        
    return is_alert, current_value, round(diff*100, 2), df

###linear regression

def code_mean(data, cat_feature, real_feature):
    """
    Возвращает словарь, где ключами являются уникальные категории признака cat_feature, 
    а значениями - средние по real_feature
    """
    return dict(data.groupby(cat_feature)[real_feature].mean())

#функция для создания переменных

def prepareData(data, lag_start=5, lag_end=20, test_size=0.15):

    data = pd.DataFrame(data[data['os']==group].copy())
    data.columns = ['ts', 'date', 'hm', 'y', 'os']

    # считаем индекс в датафрейме, после которого начинается тестовыый отрезок
    test_index = int(len(data)*(1-test_size))

    # добавляем лаги исходного ряда в качестве признаков
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.y.shift(i)

    #data['ts'] = data['ts'].to_datetime()
    data['ts'] = pd.to_datetime(data['ts'])
    data['hour'] = data['ts'].dt.hour
    data['weekday'] = data['ts'].dt.weekday
    data['is_weekend'] = data['ts'].dt.weekday.isin([5,6])*1

    # считаем средние только по тренировочной части, чтобы избежать лика
    data['weekday_average'] = list(map(code_mean(data[:test_index], 'weekday', "y").get, data.weekday))
    data["hour_average"] = list(map(code_mean(data[:test_index], 'hour', "y").get, data.hour))

    # выкидываем закодированные средними признаки 
    data.drop(["hour", "weekday"], axis=1, inplace=True)

    data = data.dropna()
    data = data.drop(columns = ['ts', 'hm', 'os'])
    data = data.reset_index(drop=True)
    data['date'] = pd.to_numeric(pd.to_datetime(data['date']))

    # разбиваем весь датасет на тренировочную и тестовую выборку
    X_train = data.loc[:test_index].drop(["y"], axis=1)
    y_train = data.loc[:test_index]["y"]
    X_test = data.loc[test_index:].drop(["y"], axis=1)
    y_test = data.loc[test_index:]["y"]

    return X_train, X_test, y_train, y_test

def check_anomaly_lr(df, metric, group, threshold = 0.3):
    df = df[df['os']==group]
    X_train, X_test, y_train, y_test = prepareData(df, test_size=0.3, lag_start=12, lag_end=48)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    prediction = lr.predict(X_test)
    
    #вычисляем отклонение
    if y_test.values[-1] <= prediction[-1]: 
        diff = abs(y_test.values[-1] / prediction[-1] - 1)
    else:
        diff = abs(prediction[-1] / y_test.values[-1] - 1)
    
    #сравниваем отклонение с пороговым значением
    if diff > threshold:
        is_alert = 1
    else:
        is_alert = 0
        
    return is_alert, current_value, round(diff*100, 2), df

### xgboost forecast
def check_anomaly_xgb(df, metric, group, threshold = 0.3):
    
    df = df[df['os']==group]
    X_train, X_test, y_train, y_test = prepareData(df, lag_start=5, lag_end=20, test_size=0.15)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    
    # задаём параметры
    params = {
        'objective': 'reg:linear',
        'booster':'gblinear'
    }
    trees = 1000

    # прогоняем на кросс-валидации с метрикой rmse
    cv = xgb.cv(params, dtrain, metrics = ('rmse'), verbose_eval=False, nfold=10, show_stdv=False, num_boost_round=trees)

    # обучаем xgboost с оптимальным числом деревьев, подобранным на кросс-валидации
    bst = xgb.train(params, dtrain, num_boost_round=cv['test-rmse-mean'].argmin())

    # запоминаем ошибку на кросс-валидации
    deviation = cv.loc[cv['test-rmse-mean'].argmin()]["test-rmse-mean"]

    # посмотрим, как модель вела себя на тренировочном отрезке ряда
    prediction_train = bst.predict(dtrain)
    
    # и на тестовом
    prediction_test = bst.predict(dtest)
    
    scale=1.96
    
    lower = prediction_test-scale*deviation
    upper = prediction_test+scale*deviation

    Anomalies = np.array([np.NaN]*len(y_test))
    Anomalies[y_test<lower] = y_test[y_test<lower]
    Anomalies[y_test>upper] = y_test[y_test>upper]
    
    #сравниваем отклонение с пороговым значением
    if np.isnan(Anomalies[-1]):
        is_alert = 0
    else:
        is_alert = 1
        
    return is_alert, current_value, deviation, df

def alert(metric, group, data, function = check_anomaly): 
    
    if function == check_anomaly:
        is_alert, current_value, diff, df = check_anomaly(data, metric, group, threshold = 0.1)
        
    elif function == check_anomaly_iq:
        is_alert, current_value, diff, df = check_anomaly_iq(data, metric, group, threshold = 0.1)
        
    elif function == check_anomaly_rolling:
        is_alert, current_value, diff, df = check_anomaly_rolling(data, metric, group, threshold = 0.1)
        
    elif function == check_anomaly_exponential:
        is_alert, current_value, diff, df = check_anomaly_exponential(data, metric, group, threshold = 0.1)
        
    elif function == check_anomaly_lr:
        is_alert, current_value, diff, df = check_anomaly_lr(data, metric, group, threshold = 0.1)
        
    elif function == check_anomaly_xgb:
        is_alert, current_value, diff, df = check_anomaly_xgb(data, metric, group, threshold = 0.1)
           
    if is_alert:
        msg = f'''Метрика {metric} в срезе {group}. Значение {current_value} в {data.ts.max()}. Отклонение более {diff}%.
        '''
        sns.set(rc = {'figure.figsize': (16, 10)})
        plt.tight_layout()

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        
        l1 = sns.lineplot(x=df.ts.dt.hour,y = df[metric],color="#0188A8", ax=ax2)
        
        if function == check_anomaly_rolling:
            l2 = sns.lineplot(x=df.ts.dt.hour, y = df['rolling'],color="#D42227", ax=ax2)
        
        elif function == check_anomaly_exponential:
            l2 = sns.lineplot(x=df.ts.dt.hour, y = df['exp'],color="#D42227", ax=ax2)
            
        elif function == check_anomaly_lr:
            plt.plot(prediction, "r", label="prediction")
            plt.plot(y_test.values, label="actual")
            plt.legend()
            
        elif function == check_anomaly_xgb:
            plt.plot(prediction_test, label="prediction")
            plt.plot(lower, "r--", label="upper bond / lower bond")
            plt.plot(upper, "r--")
            plt.plot(list(y_test), label="y_test")
            plt.plot(Anomalies, "ro", markersize=10)
            plt.legend(loc="best")
            plt.axis('tight')
               
        
        plt.xticks(rotation = 30, fontsize = 8)
        
        for ind, label in enumerate(ax.xaxis.get_ticklabels()):
            if ind % 15 == 0:
                label.set_visible(False)
                plt.xticks(rotation = 30, fontsize = 8)
        
        ax1.set(xlabel = 'time')
        ax1.set(ylabel = metric)

        ax1.set_title('{}'.format(metric))
        ax1.set(ylim = (0, None))

        plot_object = io.BytesIO()
        ax1.figure.savefig(plot_object) 

        plot_object.seek(0)
        plot_object.name = '{0}.png'.format(metric)
        plt.close()
        
        bot.sendMessage(chat_id=chat_id, text=msg)
        bot.sendPhoto(chat_id=chat_id, photo = plot_object)
    
    try:
        run_alerts()
    except Exception as e:
        print(e)

    try:
        run_alerts()
    except Exception as e:
        print(e)
        
        
def run_alerts(chat = None):
    chat_id = chat or 456265830
    bot = telegram.Bot(token=my_token)
    
    
# Интервал запуска DAG
schedule_interval = '*/15 * * * *' #каждые 15 минут

# Дефолтные параметры, которые прокидываются в таски
default_args = {
    'owner': 'a-rybakova-8',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2022, 7, 11),
}

q = '''with get_feed as
                (select 
                    toStartOfFifteenMinutes(time) as ts,
                    toDate(ts) as date,
                    formatDateTime(ts, '%R') as hm,
                    uniqExact(user_id) as users_lenta, 
                    countIf(action = 'like') as likes, 
                    countIf(action = 'view') as views,
                    round(likes/views, 3) as CTR,
                    os
                from simulator_20220620.feed_actions
                where ts >= today() - 1 and ts < toStartOfFifteenMinutes(now())
                group by ts, date, hm, os
                order by ts), 
                
                get_mes as
                (select 
                    toStartOfFifteenMinutes(time) as ts,
                    toDate(ts) as date,
                    formatDateTime(ts, '%R') as hm,
                    count(user_id) as messages,
                    uniqExact(user_id) as users_mes,
                    os
                from simulator_20220620.message_actions
                where ts >= today() - 1 and ts < toStartOfFifteenMinutes(now())
                group by ts, date, hm, os
                order by ts)
                
                select * 
                from get_mes
                inner join get_feed
                using (ts, date, hm, os)
'''

@dag(default_args=default_args, schedule_interval=schedule_interval, catchup=False)
def dag_anomaly_rybakova():
    
    df = pandahouse.read_clickhouse(q, connection = connection_simulator)
    
    #проверяем все метрики
    @task()
    def make_report():
        #функцию для детектирования аномалию можно выбрать любую из пяти
        #активные пользователи в ленте
        alert(metric = 'users_lenta', group = 'iOS', data = df, function = check_anomaly)
        alert(metric = 'users_lenta', group = 'Android', data = df, function = check_anomaly)
        
        #активные пользователи в мессенджере
        alert(metric = 'users_mes', group = 'iOS', data = df, function = check_anomaly)
        alert(metric = 'users_mes', group = 'Android', data = df, function = check_anomaly)
        
        #лайки
        alert(metric = 'likes', group = 'iOS', data = df, function = check_anomaly)
        alert(metric = 'likes', group = 'Android', data = df, function = check_anomaly)
    
        #просмотры
        alert(metric = 'views', group = 'iOS', data = df, function = check_anomaly)
        alert(metric = 'views', group = 'Android', data = df, function = check_anomaly)
        
        #CTR
        alert(metric = 'CTR', group = 'iOS', data = df, function = check_anomaly)
        alert(metric = 'CTR', group = 'Android', data = df, function = check_anomaly)
        
        #таск количество отправленных сообщений
        alert(metric = 'messages', group = 'iOS', data = df, function = check_anomaly)
        alert(metric = 'messages', group = 'Android', data = df, function = check_anomaly)
        
    
    make_report()
    
dag_anomaly_rybakova = dag_anomaly_rybakova()