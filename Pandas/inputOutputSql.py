import numpy as np
import pandas as pd
from sqlalchemy import create_engine

res = ''

#Создание временной базы данных
temp_db = create_engine('sqlite:///:memory:')

#Генерация данных для df
df = pd.DataFrame(data=np.random.randint(low=0,high=100,size = (4,4)),columns=['a','b','c','d'])
res = df

#Добавляем таблицу в базу данных
df.to_sql(name='new_table',con=temp_db)

#Чтение таблицы целиком
new_df = pd.read_sql(sql='new_table',con=temp_db)
res = new_df

#Чтение через запрос
new_df = pd.read_sql_query(sql='SELECT a,c FROM new_table',con=temp_db)
res = new_df


print(res)