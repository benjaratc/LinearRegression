#!/usr/bin/env python
# coding: utf-8

# In[139]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# 1. โหลด csv เข้าไปใน Python Pandas

# In[140]:


df = pd.read_csv('../Desktop/DataCamp/USA_housing.csv')
df


# 2. เขียนโค้ดแสดง หัว10แถว ท้าย10แถว และสุ่ม10แถว

# In[141]:


df.head(10)


# In[142]:


df.tail(10)


# In[143]:


df.sample(10)


# 3. เช็คว่ามีข้อมูลที่หายไปไหม สามารถจัดการได้ตามความเหมาะสม

# In[144]:


df.isnull().any()


# 4. ใช้ info และ describe อธิบายข้อมูลเบื้องต้น

# In[145]:


df.info()


# In[146]:


df.describe()


# 5. ใช้ pairplot ดูความสัมพันธ์เบื้องต้น

# In[147]:


sns.pairplot(df)


# 6. ใช้ displot เพื่อดูการกระจายของแต่ละคอลัมน์

# In[148]:


sns.distplot(df['Avg. Area Income'])


# In[149]:


sns.distplot(df['Price'])


# In[150]:


sns.distplot(df['Avg. Area House Age'])


# In[151]:


sns.distplot(df['Avg. Area Number of Rooms'])


# In[152]:


sns.distplot(df['Avg. Area Number of Bedrooms'])


# In[153]:


sns.distplot(df['Area Population'])


# 7. ใช้ heatmap ดูความสัมพันธ์ของคอลัมน์ที่สนใจ

# In[154]:


sns.heatmap(df.corr())


# 8. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation สูงสุด

# In[155]:


sns.scatterplot(data = df, y = 'Price', x = 'Avg. Area Income')


# 9. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation ต่ำสุด

# In[156]:


sns.scatterplot(data = df, y = 'Avg. Area House Age', x = 'Avg. Area Number of Bedrooms')


# 10. สร้าง histogram ของ price

# In[157]:


sns.distplot(df['Price'])


# 11. สร้าง box plot ของราคา

# In[158]:


sns.boxplot(df['Price'], orient = 'v')


# 12. สร้าง train/test split ของบ้าน สามารถลองทดสอบ 70:30, 80:20, 90:10 ratio ได้ตามใจชอบ

# In[159]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[160]:


x = df['Avg. Area Income']
y = df['Price']


# In[161]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[162]:


x_train = np.array(x_train).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)


# In[163]:


lm = LinearRegression()
lm.fit(x_train,y_train)


# In[164]:


print(lm.intercept_)
print(lm.coef_)


# In[165]:


predicted = lm.predict(x_test)
predicted


# 14. ทดสอบโมเดลวัดค่า MAE, MSE, RMSE

# In[166]:


print('MAE', metrics.mean_absolute_error(y_test,predicted))
print('MSE', metrics.mean_squared_error(y_test,predicted))
RMSE1 = print('RMSE', np.sqrt(metrics.mean_squared_error(y_test,predicted)))
RMSE1


# 15. สร้าง distribution plot เพื่อดูว่า predicted results เป็น normal ไหม

# In[167]:


sns.distplot(y_test-predicted)


# 16. สร้าง scatter plot และ prediction line ของ simple linear regression

# In[168]:


fig = plt.figure(figsize = (12,8))
plt.scatter(x_test,y_test,color = 'blue', label = 'real price')
plt.plot(x_test,predicted,color = 'red', label = 'Linear regression price')
plt.xlabel('income')
plt.ylabel('Housing price')
plt.title('the relationship between income and housing price')
plt.legend()


# 17. เทรนโมเดลแบบ Multiple Linear Regression 

# In[169]:


x1 = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms',
        'Area Population','Avg. Area Number of Bedrooms']]
y1 = df['Price']


# In[170]:


x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.3, random_state = 100)


# In[171]:


lm1 = LinearRegression()
lm1.fit(x1_train,y1_train)


# In[172]:


predicted_1 = lm1.predict(x1_test)
predicted_1 


# In[173]:


print(lm1.intercept_)
print(lm1.coef_)


# 18. ทดสอบโมเดลวัดค่า MAE, MSE, RMSE

# In[174]:


print('MAE', metrics.mean_absolute_error(y1_test,predicted_1))
print('MSE', metrics.mean_squared_error(y1_test,predicted_1))
RMSE2 = print('RMSE', np.sqrt(metrics.mean_squared_error(y1_test,predicted_1)))
RMSE2


# 19. สร้าง distribution plot เพื่อดูว่า predicted results เป็น normal ไหม

# In[175]:


sns.distplot(y_test-predicted)


# 20. ค่า RMSE ของ All-Features Multi Linear Regression มากกว่าหรือน้อย ค่า RMSE จากคู่ที่ดีที่สุดของ Simple Linear Regression มากกว่า/น้อยกว่า เท่าใด

# RMSE ของ All-Features Multi Linear Regression หรือน้อย ค่า RMSE จากคู่ที่ดีที่สุดของ Simple Linear Regression =  271707.51-101688.39 = 170019.12
