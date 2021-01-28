#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 2개의 peak만 먼저 해보자
import numpy as np

data_number=5000000

peak_2_graph1 = np.zeros((data_number,3))
peak_2_graph2 = np.zeros((data_number,3))

peak_3_graph1 = np.zeros((data_number,3))
peak_3_graph2 = np.zeros((data_number,3))
peak_3_graph3 = np.zeros((data_number,3))


# In[2]:


#a= center, b = width, c= amp

x = np.linspace(0, 15, 401) 

def y(a,b,c,x):
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    y = c * ( (0.7*np.exp(-np.log(2)*(x - a)**2 / (beta * b)**2)) + (0.3/(1 + (x - a)**2 / (gamma * b)**2)))
#     y = c*(b**2)/((x-a)**2+b**2)
    return y

    # (center, width, amplitude)


# In[3]:


import random #(a,b,c 만드는 작업)


for i in range(data_number):

    peak_2_graph1[i] = [2+11.0*np.random.rand(),0.3+1.6*np.random.rand(),0.1+np.random.rand()]
    peak_2_graph2[i] = [2+11.0*np.random.rand(),0.3+1.6*np.random.rand(),0.1+np.random.rand()]

    peak_3_graph1[i] = [2+11.0*np.random.rand(),0.3+1.6*np.random.rand(),0.1+np.random.rand()]
    peak_3_graph2[i] = [2+11.0*np.random.rand(),0.3+1.6*np.random.rand(),0.1+np.random.rand()]
    peak_3_graph3[i] = [2+11.0*np.random.rand(),0.3+1.6*np.random.rand(),0.1+np.random.rand()]


# In[4]:


print(peak_2_graph1.shape)
print(peak_2_graph2.shape)
print(peak_3_graph1.shape)
print(peak_3_graph2.shape)
print(peak_3_graph3.shape)


# In[5]:


peak_2_graph1[7]


# In[6]:


peak_2_graph1_show = []
peak_2_graph2_show = []
peak_3_graph1_show = []
peak_3_graph2_show = []
peak_3_graph3_show = []

noise_level = 0.05
for i in range(data_number):
    
    
    noise1 = []
    noise2 = []
    noise3 = []
    noise4 = []
    noise5 = []
    for j in range(len(x)):
        noise1.append(np.random.rand()*noise_level-noise_level*0.5)
        noise2.append(np.random.rand()*noise_level-noise_level*0.5)
        noise3.append(np.random.rand()*noise_level-noise_level*0.5)
        noise4.append(np.random.rand()*noise_level-noise_level*0.5)
        noise5.append(np.random.rand()*noise_level-noise_level*0.5)

    peak_2_graph1_show.append(
        y(peak_2_graph1[i][0],peak_2_graph1[i][1],peak_2_graph1[i][2],x)+np.array(noise1))
    peak_2_graph2_show.append(
        y(peak_2_graph2[i][0],peak_2_graph2[i][1],peak_2_graph2[i][2],x)+np.array(noise2))
    
    peak_3_graph1_show.append(
        y(peak_3_graph1[i][0],peak_3_graph1[i][1],peak_3_graph1[i][2],x)+np.array(noise3))
    peak_3_graph2_show.append(
        y(peak_3_graph2[i][0],peak_3_graph2[i][1],peak_3_graph2[i][2],x)+np.array(noise4))
    peak_3_graph3_show.append(
        y(peak_3_graph3[i][0],peak_3_graph3[i][1],peak_3_graph3[i][2],x)+np.array(noise5))


# In[7]:


# peak 2의 각각의 그래프
import matplotlib.pyplot as plt

for i in range(10):

    plt.figure(figsize =(10,4))
    plt.plot(x,peak_2_graph1_show[i],c = 'r')
    plt.plot(x,peak_2_graph2_show[i],c='r')
    plt.plot(x,peak_2_graph1_show[i]+peak_2_graph2_show[i],c='b')
    plt.ylim(0,3)
    plt.grid(True)
    plt.show
    # noise_level = 0.05
    # noise = np.array([np.random.rand()*noise_level - noise_level*0.5 for i in range(len(x))]


# In[8]:


# peak 3의 각각의 그래프
import matplotlib.pyplot as plt

for i in range(10):

    plt.figure(figsize =(10,4))
    plt.plot(x,peak_3_graph1_show[i],c = 'r')
    plt.plot(x,peak_3_graph2_show[i],c='r')
    plt.plot(x,peak_3_graph3_show[i],c='r')
    plt.plot(x,peak_3_graph1_show[i]+peak_3_graph2_show[i]+peak_3_graph3_show[i],c='b')
    plt.ylim(0,3)
    plt.grid(True)
    plt.show


# In[9]:


# peak2 ,peak3 각각 noise있는 그래프 2개씩,3개씩 더해줌

peak2 = []
for i in range(data_number):
    peak2.append(np.array(peak_2_graph1_show[i])+np.array(peak_2_graph2_show[i]))

peak3 = []
for j in range(data_number):
    peak3.append(np.array(peak_3_graph1_show[j])+np.array(peak_3_graph2_show[j])+np.array(peak_3_graph3_show[j]))


# In[10]:


# center, width,amp 만들고 peak갯수까지 feature 사용하자
# peak 갯수 만들어서 zip 써도 된다
# [0,0,0,0,0,0,0,0,0]꼴 만들기

peak2_params = []
peak3_params = []
for i in range(data_number):

    peak2_params.append(list(peak_2_graph1[i])+list(peak_2_graph2[i])+3*[0]+[2])
    peak3_params.append(list(peak_3_graph1[i])+list(peak_3_graph2[i])+list(peak_3_graph3[i])+[3])


# In[11]:


# 확인하기 
print(np.array(peak2_params).shape)
print(np.array(peak2).shape)
print(len(peak2+peak3))
print(len(peak2_params +peak3_params))


# In[12]:


graph=peak2+peak3
graph_params = peak2_params +peak3_params


# In[13]:


# 2peak 600개, 3peak 600개 만들엇어
# zip
unshuffle = []
for i in zip(graph,graph_params):
    unshuffle.append(i)

random.shuffle(unshuffle)


# In[14]:


shuffle_data = []
shuffle_labels = []

for i in range(data_number*2):
    shuffle_data.append(unshuffle[i][0])
    shuffle_labels.append(unshuffle[i][1]) 


# In[15]:


#ex
shuffle_labels[5]


# In[16]:


shuffle_labels[599][-1]


# In[17]:


peak_number = []
for i in range(data_number*2):
    peak_number.append(shuffle_labels[i][-1])
    # shuffle_labels[i].pop()


# In[18]:


#============================================
            # 조심!! 계쏙 누르면 다 사라짐
#============================================

for i in range(data_number*2):
    shuffle_labels[i].pop()


# In[19]:


shuffle_labels[1]


# In[20]:


center = []
width = []
amp = []
for i in range(data_number*2):
    center.append(np.array((shuffle_labels[i][::3])))
    width.append(np.array(shuffle_labels[i][1::3]))
    amp.append(np.array(shuffle_labels[i][2::3]))
# 굳이 안되면 for 3번 쓰면돼
#일단 정렬 안해보고 해보자


# In[ ]:


import numpy as np

e = []
for i in zip(center,width,amp):
    e.append(i)

g= []
for i in range(len(e)):
    for j in zip(e[i][0],e[i][1],e[i][2]):
        g.append(j)

center = np.zeros((len(e),3))
width = np.zeros((len(e),3))
amp = np.zeros((len(e),3))

total = []
for i in range(len(e)):
    print(sorted(g[3*i:3*i+3]))
    center[i][0] = sorted(g[3*i:3*i+3])[0][0]
    center[i][1] = sorted(g[3*i:3*i+3])[1][0]
    center[i][2] = sorted(g[3*i:3*i+3])[2][0]

    width[i][0] = sorted(g[3*i:3*i+3])[0][1]
    width[i][1] = sorted(g[3*i:3*i+3])[1][1]
    width[i][2] = sorted(g[3*i:3*i+3])[2][1]

    amp[i][0] = sorted(g[3*i:3*i+3])[0][2]
    amp[i][1] = sorted(g[3*i:3*i+3])[1][2]
    amp[i][2] = sorted(g[3*i:3*i+3])[2][2]

print(center)
print(width)
print(amp)


# In[ ]:


np.array(center).shape


# In[ ]:


# 잘 정리됐는지 확인
graph = np.array(shuffle_data)
center = np.array(center)
width = np.array(width)
amp = np.array(amp)
peak_number = np.array(peak_number)
print(center[0:10])
print(width[0:10])
print(amp[0:10])
print(peak_number[0:10])
print(center.shape)
print(width.shape)
print(amp.shape)
print(peak_number.shape)
print(graph.shape)
print(graph)


# In[ ]:


center[1][1],width[i][1],amp[1][1]


# In[ ]:


for i in range(20,40):
    plt.figure(figsize = (8,4))
    plt.plot(x,graph[i],color = 'black')
    plt.plot(x,y(center[i][0],width[i][0],amp[i][0],x), color = 'blue')
    plt.plot(x,y(center[i][1],width[i][1],amp[i][1],x), color= 'green')
    plt.plot(x,y(center[i][2],width[i][2],amp[i][2],x), color='red')


# In[ ]:


graph = np.array(shuffle_data)
center = np.array(center)
width = np.array(width)
amp = np.array(amp)
peak_number = np.array(peak_number)


# In[ ]:


print(graph.shape)
print(center.shape)
print(width.shape)
print(amp.shape)
print(peak_number.shape)


# In[ ]:


import pandas as pd
df_graph = pd.DataFrame(graph)
df_center = pd.DataFrame(center)
df_width = pd.DataFrame(width)
df_amp = pd.DataFrame(amp)
df_peak_number = pd.DataFrame(peak_number)

df_graph.head()
df_center.head()
df_width.head()
df_amp.head()
df_peak_number.head()


# In[ ]:


df_graph.to_csv('graph_1000.csv')


# In[ ]:


df_center.to_csv('center_1000.csv')
df_peak_number.to_csv('peak_number_2000.csv')


# In[ ]:


df_width.to_csv('width_1000.csv')
df_amp.to_csv('amp_2000.csv')


# In[ ]:


get_ipython().run_line_magic('ls', '')


# In[ ]:


# from google.colab import drive
# drive.mount("/content/drive/")


# In[ ]:


# import csv

# df = pd.DataFrame(graph)
# df.to_csv('C:\Users\bad_w\Desktop\메모장',encoding='utf-8')

