# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:25:57 2025

@author: Administrator
"""

import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import shap
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors
from scipy.stats import gaussian_kde
from matplotlib import ticker
import pylab
from scipy import stats
from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
plt.style.use('ggplot')
#plt.style.use('default')
#处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#坐标轴负号的处理
plt.rcParams['axes.unicode_minus']=False
shap.initjs()

file_path = r'E:\study\paper\papre\code\coded\satlinity\weight_analysis//'

file_list = glob(file_path + 'AIEM*csv')
all_data = pd.DataFrame()
#将数据合并
for each_csv in file_list:
    name = os.path.basename(each_csv)
    print(name)
    data = pd.read_csv(each_csv)
    #all_data.append(data)
    all_data = pd.concat([all_data, data], ignore_index=True)
#取数据
all_data = all_data.dropna()
all_data = all_data[all_data['f']==5]
all_data = all_data[all_data['T']==20]
all_data = all_data[all_data['angle']==60]
all_data = all_data.sample(frac=0.2)
X = all_data[['Mv', 's', 'c', 'T', 'f', 'Ssw','rms', 'cl']]
y_hh = all_data['HH1']
y_vv = all_data['VV1']
# 选数据
X = X[y_hh < 0]
y_hh = y_hh[y_hh < 0]
#作图，展示数据的值域分布
#plt.hist(all_data['Mv'], bins = 20,color = 'steelblue',edgecolor = 'black')
#plt.hist(all_data['s'], bins = 20,color = 'steelblue',edgecolor = 'black')
#plt.hist(all_data['c'], bins = 20,color = 'steelblue',edgecolor = 'black')
#plt.hist(all_data['T'], bins = 20,color = 'steelblue',edgecolor = 'black')
#plt.hist(all_data['f'], bins = 20,color = 'steelblue',edgecolor = 'black')
#plt.hist(all_data['Ssw'], bins = 20,color = 'steelblue',edgecolor = 'black')
#plt.hist(all_data['rms'], bins = 20,color = 'steelblue',edgecolor = 'black')
#plt.hist(all_data['rms'], bins = 20,color = 'steelblue',edgecolor = 'black')
#np.random()
#设置随机数，随机选取数据作为训练数据
###############################################################################################
# RF
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
font_latex1 = {'family':'Times New Roman', 'size':16, 'color':'k'}
# font_latex2 = {'family':'SimHe', 'size':16, 'color':'k'}
font_latex2 ={'family': 'Times New Roman','color':'black','weight':'normal','size': 16,}
sample_num1 = []
R2_num = []
RMSE_num = []
for i in range(10,1000):
    sample_num = i
    dataset_len = len(y_hh)
    Sample_location = np.random.random(size = sample_num)
    Sample_location_adjust = Sample_location * dataset_len
    Sample_location_adjust = Sample_location_adjust.astype(int)
    X_train = X.iloc[Sample_location_adjust]
    y_train = y_hh.iloc[Sample_location_adjust]
    #Random Forest Model
    model_RF = RandomForestRegressor()
    model_RF.fit(X_train, y_train)
    testsample_num = 1000
    Sample_location = np.random.random(size = testsample_num)
    Sample_location_adjust = Sample_location * dataset_len
    Sample_location_adjust = Sample_location_adjust.astype(int)
    X_test = X.iloc[Sample_location_adjust]
    y_test = y_hh.iloc[Sample_location_adjust]
    y_predict = model_RF.predict(X_test)
    #
    #画图参数统计分布和模型验证精度之间的关系
    #
    # 计算相关系数和 R^2
    print('Correlation coefficients：')
    print(np.corrcoef(y_test, y_predict))
    correlation = np.corrcoef(y_test, y_predict)[0, 1]
    R_square = correlation ** 2
    print("R^2：", R_square)
    R2_num.append(R_square)
    RMSE_num.append(mean_squared_error(y_test, y_predict, squared=False))
    sample_num1.append(i)
x = np.array(sample_num1)
y = np.array(R2_num)
my_cmap = "Spectral_r"
# 绘制密度散点图
# 核密度估计
x_and_y = np.vstack([x, y])
# kde = gaussian_kde(x_and_y)
# z = kde(x_and_y)
z = stats.gaussian_kde(x_and_y)(x_and_y)
idx = z.argsort()
xx, y, z = x[idx], y[idx], z[idx]
plt.scatter(xx, y, c=z, marker = 'o',s = 15,cmap=my_cmap)
plt.colorbar(label='Density')
z1 = np.polyfit(x, y, 3) 
p1 = np.poly1d(z1)
print(p1)
y_pred = p1(x) 
# plt.plot(sample_num1, R2_num, 'o',color = 'b', label='Predicted')
plot2 = pylab.plot(x, y_pred, '#800080',linewidth = 3.0, label='fit values')
# 设置坐标轴刻度
plt.tick_params(axis='x', direction='out', labelsize=13, length=4.6, width=1.15)
plt.tick_params(axis='y', direction='out', labelsize=13, length=4.6, width=1.15)
# 设置 X 轴和 Y 轴的标签、字体、刻度和刻度标签在内的坐标轴边界框中的间距
plt.xlabel("Train Sample Number", fontdict=font_latex2, labelpad=8)
plt.ylabel('$ {R^2} $', fontdict=font_latex2, labelpad=8)
output_path = r'E:\study\paper\machine learning and quantitative remote sensing\result\RF\\'
output_name = output_path[:-1] +'RF_curve.jpg'
#plt.savefig(output_name,dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.savefig(output_name,dpi=300, bbox_inches='tight')