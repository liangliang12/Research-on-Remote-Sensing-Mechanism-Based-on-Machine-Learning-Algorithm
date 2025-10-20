# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 21:49:37 2025

@author: Liang
"""

# model performance and sample distribution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from glob import glob
import os
from sklearn.model_selection import GridSearchCV 
import scipy.stats as st
import seaborn as sns

# 示例数据（替换为你的数据）
# 读取模拟数据
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
all_data = all_data.sample(frac=0.2)
all_data = all_data[all_data['f']==5]
all_data = all_data[all_data['T']==20]
X = all_data[['Mv', 's', 'c', 'T', 'f', 'Ssw','rms', 'cl','angle']]
y_hh = all_data['HH1']
y_vv = all_data['VV1']
# 选数据
X = X[y_vv < 0]
y_hh = y_vv[y_vv < 0]
# y_vv = all_data['VV1']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_hh, test_size=0.2, random_state=42)

font_latex1 = {'family':'Times New Roman', 'size':16, 'color':'k'}
# font_latex2 = {'family':'SimHe', 'size':16, 'color':'k'}
font_latex2 ={'family': 'Times New Roman','color':'black','weight':'normal','size': 16,}
sample_num1 = []
RFR2_num = []
RFRMSE_num = []
Mean_mv = []
std_mv = []
low_CI_bound_mv = []; high_CI_bound_mv = []
Mean_s = []
std_s = []
low_CI_bound_s = []; high_CI_bound_s = []
Mean_c = []
std_c = []
low_CI_bound_c = []; high_CI_bound_c = []
Mean_T = []
std_T = []
low_CI_bound_T = []; high_CI_bound_T = []
Mean_f = []
std_f = []
low_CI_bound_f = []; high_CI_bound_f = []
Mean_ssw = []
std_ssw = []
low_CI_bound_ssw = []; high_CI_bound_ssw = []
Mean_rms = []
std_rms = []
low_CI_bound_rms = []; high_CI_bound_rms = []
Mean_cl = []
std_cl = []
low_CI_bound_cl = []; high_CI_bound_cl = []
Mean_angle = []
std_angle = []
low_CI_bound_angle = []; high_CI_bound_angle = []
for i in range(10,1000):
    sample_num = 1000#把样本设置成固定的，看样本分布和精度之间的关系
    dataset_len = len(y_hh)
    Sample_location = np.random.random(size = sample_num)
    Sample_location_adjust = Sample_location * dataset_len
    Sample_location_adjust = Sample_location_adjust.astype(int)
    X_train = X.iloc[Sample_location_adjust]
    y_train = y_hh.iloc[Sample_location_adjust]
    #Random Forest Model
    model_RF = RandomForestRegressor()
    model_RF.fit(X_train, y_train)
    testsample_num = 200
    Sample_location = np.random.random(size = testsample_num)
    Sample_location_adjust = Sample_location * dataset_len
    Sample_location_adjust = Sample_location_adjust.astype(int)
    X_test = X.iloc[Sample_location_adjust]
    y_test = y_hh.iloc[Sample_location_adjust]
    y_predict = model_RF.predict(X_test)
    # 计算相关系数和 R^2
    # print('Correlation coefficients：')
    # print(np.corrcoef(y_test, y_predict))
    correlation = np.corrcoef(y_test, y_predict)[0, 1]
    R_square = correlation ** 2
    # print("R^2：", R_square)
    RMSE = mean_squared_error(y_test, y_predict, squared=False)
    RFR2_num.append(R_square)
    RFRMSE_num.append(RMSE)
    Mean_mv.append(np.mean(X_train['Mv']))
    std_mv.append(np.var(X_train['Mv']))
    low_CI_bound, high_CI_bound = st.t.interval(0.95, sample_num - 1,loc=np.mean(X_train['Mv'], 0),scale=st.sem(X_train['Mv']))
    low_CI_bound_mv.append(low_CI_bound); high_CI_bound_mv.append(high_CI_bound)
    Mean_s.append(np.mean(X_train['s']))
    std_s.append(np.var(X_train['s']))
    low_CI_bound, high_CI_bound = st.t.interval(0.95, sample_num - 1,loc=np.mean(X_train['s'], 0),scale=st.sem(X_train['s']))
    low_CI_bound_s.append(low_CI_bound); high_CI_bound_s.append(high_CI_bound)
    Mean_c.append(np.mean(X_train['c']))
    std_c.append(np.var(X_train['c']))
    low_CI_bound, high_CI_bound = st.t.interval(0.95, sample_num - 1,loc=np.mean(X_train['c'], 0),scale=st.sem(X_train['c']))
    low_CI_bound_c.append(low_CI_bound); high_CI_bound_c.append(high_CI_bound)
    Mean_T.append(np.mean(X_train['T']))
    std_T.append(np.var(X_train['T']))
    low_CI_bound, high_CI_bound = st.t.interval(0.95, sample_num - 1,loc=np.mean(X_train['T'], 0),scale=st.sem(X_train['T']))
    low_CI_bound_T.append(low_CI_bound); high_CI_bound_T.append(high_CI_bound)
    Mean_f.append(np.mean(X_train['f']))
    std_f.append(np.var(X_train['f']))
    low_CI_bound, high_CI_bound = st.t.interval(0.95, sample_num - 1,loc=np.mean(X_train['f'], 0),scale=st.sem(X_train['f']))
    low_CI_bound_f.append(low_CI_bound); high_CI_bound_f.append(high_CI_bound)
    Mean_ssw.append(np.mean(X_train['Ssw']))
    std_ssw.append(np.var(X_train['Ssw']))
    low_CI_bound, high_CI_bound = st.t.interval(0.95, sample_num - 1,loc=np.mean(X_train['Ssw'], 0),scale=st.sem(X_train['Ssw']))
    low_CI_bound_ssw.append(low_CI_bound); high_CI_bound_ssw.append(high_CI_bound)
    Mean_rms.append(np.mean(X_train['rms']))
    std_rms.append(np.var(X_train['rms']))
    low_CI_bound, high_CI_bound = st.t.interval(0.95, sample_num - 1,loc=np.mean(X_train['rms'], 0),scale=st.sem(X_train['rms']))
    low_CI_bound_rms.append(low_CI_bound); high_CI_bound_rms.append(high_CI_bound)
    Mean_cl.append(np.mean(X_train['cl']))
    std_cl.append(np.var(X_train['cl']))
    low_CI_bound, high_CI_bound = st.t.interval(0.95, sample_num - 1,loc=np.mean(X_train['cl'], 0),scale=st.sem(X_train['cl']))
    low_CI_bound_cl.append(low_CI_bound); high_CI_bound_cl.append(high_CI_bound)
    Mean_angle.append(np.mean(X_train['angle']))
    std_angle.append(np.var(X_train['angle']))
    low_CI_bound, high_CI_bound = st.t.interval(0.95, sample_num - 1,loc=np.mean(X_train['angle'], 0),scale=st.sem(X_train['angle']))
    low_CI_bound_angle.append(low_CI_bound); high_CI_bound_angle.append(high_CI_bound)
    sample_num1.append(i)
############################################################################################################################################
#置信区间
# plt.fill_between(sample_num1, low_CI_bound, high_CI_bound, alpha=0.5,label='confidence interval')    
# 设置画布大小
plt.figure(figsize=(30, 14), dpi=300)
# fig1, axes1 = plt.subplots(3, 3, figsize=(14, 4))


# fig, ax1 = 
plt.subplot(3, 3, 1)
plt.plot(sample_num1,RFR2_num)
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
plt.plot(sample_num1,RFRMSE_num)
# ax2.
plt.subplot(3, 3, 2)
plt.plot(sample_num1,Mean_mv)
plt.fill_between(np.array(sample_num1), np.array(low_CI_bound_mv), np.array(high_CI_bound_mv), alpha=0.5,label='confidence interval')
plt.subplot(3, 3, 3)
plt.plot(sample_num1,Mean_s)
plt.fill_between(np.array(sample_num1), np.array(low_CI_bound_s), np.array(high_CI_bound_s), alpha=0.5,label='confidence interval')
plt.subplot(3, 3, 4)
plt.plot(sample_num1,Mean_T)
plt.fill_between(np.array(sample_num1), np.array(low_CI_bound_T), np.array(high_CI_bound_T), alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_T)
plt.subplot(3, 3, 5)
plt.plot(sample_num1,Mean_f)
# plt.plot(sample_num1,std_f)
plt.fill_between(np.array(sample_num1), np.array(low_CI_bound_f), np.array(high_CI_bound_f), alpha=0.5,label='confidence interval')
plt.subplot(3, 3, 6)
plt.plot(sample_num1,Mean_ssw)
plt.fill_between(np.array(sample_num1), np.array(low_CI_bound_ssw), np.array(high_CI_bound_ssw), alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_ssw)
plt.subplot(3, 3, 7)
plt.plot(sample_num1,Mean_rms)
plt.fill_between(np.array(sample_num1), np.array(low_CI_bound_rms), np.array(high_CI_bound_rms), alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_rms)
plt.subplot(3, 3, 8)
plt.plot(sample_num1,Mean_cl)
plt.fill_between(np.array(sample_num1), np.array(low_CI_bound_cl), np.array(high_CI_bound_cl), alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_cl)
plt.subplot(3, 3, 9)
plt.plot(sample_num1,Mean_angle)
plt.fill_between(np.array(sample_num1), np.array(low_CI_bound_angle), np.array(high_CI_bound_angle), alpha=0.5,label='confidence interval')
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\sample_distribution.jpg'
plt.savefig(output_file,dpi=300)
# plt.plot(sample_num1,std_angle)
#做成numpy数据
sample_num1 = np.array(sample_num1); RFR2_num = np.array(RFR2_num); RFRMSE_num = np.array(RFRMSE_num); Mean_mv = np.array(Mean_mv); std_mv = np.array(std_mv)
low_CI_bound_mv = np.array(low_CI_bound_mv); high_CI_bound_mv = np.array(high_CI_bound_mv)
Mean_s = np.array(Mean_s); std_s = np.array(std_s); low_CI_bound_s = np.array(low_CI_bound_s); high_CI_bound_s = np.array(high_CI_bound_s)
Mean_c = np.array(Mean_c); std_c = np.array(std_c); low_CI_bound_c = np.array(low_CI_bound_c); high_CI_bound_c = np.array(high_CI_bound_c)
Mean_T = np.array(Mean_T); std_T = np.array(std_T); low_CI_bound_T = np.array(low_CI_bound_T); high_CI_bound_T = np.array(high_CI_bound_T)
Mean_f = np.array(Mean_f); std_f = np.array(std_f); low_CI_bound_f = np.array(low_CI_bound_f); high_CI_bound_f = np.array(high_CI_bound_f)
Mean_ssw = np.array(Mean_ssw); std_ssw = np.array(std_ssw); low_CI_bound_ssw = np.array(low_CI_bound_ssw); high_CI_bound_ssw = np.array(high_CI_bound_ssw)
Mean_rms = np.array(Mean_rms); std_rms = np.array(std_rms); low_CI_bound_rms = np.array(low_CI_bound_rms); high_CI_bound_rms = np.array(high_CI_bound_rms)
Mean_cl = np.array(Mean_cl); std_cl = np.array(std_cl); low_CI_bound_cl = np.array(low_CI_bound_cl); high_CI_bound_cl = np.array(high_CI_bound_cl)
Mean_angle = np.array(Mean_angle); std_angle = np.array(std_angle); low_CI_bound_angle = np.array(low_CI_bound_angle); high_CI_bound_angle = np.array(high_CI_bound_angle)
############################################################################################################################################
#按照R2和RMSE排序
sort_data = {'sample_num1':sample_num1, 'RFR2_num':RFR2_num, 'RFRMSE_num':RFRMSE_num, 'Mean_mv':Mean_mv, 'std_mv':std_mv, 'low_CI_bound_mv':low_CI_bound_mv,
             'high_CI_bound_mv':high_CI_bound_mv, 'Mean_s':Mean_s, 'std_s':std_s, 'low_CI_bound_s':low_CI_bound_s, 'high_CI_bound_s':high_CI_bound_s, 'Mean_c':Mean_c,
             'std_c':std_c, 'low_CI_bound_c':low_CI_bound_c,'high_CI_bound_c':high_CI_bound_c, 'Mean_T':Mean_T, 'std_T':std_T, 'low_CI_bound_T':low_CI_bound_T,
             'high_CI_bound_T':high_CI_bound_T,'Mean_f':Mean_f,'std_f':std_f,'low_CI_bound_f':low_CI_bound_f,'high_CI_bound_f':high_CI_bound_f,
             'Mean_ssw':Mean_ssw,'std_ssw':std_ssw,'low_CI_bound_ssw':low_CI_bound_ssw,'high_CI_bound_ssw':high_CI_bound_ssw,'Mean_rms':Mean_rms,
             'std_rms':std_rms,'low_CI_bound_rms':low_CI_bound_rms,'high_CI_bound_rms':high_CI_bound_rms,'Mean_cl':Mean_cl,'std_cl':std_cl,
             'low_CI_bound_cl':low_CI_bound_cl,'high_CI_bound_cl':high_CI_bound_cl,'Mean_angle':Mean_angle,'std_angle':std_angle,'low_CI_bound_angle':low_CI_bound_angle,
             'high_CI_bound_angle':high_CI_bound_angle}
sort_data = pd.DataFrame(sort_data)
############################################################################################################################################
#按照R2升序的方式排序
sort_data = sort_data.sort_values(by='RFR2_num', ascending = True)
plt.figure(figsize=(30, 14), dpi=300)
# fig1, axes1 = plt.subplots(3, 3, figsize=(14, 4))
# fig, ax1 = 
plt.subplot(3, 3, 1)
plt.plot(sample_num1, sort_data['RFR2_num'])
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
plt.plot(sample_num1, sort_data['RFRMSE_num'])
plt.subplot(3, 3, 2)
plt.plot(sample_num1, sort_data['Mean_mv'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_mv'], sort_data['high_CI_bound_mv'], alpha=0.5,label='confidence interval')
plt.subplot(3, 3, 3)
plt.plot(sample_num1, sort_data['Mean_s'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_s'], sort_data['high_CI_bound_s'], alpha=0.5,label='confidence interval')
plt.subplot(3, 3, 4)
plt.plot(sample_num1, sort_data['Mean_T'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_T'], sort_data['high_CI_bound_T'], alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_T)
plt.subplot(3, 3, 5)
plt.plot(sample_num1, sort_data['Mean_f'])
# plt.plot(sample_num1,std_f)
plt.fill_between(sample_num1, sort_data['low_CI_bound_f'], sort_data['high_CI_bound_f'], alpha=0.5,label='confidence interval')
plt.subplot(3, 3, 6)
plt.plot(sample_num1, sort_data['Mean_ssw'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_ssw'], sort_data['high_CI_bound_ssw'], alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_ssw)
plt.subplot(3, 3, 7)
plt.plot(sample_num1,sort_data['Mean_rms'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_rms'], sort_data['high_CI_bound_rms'], alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_rms)
plt.subplot(3, 3, 8)
plt.plot(sample_num1, sort_data['Mean_cl'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_cl'], sort_data['high_CI_bound_cl'], alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_cl)
plt.subplot(3, 3, 9)
plt.plot(sample_num1, sort_data['Mean_angle'])
plt.fill_between(sample_num1, low_CI_bound_angle, high_CI_bound_angle, alpha=0.5,label='confidence interval')
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\sample_sortR2_distribution.jpg'
plt.savefig(output_file,dpi=300)
############################################################################################################################################
#按照RMSE升序的方式排序
#按照R2升序的方式排序
sort_data = sort_data.sort_values(by='RFRMSE_num', ascending = True)
plt.figure(figsize=(30, 14), dpi=300)
# fig1, axes1 = plt.subplots(3, 3, figsize=(14, 4))
# fig, ax1 = 
plt.subplot(3, 3, 1)
plt.plot(sample_num1, sort_data['RFR2_num'])
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
plt.plot(sample_num1, sort_data['RFRMSE_num'])
plt.subplot(3, 3, 2)
plt.plot(sample_num1, sort_data['Mean_mv'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_mv'], sort_data['high_CI_bound_mv'], alpha=0.5,label='confidence interval')
plt.subplot(3, 3, 3)
plt.plot(sample_num1, sort_data['Mean_s'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_s'], sort_data['high_CI_bound_s'], alpha=0.5,label='confidence interval')
plt.subplot(3, 3, 4)
plt.plot(sample_num1, sort_data['Mean_T'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_T'], sort_data['high_CI_bound_T'], alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_T)
plt.subplot(3, 3, 5)
plt.plot(sample_num1, sort_data['Mean_f'])
# plt.plot(sample_num1,std_f)
plt.fill_between(sample_num1, sort_data['low_CI_bound_f'], sort_data['high_CI_bound_f'], alpha=0.5,label='confidence interval')
plt.subplot(3, 3, 6)
plt.plot(sample_num1, sort_data['Mean_ssw'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_ssw'], sort_data['high_CI_bound_ssw'], alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_ssw)
plt.subplot(3, 3, 7)
plt.plot(sample_num1,sort_data['Mean_rms'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_rms'], sort_data['high_CI_bound_rms'], alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_rms)
plt.subplot(3, 3, 8)
plt.plot(sample_num1, sort_data['Mean_cl'])
plt.fill_between(sample_num1, sort_data['low_CI_bound_cl'], sort_data['high_CI_bound_cl'], alpha=0.5,label='confidence interval')
# plt.plot(sample_num1,std_cl)
plt.subplot(3, 3, 9)
plt.plot(sample_num1, sort_data['Mean_angle'])
plt.fill_between(sample_num1, low_CI_bound_angle, high_CI_bound_angle, alpha=0.5,label='confidence interval')
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\sample_sortRMSE_distribution.jpg'
plt.savefig(output_file,dpi=300)
# ############################################################################################################################################
# #均值为一个坐标，纵坐标为RMSE或者R2，误差棒为方差，或者分区统计
# #根据R2分等级
# #0.78-0.91
# #制作数据
# # r2_class = {'R2':[],'Mv':[],'s':[],'c':[],'T':[],'f':[],'Ssw':[],'rms':[],'cl':[],'angle':[],'RMSE':[],'R2':[]}
# plt.figure(figsize=(30, 14), dpi=300)
# r2_class = {'R2':[],'value':[],'label1':[]}
# min_r2 = round(np.min(RFR2_num), 2) - 0.01
# max_r2 = round(np.max(RFR2_num), 2) + 0.01
# r2_range = (max_r2 - min_r2)/5
# for each_i in range(5):
#     location = np.where(((RFR2_num < (min_r2+((each_i+1) * r2_range))) * (RFR2_num > (min_r2+(each_i * r2_range))))==True)
#     len_data = len(location[0])
#     label = str(round(min_r2+each_i * r2_range,2)) + '_' + str(round(min_r2+(each_i+1) * r2_range,2))
#     R2_R2 = RFR2_num[location]
#     # r2_class['R2'] = r2_class['R2'] + R2_R2
#     arr_label = [label for i in range(len_data)]
#     #r2_class['R2'] = r2_class['R2'] + arr_label
#     Mv_R2 = Mean_mv[location]
#     r2_class['value'] = r2_class['value'] + Mv_R2.tolist()
#     r2_class['label1'] = r2_class['label1'] + ['Mv' for i in range(len_data)]
#     r2_class['R2'] = r2_class['R2'] + arr_label
#     s_R2 = Mean_s[location]
#     r2_class['value'] = r2_class['value'] + s_R2.tolist()
#     r2_class['label1'] = r2_class['label1'] + ['s' for i in range(len_data)]
#     r2_class['R2'] = r2_class['R2'] + arr_label
#     c_R2 = Mean_c[location]
#     r2_class['value'] = r2_class['value'] + c_R2.tolist()
#     r2_class['label1'] = r2_class['label1'] + ['c' for i in range(len_data)]
#     r2_class['R2'] = r2_class['R2'] + arr_label
#     T_R2 = Mean_T[location]
#     r2_class['value'] = r2_class['value'] + T_R2.tolist()
#     r2_class['label1'] = r2_class['label1'] + ['T' for i in range(len_data)]
#     r2_class['R2'] = r2_class['R2'] + arr_label
#     f_R2 = Mean_f[location]
#     r2_class['value'] = r2_class['value'] + f_R2.tolist()
#     r2_class['label1'] = r2_class['label1'] + ['f' for i in range(len_data)]
#     r2_class['R2'] = r2_class['R2'] + arr_label
#     Ssw_R2 = Mean_ssw[location]
#     r2_class['value'] = r2_class['value'] + Ssw_R2.tolist()
#     r2_class['label1'] = r2_class['label1'] + ['Ssw' for i in range(len_data)]
#     r2_class['R2'] = r2_class['R2'] + arr_label
#     rms_R2 = Mean_rms[location]
#     r2_class['value'] = r2_class['value'] + rms_R2.tolist()
#     r2_class['label1'] = r2_class['label1'] + ['rms' for i in range(len_data)]
#     r2_class['R2'] = r2_class['R2'] + arr_label
#     cl_R2 = Mean_cl[location]
#     r2_class['value'] = r2_class['value'] + cl_R2.tolist()
#     r2_class['label1'] = r2_class['label1'] + ['cl' for i in range(len_data)]
#     r2_class['R2'] = r2_class['R2'] + arr_label
#     angle_R2  = Mean_angle[location]
#     r2_class['value'] = r2_class['value'] + angle_R2.tolist()
#     r2_class['label1'] = r2_class['label1'] + ['angle' for i in range(len_data)]
#     r2_class['R2'] = r2_class['R2'] + arr_label
# sns.boxplot(x='R2',y='value',hue='label1',data=r2_class)
# output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\sample_sortRMSE_distribution_class.jpg'
# plt.savefig(output_file,dpi=300)
r2_class = {'R2':[],'Mv':[],'s':[],'c':[],'T':[],'f':[],'Ssw':[],'rms':[],'cl':[],'angle':[],}
min_r2 = round(np.min(RFR2_num), 2) - 0.01
max_r2 = round(np.max(RFR2_num), 2) + 0.01
r2_range = (max_r2 - min_r2)/5
for each_i in range(5):
    location = np.where(((RFR2_num < (min_r2+((each_i+1) * r2_range))) * (RFR2_num > (min_r2+(each_i * r2_range))))==True)
    label = str(round(min_r2+each_i * r2_range,2)) + '-' + str(round(min_r2+(each_i+1) * r2_range,2))
    len_data = len(location[0])
    arr_label = [label for i in range(len_data)]
    #r2_class['R2'] = r2_class['R2'] + arr_label
    r2_class['R2'] = r2_class['R2'] + arr_label
    Mv_R2 = Mean_mv[location]
    r2_class['Mv'] = r2_class['Mv'] + Mv_R2.tolist()
    s_R2 = Mean_s[location]
    r2_class['s'] = r2_class['s'] + s_R2.tolist()
    c_R2 = Mean_c[location]
    r2_class['c'] = r2_class['c'] + c_R2.tolist()
    T_R2 = Mean_T[location]
    r2_class['T'] = r2_class['T'] + T_R2.tolist()
    f_R2 = Mean_f[location]
    r2_class['f'] = r2_class['f'] + f_R2.tolist()
    Ssw_R2 = Mean_ssw[location]
    r2_class['Ssw'] = r2_class['Ssw'] + Ssw_R2.tolist()
    rms_R2 = Mean_rms[location]
    r2_class['rms'] = r2_class['rms'] + rms_R2.tolist()
    cl_R2 = Mean_cl[location]
    r2_class['cl'] = r2_class['cl'] + cl_R2.tolist()
    angle_R2  = Mean_angle[location]
    r2_class['angle'] = r2_class['angle'] + angle_R2 .tolist()
    # RMSE_R2  = Mean_RMSE[location]
    # r2_class['RMSE'] = r2_class['RMSE'] + RMSE_R2 
    # r2_class['R2'] = r2_class['R2'] + RMSE_R2 
r2_class = pd.DataFrame(r2_class)
# color_palette=sns.color_palette("hls",8)
color_palette=sns.hls_palette(5,l=.5,s=.9)
# sns.palplot(color_palette) 
# sns.set_palette(sns.color_palette(colors))
# fig1, axes = plt.subplots(3, 3, figsize=(14, 20))
plt.figure(figsize=(7.5, 4), dpi=300)
f1 = sns.boxplot(x='R2',y='Mv',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
# f1 = sns.boxplot(x='R2',y='Mv',data=r2_class, width=0.5,boxprops=dict(facecolor="lightblue", edgecolor="darkblue", linewidth=2), palette=color_palette)
# sns.boxenplot(x='R2', y='Mv', data=r2_class,)
# 设置坐标轴标签及字体大小
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('Mv', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_Mean_Mv.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4))
sns.boxplot(x='R2',y='s',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  }, )
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('s', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_Mean_s.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='c',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('c', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_Mean_c.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='T',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('T', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_Mean_T.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='f',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('f', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_Mean_f.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='Ssw',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('Ssw', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_Mean_Ssw.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='rms',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('rms', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_Mean_rms.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='cl',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('cl', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_Mean_cl.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='angle',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('angle', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_Mean_angle.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')
# ############################################################################################################################################
#方差
r2_class = {'R2':[],'Mv':[],'s':[],'c':[],'T':[],'f':[],'Ssw':[],'rms':[],'cl':[],'angle':[],}
min_r2 = round(np.min(RFR2_num), 2) - 0.01
max_r2 = round(np.max(RFR2_num), 2) + 0.01
r2_range = (max_r2 - min_r2)/5
for each_i in range(5):
    location = np.where(((RFR2_num < (min_r2+((each_i+1) * r2_range))) * (RFR2_num > (min_r2+(each_i * r2_range))))==True)
    label = str(round(min_r2+each_i * r2_range,2)) + '-' + str(round(min_r2+(each_i+1) * r2_range,2))
    len_data = len(location[0])
    arr_label = [label for i in range(len_data)]
    #r2_class['R2'] = r2_class['R2'] + arr_label
    r2_class['R2'] = r2_class['R2'] + arr_label
    Mv_R2 = std_mv[location]
    r2_class['Mv'] = r2_class['Mv'] + Mv_R2.tolist()
    s_R2 = std_s[location]
    r2_class['s'] = r2_class['s'] + s_R2.tolist()
    c_R2 = std_c[location]
    r2_class['c'] = r2_class['c'] + c_R2.tolist()
    T_R2 = std_T[location]
    r2_class['T'] = r2_class['T'] + T_R2.tolist()
    f_R2 = std_f[location]
    r2_class['f'] = r2_class['f'] + f_R2.tolist()
    Ssw_R2 = std_ssw[location]
    r2_class['Ssw'] = r2_class['Ssw'] + Ssw_R2.tolist()
    rms_R2 = std_rms[location]
    r2_class['rms'] = r2_class['rms'] + rms_R2.tolist()
    cl_R2 = std_cl[location]
    r2_class['cl'] = r2_class['cl'] + cl_R2.tolist()
    angle_R2  = std_angle[location]
    r2_class['angle'] = r2_class['angle'] + angle_R2 .tolist()
    # RMSE_R2  = Mean_RMSE[location]
    # r2_class['RMSE'] = r2_class['RMSE'] + RMSE_R2 
    # r2_class['R2'] = r2_class['R2'] + RMSE_R2 
r2_class = pd.DataFrame(r2_class)
# color_palette=sns.color_palette("hls",8)
color_palette=sns.hls_palette(5,l=.5,s=.9)
# sns.palplot(color_palette) 
# sns.set_palette(sns.color_palette(colors))
# fig1, axes = plt.subplots(3, 3, figsize=(14, 20))
plt.figure(figsize=(7.5, 4), dpi=300)
f1 = sns.boxplot(x='R2',y='Mv',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
# f1 = sns.boxplot(x='R2',y='Mv',data=r2_class, width=0.5,boxprops=dict(facecolor="lightblue", edgecolor="darkblue", linewidth=2), palette=color_palette)
# sns.boxenplot(x='R2', y='Mv', data=r2_class,)
# 设置坐标轴标签及字体大小
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('Mv', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_std_Mv.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4))
sns.boxplot(x='R2',y='s',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  }, )
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('s', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_std_s.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='c',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('c', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_std_c.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='T',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('T', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_std_T.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='f',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('f', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_std_f.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='Ssw',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('Ssw', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_std_Ssw.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='rms',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('rms', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_std_rms.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='cl',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('cl', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_std_cl.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='angle',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('$ {R^2} $', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('angle', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\R2_std_angle.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')
# ############################################################################################################################################
#根据RMSE分级,统计平均值
r2_class = {'R2':[],'Mv':[],'s':[],'c':[],'T':[],'f':[],'Ssw':[],'rms':[],'cl':[],'angle':[],}
min_r2 = round(np.min(RFRMSE_num), 2) - 0.01
max_r2 = round(np.max(RFRMSE_num), 2) + 0.01
r2_range = (max_r2 - min_r2)/5
for each_i in range(5):
    location = np.where(((RFRMSE_num < (min_r2+((each_i+1) * r2_range))) * (RFRMSE_num > (min_r2+(each_i * r2_range))))==True)
    label = str(round(min_r2+each_i * r2_range,2)) + '-' + str(round(min_r2+(each_i+1) * r2_range,2))
    len_data = len(location[0])
    arr_label = [label for i in range(len_data)]
    #r2_class['R2'] = r2_class['R2'] + arr_label
    r2_class['R2'] = r2_class['R2'] + arr_label
    Mv_R2 = Mean_mv[location]
    r2_class['Mv'] = r2_class['Mv'] + Mv_R2.tolist()
    s_R2 = Mean_s[location]
    r2_class['s'] = r2_class['s'] + s_R2.tolist()
    c_R2 = Mean_c[location]
    r2_class['c'] = r2_class['c'] + c_R2.tolist()
    T_R2 = Mean_T[location]
    r2_class['T'] = r2_class['T'] + T_R2.tolist()
    f_R2 = Mean_f[location]
    r2_class['f'] = r2_class['f'] + f_R2.tolist()
    Ssw_R2 = Mean_ssw[location]
    r2_class['Ssw'] = r2_class['Ssw'] + Ssw_R2.tolist()
    rms_R2 = Mean_rms[location]
    r2_class['rms'] = r2_class['rms'] + rms_R2.tolist()
    cl_R2 = Mean_cl[location]
    r2_class['cl'] = r2_class['cl'] + cl_R2.tolist()
    angle_R2  = Mean_angle[location]
    r2_class['angle'] = r2_class['angle'] + angle_R2 .tolist()
    # RMSE_R2  = Mean_RMSE[location]
    # r2_class['RMSE'] = r2_class['RMSE'] + RMSE_R2 
    # r2_class['R2'] = r2_class['R2'] + RMSE_R2 
r2_class = pd.DataFrame(r2_class)
# color_palette=sns.color_palette("hls",8)
color_palette=sns.hls_palette(5,l=.5,s=.9)
# sns.palplot(color_palette) 
# sns.set_palette(sns.color_palette(colors))
# fig1, axes = plt.subplots(3, 3, figsize=(14, 20))
plt.figure(figsize=(7.5, 4), dpi=300)
f1 = sns.boxplot(x='R2',y='Mv',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
# f1 = sns.boxplot(x='R2',y='Mv',data=r2_class, width=0.5,boxprops=dict(facecolor="lightblue", edgecolor="darkblue", linewidth=2), palette=color_palette)
# sns.boxenplot(x='R2', y='Mv', data=r2_class,)
# 设置坐标轴标签及字体大小
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('Mv', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_Mean_Mv.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4))
sns.boxplot(x='R2',y='s',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  }, )
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('s', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_Mean_s.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='c',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('c', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_Mean_c.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='T',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('T', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_Mean_T.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='f',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('f', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_Mean_f.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='Ssw',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('Ssw', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_Mean_Ssw.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='rms',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('rms', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_Mean_rms.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='cl',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('cl', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_Mean_cl.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='angle',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('angle', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_Mean_angle.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')
# ############################################################################################################################################
#根据RMSE分级,统计方差
r2_class = {'R2':[],'Mv':[],'s':[],'c':[],'T':[],'f':[],'Ssw':[],'rms':[],'cl':[],'angle':[],}
min_r2 = round(np.min(RFR2_num), 2) - 0.01
max_r2 = round(np.max(RFR2_num), 2) + 0.01
r2_range = (max_r2 - min_r2)/5
for each_i in range(5):
    location = np.where(((RFR2_num < (min_r2+((each_i+1) * r2_range))) * (RFR2_num > (min_r2+(each_i * r2_range))))==True)
    label = str(round(min_r2+each_i * r2_range,2)) + '-' + str(round(min_r2+(each_i+1) * r2_range,2))
    len_data = len(location[0])
    arr_label = [label for i in range(len_data)]
    #r2_class['R2'] = r2_class['R2'] + arr_label
    r2_class['R2'] = r2_class['R2'] + arr_label
    Mv_R2 = std_mv[location]
    r2_class['Mv'] = r2_class['Mv'] + Mv_R2.tolist()
    s_R2 = std_s[location]
    r2_class['s'] = r2_class['s'] + s_R2.tolist()
    c_R2 = std_c[location]
    r2_class['c'] = r2_class['c'] + c_R2.tolist()
    T_R2 = std_T[location]
    r2_class['T'] = r2_class['T'] + T_R2.tolist()
    f_R2 = std_f[location]
    r2_class['f'] = r2_class['f'] + f_R2.tolist()
    Ssw_R2 = std_ssw[location]
    r2_class['Ssw'] = r2_class['Ssw'] + Ssw_R2.tolist()
    rms_R2 = std_rms[location]
    r2_class['rms'] = r2_class['rms'] + rms_R2.tolist()
    cl_R2 = std_cl[location]
    r2_class['cl'] = r2_class['cl'] + cl_R2.tolist()
    angle_R2  = std_angle[location]
    r2_class['angle'] = r2_class['angle'] + angle_R2 .tolist()
    # RMSE_R2  = Mean_RMSE[location]
    # r2_class['RMSE'] = r2_class['RMSE'] + RMSE_R2 
    # r2_class['R2'] = r2_class['R2'] + RMSE_R2 
r2_class = pd.DataFrame(r2_class)
# color_palette=sns.color_palette("hls",8)
color_palette=sns.hls_palette(5,l=.5,s=.9)
# sns.palplot(color_palette) 
# sns.set_palette(sns.color_palette(colors))
# fig1, axes = plt.subplots(3, 3, figsize=(14, 20))
plt.figure(figsize=(7.5, 4), dpi=300)
f1 = sns.boxplot(x='R2',y='Mv',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
# f1 = sns.boxplot(x='R2',y='Mv',data=r2_class, width=0.5,boxprops=dict(facecolor="lightblue", edgecolor="darkblue", linewidth=2), palette=color_palette)
# sns.boxenplot(x='R2', y='Mv', data=r2_class,)
# 设置坐标轴标签及字体大小
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('Mv', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_std_Mv.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4))
sns.boxplot(x='R2',y='s',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  }, )
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('s', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_std_s.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='c',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('c', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_std_c.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='T',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('T', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_std_T.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='f',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('f', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_std_f.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='Ssw',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('Ssw', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_std_Ssw.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='rms',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('rms', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_std_rms.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='cl',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('cl', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_std_cl.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')

plt.figure(figsize=(7.5, 4), dpi=300)
sns.boxplot(x='R2',y='angle',data=r2_class, width=0.6, linewidth=1,palette=color_palette, boxprops={"linewidth": 1.5, "edgecolor": "k"  }, medianprops={"linewidth": 2, "color": "b","linestyle": "--"  },)
plt.xlabel('RMSE', fontsize = 20, fontfamily = 'Times New Roman')
plt.ylabel('angle', fontsize = 20, fontfamily = 'Times New Roman')
# 设置刻度标签字体大小
plt.xticks(fontsize=16, fontfamily = 'Times New Roman')  # 横轴刻度标签
plt.yticks(fontsize=16, fontfamily = 'Times New Roman')  # 纵轴刻度标签
output_file = r'E:\study\paper\machine learning and quantitative remote sensing\result\parameters_distribution\RMSE_std_angle.jpg'
plt.savefig(output_file,dpi=300, bbox_inches = 'tight')
