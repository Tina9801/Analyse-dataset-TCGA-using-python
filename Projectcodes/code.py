import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift
from sklearn.manifold import TSNE
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from itertools import cycle
"""This is Computer Programming Course Report Code by Zeng Xiaoting, 1st July, 2018"""

"""---------1.导入及整理数据，生成数组------"""
#1-1 导入数据
odata = pd.read_excel(r'C:\Users\tsengxt\Desktop\DOING\python_class\final\TCGA2.xlsx',
                     index_col = u'mirna_id')
data = odata[odata.sum(axis = 1)!=0]
'''

"""---------2.对数据进行初步的统计分析-----------
  主要通过数据的均值和最大值查看了数据的分布情况，然后通过画箱线图对较为活跃的
  基因数据进行分析，最后通过画散布图观察它们的关联情况
"""


#2-1 查看数据分布情况

"""a. 查看均值分布情况"""
order = np.arange(1,1467)
meanVal2 = data.mean(axis = 1)

plt.figure(1)
plt.plot(order,meanVal2)
plt.title('Means of genes')
plt.savefig('means')


"""b. 查看最大值分布情况"""
maxVal = data.max(axis = 1)

plt.figure(2)
plt.plot(order,maxVal)
plt.title('Max activeness value of genes')
plt.savefig('max_genes_plot')

"""对比后发现最大值的情况与均值分布情况相似"""

"""查找出活跃值较为突出的那几个基因，储存在特定数组中"""

slt = data[data.mean(axis = 1)>10000]


"""c. 观察数据分布：箱线图"""
plt.figure(3)
plt.boxplot(slt,0,'')
plt.title('Boxplot of the most active genes')
plt.savefig('boxplot_most_active')

""" 去除极端基因后的箱线图"""
slt2 = slt.drop('hsa-mir-143')
plt.figure(4)
plt.boxplot(slt2,0,'')
plt.title('Boxplot of the most active genes(except"hsa-mir-143")')
plt.savefig('boxplot_most_active_exception')

#2-2 探索性分析：散布图及折线图

""" a. 基因之间的关联：散布图"""
test = slt.T.iloc[ :, :5]    #图像内存不足，因此仅选取了5个基因
g = sns.pairplot(test)
plt.show()
g.savefig("scatter_plots_active_genes.png")


"""------------3.聚类分析及关联分析-----------"""


#3-1 数据分类：利用均值漂移法

data_tsne = TSNE(learning_rate=100).fit_transform(data)
clf = MeanShift()
predicted = clf.fit_predict(data)
labels = clf.labels_
cluster_centers = clf.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)
print(cluster_centers)

"""绘制聚类图"""
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(data_tsne[my_members, 0], data_tsne[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.savefig(mean_shift)


#3-2 查找活跃度有相同变化趋势的基因组：利用余弦相似度
def consimi(a,b):    
    num = np.dot(a,b.T)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float('%.5f' % (num / denom))
    return cos

simi_mtx = []      
for i in range(0,1466):
    vectorA = data.iloc[i]
    for p in range(0,1466):
        vectorB = data.iloc[p]
        simi = consimi(vectorA, vectorB)
        simi_mtx.append(simi)
        p += 1
    i += 1
    

simi_ary = np.array(simi_mtx).reshape(1466,1466)
location = np.where(simi_ary>=0.85)

np.savetxt("simiarray.txt",simi_ary)
np.savetxt("location.txt",location)


#3-3 根据3-2结果对所得基因组进行关联分析：简单线性回归
'''
reg = linear_model.LinearRegression()
'''
"""对于第一个基因有95个相关基因，在此仅选取前15个恰巧相连的相关基因进行分析"""
gene1 = np.array(data.iloc[0]).reshape(-1,1)
gene_1_15 = np.array(data.iloc[1:16]).T
reg.fit(gene_1_15,gene1)
gene_1_predict = reg.predict(gene_1_15)

"""coefficients"""
coef = reg.coef_
print('Coefficients:\n', coef )

"""mean square error"""
MSE = mean_squared_error(gene1,gene_1_predict)
print('Mean squared error:%.2f ' % MSE)

"""r-squared"""
rsq = r2_score(gene1,gene_1_predict)
print('R-squared : %.5f'% rsq)

"""plot outputs"""
fig,ax = plt.subplots()
ax.scatter(gene1,gene_1_predict)
ax.plot([gene1.min(),gene1.max()],[gene1.min(),gene1.max()],'k--',lw = 3)

plt.text(200000, 50000, r'R-squared : %.5f'% rsq, fontsize=12)
plt.title('Multiple linear regression result: gene "hsa-let-7a-1"')
plt.savefig('gene1_regression')
'''

"""对于第17个基因有2个相关基因：第18、1392个"""
gene17 = np.array(data.iloc[16]).reshape(-1,1)
gene18 = np.array(data.iloc[17])
gene1393 = np.array(data.iloc[1392])

gene17_related = np.concatenate([gene18,gene1393]).reshape(2,255).T
reg.fit(gene17_related,gene17)
gene17_predict = reg.predict(gene17_related)

"""coefficients"""
coef = reg.coef_
print('Coefficients:\n', coef )

"""mean square error"""
MSE = mean_squared_error(gene17,gene17_predict)
print('Mean squared error:%.2f ' % MSE)

"""r-squared"""
rsq = r2_score(gene17,gene17_predict)
print('R-squared : %.5f'% rsq)
'''
"""plot outputs"""
fig,ax = plt.subplots()
ax.scatter(gene17,gene17_predict)
ax.plot([gene17.min(),gene17.max()],[gene17.min(),gene17.max()],'k--',lw = 3)

plt.text(30, 10, r'R-squared : %.5f'% rsq, fontsize=12)
plt.title('Multiple linear regression result: gene "hsa-mir-105-1"')
plt.savefig('gene17_regression')

'''



















