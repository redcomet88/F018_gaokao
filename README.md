# F018 Vue+Flask构建的高考预测可视化系统|带预测算法机器学习

完整项目收费，可联系微信: mmdsj186011 注明从git来的，谢谢！
也可以关注我的B站： 麦麦大数据 https://space.bilibili.com/1583208775

关注B站，有好处！

## 1 系统简介
系统简介：本系统是一个基于Vue+Flask构建的高考预测可视化系统，其核心功能围绕高考数据的展示、预测和用户管理展开。主要包括：首页，用于展示系统概览和轮播图；数据卡片，提供高考数据的概览，并支持查看位置（例如通过百度地图）及用户点赞“喜欢”的功能；可视化模块，通过丰富的图表展示专业的排名信息和关注度热力图，为用户提供直观的数据分析；分数线预测模块，利用机器学习模型进行智能预测，为考生提供参考；以及用户管理模块，包含登录与注册功能，和个人设置（允许用户修改个人信息、头像及密码），确保系统的安全性和个性化体验。
## 2 功能设计

该系统采用典型的B/S（浏览器/服务器）架构模式。用户通过浏览器访问Vue前端界面，该前端由HTML、CSS、JavaScript以及Vue.js生态系统中的Vuex（用于状态管理）、Vue Router（用于路由导航）和Echarts（用于数据可视化）等组件构建。前端通过API请求与Flask后端进行数据交互，Flask后端则负责业务逻辑处理，并利用SQLAlchemy（或类似ORM工具）与MySQL数据库进行持久化数据存储。此外，系统还包含一个独立的爬虫模块，负责从外部来源抓取数据并将其导入MySQL数据库，为整个系统提供数据支撑。
### 2.1系统架构图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/77fb0020d9b84cd9bbfcae47b2d5413e.png)
### 2.2 功能模块图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5a02ab10d10149dcaf3754ce5b8c242d.png)
## 3 功能展示
### 3.1 登录 & 注册
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/15c256b5ed594523bcc9dc40c1ba2b6d.png)
### 3.2 主页
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/322fcb5307a343aea15a3cf31f8763d1.png)
### 3.3 高校位置
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cc81550540974af99c8508fc727fc959.png)
### 3.4 可视化
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d50c68ad434e4dfc8b73879b345bba62.png)
### 3.5 高考关注度
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71c52a2e5f5c4257bcb73911a0c83079.png)
### 3.6 国家线可视化
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8a923dd023e044b192f99134f836819f.png)
### 3.7 分数线预测
基于机器学习SVM的分数线预测
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fc760552ac9a4a6088432f9f7b78822b.png)
### 3.8爬虫文件
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/95287522837643ed88953e8dcd0b6182.png)
## 4程序代码
### 4.1 代码说明
代码介绍：该代码利用支持向量回归（SVR）模型预测高考分数线。其核心思想是找到一个最优超平面，使得所有训练样本点到该超平面的距离（在一定误差容忍范围内）最小化。我们模拟了高考年份、卷子难度指数和报名人数增长率作为特征，文科或理科分数线作为目标变量。
主要实现步骤包括：数据准备，构建包含特征和目标的数据集。数据标准化，使用StandardScaler对输入特征进行缩放，因为SVR对特征尺度敏感。模型构建与优化，使用GridSearchCV进行网格搜索，通过交叉验证寻找SVR的最佳超参数（如核函数kernel、惩罚系数C、核函数参数gamma、容错率epsilon），以提高模型泛化能力。模型评估，通过均方误差（MSE）和R2分数评估模型在训练集上的拟合效果。最后，使用训练好的最佳模型对假设的未来年份（如2023年）的特征进行预测，并可视化预测结果，直观展示历史趋势与未来预测。
### 4.2 流程图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/32e6f29e8b68430e9da3611c03f7991f.png)

### 4.3 代码实例
```python
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据准备
# 假设我们有以下历史数据：年份、全国卷难度指数（模拟）、报名人数增长率（模拟）、当年文科分数线、当年理科分数线
# 实际应用中，这里的数据会更复杂，包含更多维度，如当年经济形势、教育政策等
data = {
    '年份': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    '卷子难度指数': [0.7, 0.6, 0.8, 0.75, 0.65, 0.9, 0.85, 0.78, 0.82, 0.92, 0.88, 0.75, 0.80], # 模拟数据, 0-1之间，越大越难
    '报名人数增长率': [0.01, 0.015, 0.008, 0.012, 0.009, 0.018, 0.014, 0.011, 0.016, 0.02, 0.017, 0.01, 0.013], # 模拟数据
    # 模拟分数线，通常在480-550之间
    '文科分数线': [520, 535, 505, 518, 530, 495, 508, 515, 510, 490, 498, 512, 505],
    '理科分数线': [545, 550, 520, 538, 542, 510, 528, 535, 530, 505, 515, 530, 525]
}
df = pd.DataFrame(data)

# 假设我们要预测2023年的分数线，所以将2022年及以前的数据作为训练集
# 预测目标：文科分数线 / 理科分数线
target_column = '文科分数线' # 可以切换为 '理科分数线' 来预测不同的类别

X = df[['卷子难度指数', '报名人数增长率']]
y = df[target_column]

# 分割训练集和测试集 (这里我们实际上用所有历史数据训练，然后预测未来一年)
# 在实际应用中，如果数据量大，通常会划分训练集和验证集
# 这里我们更侧重于演示模型构建和预测
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = X, y # 使用所有历史数据进行训练

# 2. 数据标准化
# 对特征进行标准化，SVM对特征的尺度很敏感
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test) # 如果有测试集，也要标准化

print("标准化后的训练数据前5行:")
print(X_train_scaled[:5])

# 3. 构建SVM回归模型 (SVR)
# 定义参数网格，用于网格搜索优化超参数
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.5, 1] # SVR特有的参数，ε-不敏感带，允许在ε范围内不惩罚误差
}

# 使用GridSearchCV进行网格搜索，寻找最佳超参数
# cv=5 表示5折交叉验证
svr = svm.SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# 打印最佳参数和最佳分数
print("\n最佳SVM回归模型参数:", grid_search.best_params_)
print("最佳交叉验证负均方误差:", grid_search.best_score_)

# 获取最佳模型
best_svr = grid_search.best_estimator_

# 4. 模型评估 (此处无独立测试集，仅展示拟合效果)
# y_pred_test = best_svr.predict(X_test_scaled)
# print(f"\n模型在测试集上的均方误差 (MSE): {mean_squared_error(y_test, y_pred_test):.2f}")
# print(f"模型在测试集上的R2分数: {r2_score(y_test, y_pred_test):.2f}")

# 可以将训练集上的预测结果与真实值进行比较
y_train_pred = best_svr.predict(X_train_scaled)
print(f"\n模型在训练集上的均方误差 (MSE): {mean_squared_error(y_train, y_train_pred):.2f}")
print(f"模型在训练集上的R2分数: {r2_score(y_train, y_train_pred):.2f}")

# 5. 预测未来高考分数线
# 假设我们对2023年的卷子难度指数和报名人数增长率进行了预测
# 注意：这些预测值也需要根据实际情况获得，是模型输入的关键
future_data = pd.DataFrame({
    '卷子难度指数': [0.73],  # 假设2023年卷子难度预测值
    '报名人数增长率': [0.015] # 假设2023年报名人数增长率预测值
})

# 对未来数据进行与训练集相同的标准化处理
future_data_scaled = scaler.transform(future_data)

# 进行预测
predicted_score = best_svr.predict(future_data_scaled)[0]

print(f"\n预测 2023 年 {target_column}: {predicted_score:.2f}")

# 6. 可视化 (用于展示拟合效果)
plt.figure(figsize=(10, 6))
plt.scatter(df['年份'], y, color='blue', label='实际分数线')
plt.plot(df['年份'], y_train_pred, color='red', linestyle='--', label='模型拟合分数线')
# 为了在图上显示预测点，可以在年份x轴上添加2023年
# 注意：y_train_pred 中不包含2023年的预测值，需要单独加入
years_with_prediction = np.append(df['年份'].values, 2023)
predicted_scores_full = np.append(y_train_pred, predicted_score)

plt.scatter([2023], [predicted_score], color='green', marker='X', s=100, label=f'2023年预测 ({predicted_score:.2f})')
plt.title(f'{target_column} 历史趋势与预测')
plt.xlabel('年份')
plt.ylabel('分数线')
plt.xticks(years_with_prediction) # 显示所有年份及预测年份
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

```
