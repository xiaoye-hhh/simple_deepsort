# 简单的追踪算法
看不懂deep_sort的骚操作操作,写一个简单的算法
- 取消了卡尔曼滤波, 简单的使用位置信息
- 使用resnet18来提取特征[在market1501上训练 短特征 rank1 0.88, 长特征rank1 0.91],相较与resnet50速度快很多,精度损失可以接受,比resnet大概低了3个点

(插眼:学了蒸馏以后,用蒸馏来训练一波...日后补上)

# 代价矩阵的计算方法
1. 求出feature_matrix
2. 求取目标的质心,判断相机是否抖动
   1. 如果大幅抖动,直接计distance_matrix为0
   2. 否则,进行简单计算distance_matrix,只考虑方框中心点是否位于100x100的范围内
      1. 在此范围内,则distance_cost计为0
      2. 否则,根据丢失跟踪的帧数计算出一个cost,丢失越久,cost越小
3. 将俩距离举证相加,得到最终的距离矩阵

# 观测发现
1. 实验数据的框不准确,且重叠较多,容易形成干扰
2. 有较多新检测器生存,要修正

# 快速开始
- 下载代码: `https://github.com/xiaoye-hhh/simple_deepsort.git`
- 下载数据集: `wget https://motchallenge.net/data/MOT15.zip`
- 解压: `unzip MOT15.zip`
- 改名: `mv MOT15 data`
- 下降权重文件: [谷歌链接](https://drive.google.com/file/d/1S9XW_oO0MixTO4f0ZoHAjcRMFopDKFmp/view?usp=sharing)
- 运行: `python main.py`
  