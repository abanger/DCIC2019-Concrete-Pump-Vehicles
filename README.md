# DCIC2019-Concrete-Pump-Vehicles
混凝土泵车砼活塞故障预警分析

## 前言
初学者，由于工作时间比较紧，准备上传个人一些想法，欢迎指正交流。

## 行业知识

泵车环境及运行参数是重要因素，赛题没有提供，只能从互联网获取，没有深入学习，提供以下供参考：
- 发动机2200-2300，一般转速在1000~3500之间，怠速可达到700。
- 液压油温度30℃-70℃属于正常，工作温度大约为60度左右，一般不会超过85度，超高容易坏。最高140℃，30-80度之间 液力传动油，适宜的粘度和良好的粘温性能。有些液力传动油传动装置在-40～170℃温度范围内正常工作。
- 输送泵功率大多在65--90kw之间。
- 90kw混凝土地泵的额定电流171A，地泵启动电流大约是171A（5-7倍）=855A——1197A。
- 排量和理论流量之间的关系是: qt=nV 式中n-一液压油泵的转速;V一一液压油泵的排量

注：网上查来，不确信准确，欢迎补充。

## 说明
- data_proc001_type.py，在参考[3]取得0.619
- data_train2cnn100.py，训练数据转为图片,cnn100.py处理，结果acc0.61左右，没有GPU，所以没进一步研究 
- data_proc005.py，对一些行业知识了解产生特征，一个方向供参考
- data_proc200.py，行业知识和常识产生特征（不准确判断），数据处理lgb->xgb,最终B为0.62530094,阈值待优化。

## 主要参考
1. [DCIC-Failure-Prediction-of-Concrete-Piston-for-Concrete-Pump-Vehicles](https://github.com/jmxhhyx/DCIC-Failure-Prediction-of-Concrete-Piston-for-Concrete-Pump-Vehicles)
2. [混凝土泵车砼活塞故障预警](https://github.com/tianshuaifei/dcic_2019)
3. [Data Fountain光伏发电量预测 Top1 开源分享](https://zhuanlan.zhihu.com/p/44755488?utm_source=qq&utm_medium=social&utm_oi=623925402599559168)

## 讨论
[留言](https://github.com/abanger/DCIC2019-Concrete-Pump-Vehicles/issues)