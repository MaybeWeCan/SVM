#-*-coding:utf-8-*-
from SVM import *
import numpy as np

# 导入数据
def loadDataSet(fileName):
    """loadDataSet（对文件进行逐行解析，从而得到第行的类标签和整个数据矩阵）

    Args:
        fileName 文件名
    Returns:
        dataMat  数据矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        data = lineArr[:-1]
        data = [float(i) for i in data]
        dataMat.append(data)
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat


if __name__ == '__main__':

    # 参数
    train_file_name = 'svm_train.txt'
    test_file_name = 'svm_test.txt'

    C = 1
    toler = 0.0001
    maxIter = 10000
    #kTup = ('rbf', 3)    # 其他参数调整发现都无用，此处调整为3效果最佳，0.583333

    kTup = ('lin', 0)     # 第二个参数无用，为什么？因为定义线性核函数的时候没用到第二个参数，
                          # 此处正确率为：1，我惊了

    # 为了方便可视化，这里取前两维度数据训练
    datas,labels = loadDataSet(train_file_name)
    the_buffer = array(datas)
    datas = the_buffer[:,:2]

    test_data,test_label = loadDataSet(test_file_name)
    the_buffer = array(test_data)
    test_data = the_buffer[:,:2]

    n = np.shape(labels)
    print(n)

    # SVM类实例化
    test_svm = SVM(datas,labels,C,toler,maxIter,kTup)

    # 训练
    test_svm.train()

    # 测试并显示准确率
    test_svm.test(test_data,test_label)

    # 预测
    #test_svm.predict(data)
