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


# 为了方便可视化，这里只画出二维结果
def svm_plot(save_fig_name,the_svm,x,y):
    x = mat(x).A
    y = np.array(y).reshape(len(y),)

    zero_dimension = x[:,0].reshape(len(x),1)
    one_dimension = x[:,1].reshape(len(x),1)
    w = the_svm.w[:2, 0]  # 前两个权重

    x1 = linspace(min(zero_dimension),max(zero_dimension),50)
    # refer: https://blog.csdn.net/qq_38412868/article/details/89363987
    x2 = (-w[0]/w[1])*x1 -the_svm.b/w[1]

    x1 = x1.reshape(len(x1),).tolist()
    x2 = x2.reshape(len(x2), ).tolist()

    fig = plt.figure(1)  # 创建图表

    plt.scatter(zero_dimension[y == -1], one_dimension[y == -1])  # 0维度和1维度可视化
    plt.scatter(zero_dimension[y == 1], one_dimension[y == 1])
    plt.scatter(x1, x2)
    plt.show()    #显示，但是会阻塞程序
    fig.savefig(save_fig_name)

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


    # SVM类实例化
    test_svm = SVM(datas,labels,C,toler,maxIter,kTup)

    # 训练
    test_svm.train()

    # 测试并显示准确率
    test_svm.test(test_data,test_label)

    # 预测
    #test_svm.predict(data)


    '''可视化'''
    # 原始训练数据 以及 分割效果
    #svm_plot('train_fig',test_svm,datas,labels)

    # 测试数据 以及 分割效果
    #svm_plot('test_fig',test_svm,test_data,test_label)
