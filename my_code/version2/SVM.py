#-*-coding:utf-8-*-
# refer: https://github.com/apachecn/AiLearning

from numpy import *
import matplotlib.pyplot as plt
import time
class SVM:
    def __init__(self,datas,labels,C,toler,maxIter,kTup):
        '''
        :param datas: 数据集
        :param labels: 标签
        :param C: 松弛变量
        :param toler: 容错率
        :param kTup: 核函数元组
        '''
        self.X = mat(datas)
        self.m, self.n = shape(datas)  # 行数
        labels = array(labels).reshape(self.m,1)
        self.labels = mat(labels)
        self.C = C
        self.toler = toler
        self.kTup = kTup
        self.maxIter = maxIter

        self.alphas = mat(zeros((self.m,1))) #初始化为0
        self.b = 0
        self.w = mat(zeros((self.n,1)))

        # echae:(有效位,E值)
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))

        for i in range(self.m):
            self.K[:,i] = self.kernelTrans(self.X,self.X[i])

    def kernelTrans(self,X,A):
        '''
        :param X: 数据集
        :param A: 数据集内第i行数据
        '''
        m,n = shape(X)
        k = mat(zeros((m,1)))
        if self.kTup[0] == 'lin':
            # linear kernal: m*n * n * 1 = m*1
            k = X*A.T
        elif self.kTup[0] == 'rbf':
            for j in range(m):
                deltaRow = X[j,:] - A
                k[j] = deltaRow*deltaRow.T
            # 径向基函数高斯版本
            k = exp(k/(-2*self.kTup[1]**2))
        else:
            raise NameError('Sorry,That Kernel is not recognized')
        return k


    def calcEk(self,k):
        ''' calcEk（求 Ek误差：预测值-真实值的差）
        该过程在完整版的SMO算法中陪出现次数较多，因此将其单独作为一个方法
        :param k: 具体的某一行
        :return:  Ek  预测结果与真实结果比对，计算误差Ek
        '''
        # mat的multiply是点积
        fxk = multiply(self.alphas,self.labels).T*self.K[:,k]+self.b
        Ek = fxk - float(self.labels[k])
        return Ek

    def selectJrand(self,i):
        """
        随机选择一个整数
        Args:
            i  第一个alpha的下标
        Returns:
            j  返回一个不为i的随机数，在0~m之间的整数值
        """
        m = len(self.alphas)
        j = i
        while j == i:
            j = random.randint(0, m - 1)
        return j

    def selectJ(self,i,Ei):
        '''selectJ（返回最优的j和Ej）
            内循环的启发式方法。
            选择第二个(内循环)alpha的alpha值
            这里的目标是选择合适的第二个alpha值以保证每次优化中采用最大步长。
            该函数的误差与第一个alpha值Ei和下标i有关。
        :param i:  具体的第i一行
        :param Ei: 预测结果与真实结果比对，计算误差Ei
        :return:
                j  随机选出的第j一行
                Ej 预测结果与真实结果比对，计算误差Ej
        '''
        maxk = -1
        maxDeltaE = 0
        Ej = 0
        # 首先将输入值Ei在缓存中设置成为有效的。这里的有效意味着它已经计算好了。
        self.eCache[i] = [1,Ei]
        # .A代表将 矩阵转化为array数组类型
        validEcacheList = nonzero(self.eCache[:, 0].A)[0]

        if (len(validEcacheList)) > 1:
            for k in validEcacheList:  # 在所有的值上进行循环，并选择其中使得改变最大的那个值
                if k == i:
                    continue  # don't calc for i, waste of time

                # 求 Ek误差：预测值-真实值的差
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    # 选择具有最大步长的j
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:  # 如果是第一次循环，则随机选择一个alpha值
            j = self.selectJrand(i)
            # 求 Ek误差：预测值-真实值的差
            Ej = self.calcEk(j)
        return j, Ej

    def updateEk(self,k):
        """updateEk（计算误差值并存入缓存中。）

        在对alpha值进行优化之后会用到这个值。
        Args:
            k   某一列的行号
        """

        # 求 误差：预测值-真实值的差
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]

    def clipAlpha(self,aj, H, L):
        """clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
        Args:
            aj  目标值
            H   最大值
            L   最小值
        Returns:
            aj  目标值
        """
        aj = min(aj, H)
        aj = max(L, aj)
        return aj

    def innerL(self,i):
        """innerL
        内循环代码
        Args:
            i   具体的某一行
        Returns:
            0   找不到最优的值
            1   找到了最优的值，并且self.Cache到缓存中
        """

        # 求 Ek误差：预测值-真实值的差
        Ei = self.calcEk(i)

        # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
        # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
        # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
        '''
        # 检验训练样本(xi, yi)是否满足KKT条件
        yi*f(i) >= 1 and alpha = 0 (outside the boundary)
        yi*f(i) == 1 and 0<alpha< C (on the boundary)
        yi*f(i) <= 1 and alpha = C (between the boundary)
        '''
        # kkT条件有点不解.....上面的写法跟下面这个条件一样？？？？
        #print(Ei)

        if ((self.labels[i] * Ei < -self.toler) and (self.alphas[i] < self.C)) or \
                ((self.labels[i] * Ei > self.toler) and (self.alphas[i] > 0)):

            # 选择最大的误差对应的j进行优化。效果更明显
            j, Ej = self.selectJ(i,Ei)

            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()

            # 计算：L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
            if (self.labels[i] != self.labels[j]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                # print("L==H")
                return 0

            # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
            # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
            eta = self.K[i, j] - self.K[i, i] - self.K[j, j]  # changed for kernel
            if eta >= 0:
                print("eta>=0")
                return 0

            # 计算出一个新的alphas[j]值,注意 ‘ -= ’
            self.alphas[j] -= self.labels[j] * (Ei - Ej) / eta
            # 并使用辅助函数，以及L和H对其进行调整
            self.alphas[j] = self.clipAlpha(self.alphas[j], H, L)
            # 更新误差缓存
            self.updateEk(j)

            # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
            if abs(self.alphas[j] - alphaJold) < 0.00001:
                # print("j not moving enough")
                return 0

            # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
            self.alphas[i] += self.labels[j] * self.labels[i] * (alphaJold - self.alphas[j])
            # 更新误差缓存
            self.updateEk(i)

            # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
            # w= Σ[1~n] ai*yi*xi => b = yi- Σ[1~n] ai*yi(xi*xj)
            # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
            # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍

            b1 = self.b - Ei - self.labels[i] * (self.alphas[i] - alphaIold) * self.K[i, i] - self.labels[j] * (
                        self.alphas[j] - alphaJold) * self.K[i, j]
            b2 = self.b - Ej - self.labels[i] * (self.alphas[i] - alphaIold) * self.K[i, j] - self.labels[j] * (
                        self.alphas[j] - alphaJold) * self.K[j, j]

            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2
            return 1

        else:
            return 0

    def smoP(self):
        """
        完整SMO算法外循环
        """

        iter = 0
        entireSet = True      # 是否边界
        alphaPairsChanged = 0

        # 循环遍历：循环maxIter次 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
        while (iter < self.maxIter) and ((alphaPairsChanged > 0) or (entireSet)):

            alphaPairsChanged = 0

            #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
            if entireSet:

                # 在数据集上遍历所有可能的alpha
                for i in range(self.m):
                    # 是否存在alpha对，存在就+1
                    alphaPairsChanged += self.innerL(i)
                    # print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))

                iter += 1

            # 对已存在 alpha对，选出非边界的alpha值，进行优化。
            else:
                # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
                nonBoundIs = nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerL(i)
                    # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1

            # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
            if entireSet:
                entireSet = False  # toggle entire set loop
            elif alphaPairsChanged == 0:
                entireSet = True
            print("iteration number: %d" % iter)

    # 根据公式计算: W
    def calcWs(self):
        """
        基于alpha计算w值
        Returns:
            wc  回归系数
        """
        X = mat(self.X)

        m, n = shape(X)
        w = zeros((n, 1))

        for i in range(m):
            w += multiply(self.alphas[i] * self.labels[i], X[i].T)
        return w

        # 为了方便可视化，这里只画出二维结果
    def plot_original(self):
        x = self.X.A
        y = self.labels.A
        #positive_index = [i for i in range(len(y)) if y[i] == 1]
        #negtive_index = [i for i in range(len(y)) if y[i] == -1]

        zero_dimension = x[:,0].reshape(len(y),1)
        one_dimension = x[:,1].reshape(len(y),1)

        fig = plt.figure(1)  # 创建图表
        plt.scatter(zero_dimension[y == -1], one_dimension[y == -1])  # 0维度和1维度可视化
        plt.scatter(zero_dimension[y == 1], one_dimension[y == 1])
        #plt.show()    #显示，但是会阻塞程序
        fig.savefig('原始数据分布图片')


    def train(self):

        # 显示原始分布,画图函数会阻塞程序
        #self.plot_original()

        # smoP算法
        self.smoP()

        # 计算权重
        self.w = self.calcWs()

    def plot_weight(self):
        x = self.X.A
        y = self.labels.A
        # positive_index = [i for i in range(len(y)) if y[i] == 1]
        # negtive_index = [i for i in range(len(y)) if y[i] == -1]

        zero_dimension = x[:, 0].reshape(len(y), 1)
        one_dimension = x[:, 1].reshape(len(y), 1)

        w = self.w[:2,0]  #前两个权重
        x1 = linspace(min(zero_dimension),max(zero_dimension),50)
        # refer: https://blog.csdn.net/qq_38412868/article/details/89363987
        x2 = (-w[0]/w[1])*x1 -self.b/w[1]

        x1 = x1.reshape(len(x1),).tolist()
        x2 = x2.reshape(len(x2), ).tolist()

        fig = plt.figure(1)  # 创建图表
        plt.scatter(zero_dimension[y == -1], one_dimension[y == -1])  # 0维度和1维度可视化
        plt.scatter(zero_dimension[y == 1], one_dimension[y == 1])
        plt.scatter(x1,x2)

        plt.show()  # 显示，但是会阻塞程序
        fig.savefig('分割效果展示')

    def plot_test(self,test_data,test_label):
        x = test_data.A
        y = mat(test_label).A.reshape(len(test_label),)


        # positive_index = [i for i in range(len(y)) if y[i] == 1]
        # negtive_index = [i for i in range(len(y)) if y[i] == -1]

        zero_dimension = x[:, 0].reshape(len(test_data), )
        one_dimension = x[:, 1].reshape(len(test_data), )

        w = self.w[:2,0]  #前两个权重
        x1 = linspace(min(zero_dimension),max(zero_dimension),50)
        x2 = (-w[0]/w[1])*x1 -self.b/w[1]

        x1 = x1.tolist()
        x2 = x2.tolist()

        fig = plt.figure(1)  # 创建图表
        plt.scatter(zero_dimension[y == -1], one_dimension[y == -1])  # 0维度和1维度可视化
        plt.scatter(zero_dimension[y == 1], one_dimension[y == 1])
        plt.scatter(x1,x2)

        plt.show()  # 显示，但是会阻塞程序
        fig.savefig('分割效果展示')

    def accuracy(self,a,b):
        # 计算时由于list和mat数据不兼容，只能采取转换
        a = a.reshape(1,len(b))
        a = a.tolist()[0]
        b = list(b)
        c = [a[i] - b[i] for i in range(len(a))]
        correct_number = len([i for i in c if i == 0])
        the_accuracy = float(correct_number/len(a))
        return the_accuracy

    def test(self,test_data,real_label):

        # 显示分割效果
        #self.plot_weight()

        test_data = mat(test_data)
        test_label = test_data * self.w + self.b
        test_label = sign(test_label)
        the_accuracy = self.accuracy(test_label,real_label)
        print('The accuracy is',the_accuracy)
        self.plot_test(test_data,real_label)

    def predict(self,predict_data):
        predict_data = mat(predict_data)
        predict_label = predict_data * self.w + self.b
        return predict_label






