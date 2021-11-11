import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# 注释：我已经把数据集改成简单的txt文件了，txt文件就是iris_data.txt，其中，Iris-setosa类用0表示，Iris-versicolor类用1表示，Iris-virginica类用2表示，“,”已经都替换为空格了
dataset = np.loadtxt('iris_data.txt')
# 打印dataset的大小，可以看出来dataset是一个150*5的矩阵
data = dataset[:,0:4]
target = dataset[:,4]
X_train, X_test, y_train, y_test = train_test_split( data, target, random_state=1)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
feat = []
print("分类器训练完成！可以进行预测：")
#对测试集进行预测
print("对测试集进行预测，预测准确率为:",knn.score(X_test, y_test)*100,"%")
print("请输入待预测的鸢尾花数据，四个数字，回车分隔：")
for i in range(4):
    feat.append(float(input()))
predict_result = knn.predict(np.array([[feat[0],feat[1],feat[2],feat[3]]]))
dic = {
    0:"setosa",
    1:"versicolor",
    2:"virginica"
}
print("预测的鸢尾花种类为：",dic[int(predict_result)])
# print(X_test)
