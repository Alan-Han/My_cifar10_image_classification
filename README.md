# My_cifar10_image_classification
本项目对 CIFAR-10 数据集进行图像分类，该数据集包含飞机，狗，猫等类型的图片。

## 1.	运行环境 

ios 10.13  python 3.5.4  tensorflow 1.2.0


## 2. 获取数据集并预处理：

(1)运行主程序main.py后，调用input_dataset.py自动下载数据集，数据集包含10000张32x32的RGB图片及对应类别标签，有飞机、狗、猫等10个类别，下载地址为https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

(2)数据集下载后运行input_dataset.cifar10_input()函数解压数据，数据分为5个训练集和1个测试集，以.p文件存放在cifar-10-batches-py文件夹中

(3)运行helper.preprocess_and_save_data进行预处理，将图片数据归一化至0~1，将label进行one hot encode，取训练集的1/10作为验证集


## 3. 训练模型

(1)模型存放在cnn.py，CNN网络结构为：卷积+池化+卷积+池化+全连接+dropout+全连接+dropout

(2)主程序中运行train()开始训练，设置optimizer = tf.train.AdamOptimizer().minimize(cost)，训练200次，训练时全连接层的dropout概率为0.5

(3)训练完毕将最优模型参数保存至result文件夹中


## 4. 测试

主程序运行test_model()开始测试，调用保存的模型并对测试集进行测试，最终准确率达到70%
