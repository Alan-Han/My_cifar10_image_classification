# My_cifar10_image_classification
本项目对 CIFAR-10 数据集进行图像分类，该数据集包含飞机，狗，猫等类型的图片。  
主程序入口为main.py  
首先对数据集进行预处理，然后用卷积神经网络对所有样本进行训练，训练前先将图像归一化（normalize），对标签进行独热编码（one-hot encode the labels），然后建立卷积层、最大池化层和全连接层。最后可以看到该模型对样本图像作出的预测结果，测试集上正确率达到70%

# My_cifar10_image_classification
本项目对 CIFAR-10 数据集进行图像分类，该数据集包含飞机，狗，猫等类型的图片。

## 1.	运行环境 

ios 10.13  python 3.5.4  tensorflow 1.2.0


## 2. 获取数据集并预处理：

(1)运行主程序main.py后，调用input_dataset.py自动下载数据集，数据集包含10000张32x32的RGB图片及对应类别标签，有飞机、狗、猫等10个类别，下载地址为https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

(2)数据集下载后运行input_dataset.cifar10_input()函数解压数据，数据分为5个训练集和1个测试集，以.p文件存放在cifar-10-batches-py文件夹中

(3)运行helper.preprocess_and_save_data进行预处理，将图片数据归一化至0~1，将label进行one hot encode，取训练集的1/10作为验证集


## 3. 模型

(1)模型存放在cnn.py，网络结构为2个卷积池化层连接2个全连接层，每个全连接层后采用dropout

(2)主程序中运行train()，

(3)我们队伍受resnet结构的启发，对bilinear cnn算法做了改进，将最后一层卷积核的输出也和前面其他层的卷积核的输出做内积，以此达到融合不同层次的特征的目的。再把得到的vector和原来的bilinear vector 融合。 我们增加了conv4_1、conv5_1对conv5_3的内积（只增加这两层是因为他们的filter numbers数量一致，pooling之后就可以做内积了，不需要加额外的卷积核）
我们的思想是：不同卷积层关注的特征不同，且对应感受视野的大小也不同（即有高低层次之分），在识别类似图像时，单独考虑特征是不够的，还需要考虑他们之间的空间关系。

(4)加载预训练的vgg模型，先训练全连接层，之后再训练整个网络。预训练权重下载地址https://www.cs.toronto.edu/~frossard/post/vgg16/

(5)训练过程中加入实时的数据增强，包括旋转、随机改变对比度、随机改变亮度、随机crop. 训练时全连接层的drop out概率为0.5


## 4. 结构

(1)train/read_data.py 是读取数据的结构。实现大数据的分次加载。

(2)train/resvgg_model.py定义了网络结构，以及读取保存的权重的方法

(3)train/train_resvgg.py定义了训练的过程

(4)train/predict_resvgg.py 输出预测结果

## 5. 加载预训练模型，微调

(1)在读取resvgg模型时，令finetune=False,实现只训练最后的全连接层。并且调用load_initial_weights(sess)，读取预训练的vgg的卷积层的参数

(2)训练设置 optimizer = tf.train.MomentumOptimizer(learning_rate=0.2, momentum=0.5).minimize(loss)，训练次数50次

(3)将过程中得到的最优模型保存下来

## 6. 全网络训练

(1)在读取resvgg模型时，令finetune=True。 调用load_own_weight(sess , model_path)，读取上一步得到的模型

(2)训练设置optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)， 训练200次

(3)将过程中得到的最优模型保存下来


## 7. 后期调整

实际训练过程中，只有第一次会在所有数据上训练满200次。在得到保存下来的模型后，之后的调参过程只取大约1/4的数据进行继续训练

## 8. 预测

(1)运行 predict_resvgg.py 预测结果
