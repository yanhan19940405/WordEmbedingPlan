# WordEmbedingPlan
基于skip变式结构的负例采样词向量训练方案与基于IMDB数据的电影情感分类源码。技术栈：pytorch

在文件skipwordembedding.py中，其源码思路来源于ELMO处理方式，模型结构如图所示：

![图1 模型结构](https://upload-images.jianshu.io/upload_images/10738320-a8c8ad683b464e9d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/569/format/webp)

模型结构与普通skip W2V结构是有区别的，由于传统skip结构词向量训练是输入一个核心词，输出其前后相邻的词汇，但由于字很多，所以此思路训练需要大量计算开销，并不明智。
所以我在这里基于负例采样的思路进行更改，其训练思路引入正实例与负例，过程为：

1.首先对输入的句子层级embedding矩阵进行随机初始化，同时对句子进行BILSTM层序列特征提取。

2.假设有K个负例，分别对负例与正例进行embedding处理后，分别经过BILSTM层特征提取

3.对得到的负例文本信息张量与句子层级的张量进行matmul运算，正例亦然。

4.直到对数损失函数值达到最优即可

以上训练过程基于无监督的思路，模型训练好后变得到了最佳句子层级的embedding矩阵，后续即可用此矩阵进行迁移训练，进行使用，但由于个人算力有限，此处效果为进一步试验，有兴趣的读者可以试试。
。

在文件sentiment.py中，这里我用IMDB情感分类进行了pytorch实践。

