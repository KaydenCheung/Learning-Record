大家好，今天我要分享的是一篇来自SIGIR2020的论文，基于全局上下文增强的图神经网络会话推荐。会话推荐在之前的主流模型是RNN，因为每一个会话都是一个物品的转移序列，但是RNN只对连续单向的转移关系进行了建模。而近些年来GNN也逐渐开始在会话推荐上使用起来，它的目的是将会话构建成一个物品转移图，从而可以探索物品之间复杂的转移关系。



作者认为之前的方法还存在着一些不足：首先是现在的方法都仅仅关注于会话的内部信息，而忽略了跨会话的信息。在这张图中，以session3中的item3耳机为例，如果不考虑跨会话信息，那么只有item8和item9会对其产生影响。那在session1中，用户可能是因为要买耳机而浏览了该耳机，在session2中用户可能因为喜欢苹果而浏览了该耳机。那这两个会话中的物品转移关系都可以作为第三个会话的补充信息。当然，这个转移关系应当和当前的会话是比较接近的，否则就是引入噪声了。

另外之前基于GNN的方法没有考虑会话中物品的顺序，所以这篇论文引入了位置信息。



这是整个模型的框架，分为4个部分，黄色部分是在所有会话上学习物品的表示，红色部分是在单个会话上学习物品的表示，蓝色部分是将这两部分学到的物品表示结合起来生成会话表示，绿色部分就是预测层。

因为物品的表示分为了两部分，所以需要构建两个图，分别是Session graph和Global Graph。Session Graph为有向图，将会话中相邻的节点相连，并且添加自连接。Global Graph针对所有会话建图，其为带权无向图，我们设定一个距离d，将会话中距离在d之内的节点相连，其权重根据节点在所有会话共现的频率决定。

从全局图上学习物品表示的好处是它能够借鉴其它会话中的有用信息，所以在这一部分的关键就是如何衡量全局图上的物品是否和当前会话相关，所以这里将当前会话s、物品j以及物品i和j之间的权重考虑起来计算一个权重，以此衡量邻居节点和当前会话的相关程度。聚合部分就是邻居信息和自身信息拼接起来做一个非线性变换。

这部分就是在单个会话上从目标节点的邻居节点中提取信息，也是运用注意力机制，然后因为这是有向图，所以作者在公式中考虑了不同的边类型。

这部分是计算会话表示，首先将之前得到的两个结果相加，然后加入位置信息，这里用的是反向位置信息，越靠后的物品应该越重要。这里的话，因为会话长度是不固定的，用前向位置信息无法衡量当前物品和预测物品之间的距离，但反向位置信息可以。

最后通过平均池化得到会话的信息，然后用注意力机制计算会话中各个物品的权重。最后按权重求和就可以得到会话的表示。

预测层的话就是做内积计算。

实验部分采用了三个数据集：第一个是某电商网站5个月的点击流数据；第二个是天猫数据库中的消费者购物数据；第三个是推特上收集的音乐收听行为。SR-GNN和FGNN模型上的结果说明了GNN在会话推荐上的有效性，而GCE-GNN通过集成跨会话信息，会话上下文信息和位置信息，获得了更好的性能。

这部分是消融实验，首先探究全局级的特征编码器的作用，第一个表示删去了全局级的特征编码器，第二个表示删去了会话级的特征编码器。可以看到删去全局级的特征编码器后会有下降，但下降的幅度也不大。第二个就是位置向量的作用，分别和采用前向位置信息和self-attention做对比，可以看到采用反向位置信息可以有很大的提升。



我觉得这篇论文比较不错的有，为了从全局图中提取相关的信息，作者使用注意力机制来自适应的选择相关的信息；另外在计算会话表示时，之前的文章都是使用会话的最后一个物品去和每个物品做attention，而这篇论文是用浓缩后的会话信息去做attention，并且加入了位置信息。







