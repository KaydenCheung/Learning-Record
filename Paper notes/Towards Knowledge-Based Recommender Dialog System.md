### 阅读笔记



该论文提出了一个称为KBRD的框架，它是一个基于知识图谱的推荐对话系统。



首先需要做的就是知识图谱的实体链接，论文中使用到了DBpedia，将对话中出现的items以及non-item entities与DBpedia中的实体链接起来。通过知识图谱，我们也许可以发现这些non-items的内容可能与某些items联系紧密，这样可以起到丰富对话内容的作用，起到更好的推荐效果。

这样一来就可以将用户表示为$ \mathcal{T}_u = \{{e_1,e_2,···,e_{|\tau_u|}}\},  e_i\in\varepsilon $,  其中$ e_i $代表了对话中链接到图谱的items以及non-item entities。



那么接下来的问题就是怎么来表示DBpedia中的实体呢？论文中采用了R-GCN算法来获得图中每个节点的特征表示
$$
h_{v}^{(l+1)} = \sigma(\sum_{r\in\mathcal R} \sum_{w\in\mathcal N_v^r} \frac{1}{c_{v,r}} W_r^{(l)}h_w^{(l)} + W_0^{(l)}h_v^{(l)})
$$
所以在最后一层$ L $，可以得到  $ \textbf H^{(L)} \in \mathbb{R}^{|\varepsilon|\times d^{(L)}}$。注：后面$d^{(L)}$简写成$d$。



在得到DBpedia中的实体特征表示矩阵后，上文中所提到的$ \mathcal{T_u}$也可以用向量的形式表示出来了，即
$$
\textbf{H}_u = (h_1,h_2,···,h_{|\tau_u|})
$$
其中$h_i\in\mathbb{R}^d$是实体$e_i$的特征表示。



但是现在又出现了新的问题，就是每个用户的对话内容大小不一，对应的$|\tau_u|$也是不相同的，论文并没有采用简单线性相加的方式，而是采用了self-attention来将不同大小的$\textbf{H}_u$转换成固定大小的向量表示。需要注意的是，论文中所采用的self-attention，并不是《Attention is all you need》论文中所提到的self-attention，而是来自《A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING》。

应用论文中所提到的方法：
$$
\alpha_u = softmax(w_{a2}tanh(W_{a1}\textbf{H}_u^T))
$$
其中$W_{a1}\in \mathbb{R}^{d_a \times d}$是一个权重矩阵，$w_{a2}$是一个维度为$d_a$的向量。

据此得到的$\alpha_u$是一个维度为$|\mathcal{T}_u|$的向量，代表了对话中不同实体的权重，每个用户就可以用$t_u = \alpha_u\textbf{H}_u$来表示，其表示的内容就是用户的偏好。

最后通过用户偏好与图谱中的实体做内积，越相似的话则内积越大，推荐的概率也就越高，即
$$
P_{rec} = softmax(mask(t_u\textbf{H}^T))
$$
其中的mask的作用是将non-item的实体的分数置为$-\infty$，因为最后要推荐的是item。



论文的另一重点就是对话模块了，论文中采用的是transformer框架，根据收集到的推荐对话训练集对该模型进行训练，在decoder中每个时间步都会有结果$o$输出。

于是在decoder的最后一层的每一个时间步，就可以输出词汇的概率分布，即
$$
P_{dialog} = softmax(Wo+b)
$$
这里$W\in\mathbb{R}^{|V|\times d}$，$b\in\mathbb{R}^{|V|}$，$P_{dialog}$是一个维度为$|V|$，即词汇表大小的向量。



但是，如果只是这样的话，那么对话模块就没有和推荐模块结合起来，在推荐模块中，我们已经学到了用户的偏好$t_u$，如果在对话模块中加入$t_u$，那么是不是就会更好的生成符合用户偏好的对话?

于是，论文中的做法就是首先经过一个前馈神经网络
$$
b_u = \mathcal{F}(t_u)
$$
其中$\mathcal{F}:\mathbb{R}^d \to \mathbb{R}^{|V|}$，这里也是先将维度统一一下。

于是，概率输出变成
$$
P_{dialog} = softmax(Wo+b + b_u)
$$


最后如何确定在每一步应该输出vocabulary还是item呢？论文中采用的方法是
$$
P(w) = p_sP_{dialog}(w) + (1-p_s)P_{rec}(w)
\\p_s = \sigma(w_so+b_s)
$$
其中$w$代表的是vocabulary或者是item，$o$是transformer最后一层的输出，$w_s\in\mathbb{R}^d$，$b_s\in\mathbb{R}$。其实这里我还不是怎么懂。。。



最后，再来回顾一下KBRD框架：

![QQ截图20201030181900.png](https://i.loli.net/2020/10/30/vQ6GbHkr1XCeVZq.png)

之前的左边框架没有很好的将对话模块和推荐模块结合起来，而KBRD很好的弥补了这个缺点。首先对话模块将内容与知识图谱联系起来，将更丰富的内容提供给了推荐模块，使得它可以给出更加符合用户偏好的推荐。同时推荐模块可以将学到的用户偏好传递给对话模块，使之可以生成更符合用户偏好的对话。








