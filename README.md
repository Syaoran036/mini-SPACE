# mini-SPACE
optimal SPACE (SPatial transcriptomics Analysis via Cell Embedding) for large dataset

original repo: https://github.com/zhangqf-lab/SPACE

**top level design**
1. 原始 SPACE 算法要求全梯度下降（Full gradient descent,FGD），需要一次性输入全部样本数据。这个要求在处理包含大量样本的数据集时并不现实。
2. 考虑到大数据集通常并不是一个样本，而是多个样本的集合，考虑样本与样本之间的细胞空间关系没有意义，因此我们可以考虑每次训练的时候输入一个样本，而非所有样本。
3. 基于上述思考，我们在每次 epoch 中对所有样本依次求损失并返回梯度。我们对学习率进行了相应调整。
