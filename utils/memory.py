 
import numpy as np
import torch
import pdb #pdb.set_trace()
class OurMemory(object):
    def __init__(self, n, dim,p):
        self.num_memory=5 #*self.num_memory
        self.batch_size=p['batch_size']
        self.n = p['img_sum']
        self.dim = dim
        self.features = (torch.zeros(self.n, self.dim),torch.zeros(self.n, self.dim))
        self.ptr = 0
        self.image_augmented_num=p['image_augmented_num']
        self.first=0

    def update(self, f1,f2):#,targets
        b = f1.size(0)

        assert (b + self.ptr <= self.n)

        self.features[0][self.ptr:self.ptr + b].copy_(f1.data.detach())
        self.features[1][self.ptr:self.ptr + b].copy_(f2.data.detach())
        #self.targets[self.ptr:self.ptr + b].copy_(targets.detach())

        self.ptr += b
        if (self.n-self.ptr)<self.batch_size:
            self.ptr=0
    def update1(self, features0,features1):
        self.store=features0.data,features1.data
    def reset(self):
        self.ptr = 0

    def get_feature1(self, f0,f1):
        if self.first == 0:
            self.first = 1
            tp = f0,f1
            return tp

        tp = self.store
        return tp
    def get_feature(self):

        tp = self.features
        return tp

    def loss(self,f,f1):
        self.features.requires_grad=False
        proj=torch.matmul(f, self.features.T)
        proj1 = torch.matmul(f1, self.features.T)
        loss0=torch.nn.KLDivLoss(size_average=False, reduce=False)(proj,proj1).sum()
        return loss0
        '''这里要实现的功能是 f 与f1 分别于feature进行内积，'''


    def to(self, device):
        self.features = (self.features[0].to(device),self.features[1].to(device))

        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')
class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        '''len(base_dataset), p['model_kwargs']['features_dim'],  p['num_classes'], p['criterion_kwargs']['temperature']'''
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 10 #100
        self.temperature = temperature
        self.C = num_classes    # 按照我们的指标

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        #predictions.shape  torch.Size([128, 128])    (128,5000)
        correlation = torch.matmul(predictions, self.features.t())  #将128张图片的output 分别与  5000张图片的output相乘  得到一致性
        #correlation.shape  torch.Size([128, 5000])   (128,5000)
                                                                            #下面的公式挑出 这5000张图片 在model上与 128中的每张图片 关联性最高的K张图片
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)  #返回该变量中 K个在dim维最大的值， yi为下标 yd为所指定下标的数值
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        #targets.shape     torch.Size([5000])
        #candidates.shape  torch.Size([128, 5000])
        #yi.shape       torch.Size([128, 100])  (1,100)
        # yi.shape       torch.Size([128, 100])
        retrieval = torch.gather(candidates, 1, yi)  #按照索引yi 将 每张图片聚类调出来的张100图片 所对应的的target拍好序
        #retrieval.shape   torch.Size([128, 100])
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()  #造一个 one-hot所需要的0阶矩阵的尺寸
        #retrieval_one_hot.shapetorch.Size([12800, 10])
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)    # 将 类别由数字转化为序列  如2  ：    00100        1：   00010
        #retrieval_one_hot.shapetorch.Size([12800, 10])
        yd_transform = yd.clone().div_(self.temperature).exp_() #softmax的功能函数！！！！
        #yd_transform.shape  torch.Size([128, 100])

        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), #（128，100，10）
                          yd_transform.view(batchSize, -1, 1)), 1)  #yd（128,100,1）    算出
        #probs.shape         torch.Size([128, 10])
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred


    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()   #这里的feature都是通过model之后的输出   而且是有相对应的target的  也就是这个数据集是已知的
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            '''print("targets.shape" + str(targets.shape))
            print("targets" + str(targets))
            pdb.set_trace()
            '''
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval

            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)

            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)

        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device
    def get_feature(self):

        tp = self.features
        return tp
    def get_target(self):

        tp = self.targets
        return tp
    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')
 