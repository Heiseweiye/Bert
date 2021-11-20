#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: heiseweiye
# time: 2021/11/20 12:42

import torch
import random
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import BertForMaskedLM
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pysnooper
from transformers import BertForSequenceClassification
import os
import pandas as pd

"""
制作一个可以用来读取训练 / 测试集的Dataset， 这是你需要彻底了解的部分
此 Dataset 每次将 tsv 里的一个成对句子转换成 BERT 相容的格式，并返回 3 个 tensors：
- tokens_tensor：两个句子合并后的索引序列，包含 [CLS] 和 [SEP]
- segments_tensor：可以用来区别两个句子的 binary tensor
- label_tensor：将分类标签转换成类别索引的 tensor，如果是测试集测 返回 None
"""
class FakeNewsDataset(Dataset):
    # 读取之前处理后的 tsv 并初始化一些参数
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]  # 一般训练你会需要 dev set
        self.mode = mode
        # 大数据你会需要用 iterator=True
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {'agreed': 0, 'disagreed': 1, 'unrelated': 2}
        self.tokenizer = tokenizer  # 我们将使用 BERT tokenizer

    # @pysnooper.snoop()  # 加入以了解所有转换过程

    # 定义一个训练 / 测试数据的函数
    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            # 将 label 文字也转换成索引，方便转换成 tensor
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)

        # 建立第一个句子的 BERT tokens 并加入分隔符 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)

        # 第二个句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a

        # 将整个 token 序列转换成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # 将第一句包含 [SEP] 的 token 位置设为 0，其他为 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b,
                                       dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.len

"""
制作可以一次返回一个 mini-batch 的 DataLoader
这个 DataLoader 吃我们上面定义的 `FakeNewsDataset`，
返回训练 BERT 时会需要的 4 个 tensors：
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
这个函数的输入 `samples` 是一個 list，里面的每个 element 都是
刚刚定义的 `FakeNewsDataset` 返回的一个样本，每个样本都包含 3 tensors：
- tokens_tensor
- segments_tensor
- label_tensor
它会对前两个 tensors 作 zero padding，并产生前面说明过的 masks_tensors
"""
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    # 测试集只有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    # zero pad 到同一序列长度
    tokens_tensors = pad_sequence(tokens_tensors,
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,
                                    batch_first=True)

    # attention masks，将 tokens_tensors 里面不为 zero padding
    # 的位置设为 1 让 BERT 只关注这些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape,
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids

"""
定义一个可以针对特定 DataLoader 取得模型预测结果以及分类准确度的函数
之后也可以用来生成上面的 Kaggle 竞赛的预测结果
在將 `tokens`、`segments_tensors` 等 tensors
丟入模型时，强力建议指定每个 tensor 对应的参数名称，以避免 HuggingFace
更新 repo 代码并改变参数顺序时影响到我们的结果。
"""
def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    with torch.no_grad():
        # 遍历整个资料集
        for data in dataloader:
            # 将所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]


            # 别忘记前 3 个 tensors 分別为 tokens, segments 以及 masks
            # 且强烈建议在将这些 tensors 丢入 `model` 时指定对应的参数名称
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors)

            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)

            # 用来计算训练集的分类准确率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            # 将当前的 batch 记录下来
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions


def main(SAMPLE_FRAC = 0.01, Train_BATCH_SIZE = 64, EPOCHS = 8, Test_Batch_Size=256):
    # 确定与训练模型
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    # 从csv文件读取数据
    df_train = pd.read_csv("./datasets/train.csv")
    empty_title = ((df_train['title2_zh'].isnull()) \
                   | (df_train['title1_zh'].isnull()) \
                   | (df_train['title2_zh'] == '') \
                   | (df_train['title2_zh'] == '0'))
    df_train = df_train[~empty_title]
    MAX_LENGTH = 30
    df_train = df_train[~(df_train.title1_zh.apply(lambda x : len(x)) > MAX_LENGTH)]
    df_train = df_train[~(df_train.title2_zh.apply(lambda x : len(x)) > MAX_LENGTH)]
    # 只用 1% 的训练集看看 BERT 对少量标注数据有多少帮助

    df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=9527)
    # 去除不必要的欄位並重新命名兩標題的欄位名
    df_train = df_train.reset_index()
    df_train = df_train.loc[:, ['title1_zh', 'title2_zh', 'label']]
    df_train.columns = ['text_a', 'text_b', 'label']
    # idempotence, 将处理结果另存成 tsv 供 PyTorch 使用
    df_train.to_csv("train.tsv", sep="\t", index=False)
    print("训练样本数：", len(df_train))

    # 初始化一个专门读取训练样本的 Dataset，使用中文 BERT 断词
    trainset = FakeNewsDataset("train", tokenizer=tokenizer)
    # 初始化一个每次返回 Batchsize 个训练样本的 DataLoader
    # 利用 `collate_fn` 将 list of samples 合并成一个 mini-batch 是关键
    trainloader = DataLoader(trainset, batch_size=Train_BATCH_SIZE, collate_fn=create_mini_batch)
    # 加载文本分类预训练模型
    NUM_LABELS = 3
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    # 该模型跑在 GPU 上并取得训练集的分类准确率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)
    # 训练模式
    model.train()
    # 使用 Adam Optim 更新整个分类模型的參數
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            tokens_tensors, segments_tensors, \
            masks_tensors, labels = [t.to(device) for t in data]
            # 将参数梯度归0
            optimizer.zero_grad()
            # forward pass
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors,
                            labels=labels)
            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()
            # 记录当前 batch loss
            running_loss += loss.item()
        # 计算分类准确率
        _, acc = get_predictions(model, trainloader, compute_acc=True)
        print('[epoch %d] loss: %.3f, acc: %.3f' %
              (epoch + 1, running_loss, acc))

    # 模型保存
    PATH = "./logs/my_net.pth"
    torch.save(model.state_dict(), PATH)
    # # 建立測試集。這邊我們可以用跟訓練時不同的 batch_size，看你 GPU 多大
    # testset = FakeNewsDataset("test", tokenizer=tokenizer)
    # testloader = DataLoader(testset, batch_size=Test_Batch_Size,
    #                         collate_fn=create_mini_batch)
    # # 用分类模型预测测试集
    # predictions = get_predictions(model, testloader)
    # # 用来将预测的 label id 转换为 label 文字
    # index_map = {v: k for k, v in testset.label_map.items()}
    # # 生成 Kaggle 提交文件
    # df = pd.DataFrame({"Category": predictions.tolist()})
    # df['Category'] = df.Category.apply(lambda x: index_map[x])
    # df_pred = pd.concat([testset.df.loc[:, ["Id"]],
    #                      df.loc[:, 'Category']], axis=1)
    # df_pred.to_csv('./logs/bert_1_prec_training_samples.csv', index=False)

if __name__ == '__main__':
    main(SAMPLE_FRAC=1, Train_BATCH_SIZE = 256, EPOCHS = 10)