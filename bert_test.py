#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: heiseweiye
# time: 2021/11/20 12:42
"""
這段程式碼載入已經訓練好的 masked 語言模型並對有 [MASK] 的句子做預測
"""
import torch
import random
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import BertForMaskedLM

PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
# vocab = tokenizer.vocab
#
# random_tokens = random.sample(list(vocab), 10)
# random_ids = [vocab[t] for t in random_tokens]
#
# text = "[CLS] 我饿了，我想[MASK]汉堡。"
# tokens = tokenizer.tokenize(text)
# ids = tokenizer.convert_tokens_to_ids(tokens)
#
# # 除了 tokens 以外我们还需要区分句子的 segment ids
# # tokens_tensor 存放一每个字的ids
# # 加载mask模型
# tokens_tensor = torch.tensor([ids])  # (1, seq_len)
# segments_tensors = torch.zeros_like(tokens_tensor)  # (1, seq_len)
# maskedLM_model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
#
# # 使用 masked LM 估计 [MASK] 位置所代表的实际 token
# maskedLM_model.eval()
# with torch.no_grad():
#     outputs = maskedLM_model(tokens_tensor, segments_tensors)
#     predictions = outputs[0]
#     # (1, seq_len, num_hidden_units)
# del maskedLM_model
#
# # 將 [MASK] 位置的機率分佈取 top k 最有可能的 tokens 出來
# # probs存放预测的mask中的前三的概率，indices存放前三的ids
# # predicted_tokens存放前三的token
# masked_index = 7
# k = 3
# probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k)
# predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
#
# # 顯示 top k 可能的字。一般我們就是取 top 1 當作預測值
# print("輸入 tokens ：", tokens[:10], '...')
# print('-' * 50)
# for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
#     tokens[masked_index] = t
#     print("Top {} ({:2}%)：{}".format(i, int(p.item() * 100), tokens[:10]), '...')

"""
前處理原始的訓練數據集。
你不需了解細節，只需要看註解了解邏輯或是輸出的數據格式即可
"""

import os
import pandas as pd

# 解压Kaggle竞赛下载的训练集
# os.system("unzip train.csv.zip")

# 简单的数据清理，去除空白标题的 examples
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
SAMPLE_FRAC = 0.01
df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=9527)

# 去除不必要的欄位並重新命名兩標題的欄位名
df_train = df_train.reset_index()
df_train = df_train.loc[:, ['title1_zh', 'title2_zh', 'label']]
df_train.columns = ['text_a', 'text_b', 'label']

# idempotence, 将处理结果另存成 tsv 供 PyTorch 使用
df_train.to_csv("train.tsv", sep="\t", index=False)

print("训练样本数：", len(df_train))

# df_train.label.value_counts() / len(df_train)
# unrelated    0.679338
# agreed       0.294317
# disagreed    0.026346
# Name: label, dtype: float64

# df_train.head()
# text_a                          text_b      label
# 0       苏有朋要结婚了，但网友觉得他还是和林心如比较合适  好闺蜜结婚给不婚族的秦岚扔花球，倒霉的秦岚掉水里笑哭苏有朋！  unrelated
# 1  爆料李小璐要成前妻了贾乃亮模仿王宝强一步到位、快刀斩乱麻！  李小璐要变前妻了？贾乃亮可能效仿王宝强当机立断，快刀斩乱麻！     agreed
# 2  为彩礼，母亲把女儿嫁给陌生男子，十年后再见面，母亲湿了眼眶  阿姨，不要彩礼是觉得你家穷，给你台阶下，不要以为我嫁不出去！  unrelated
# 3         猪油是个宝，一勺猪油等于十副药，先备起来再说  传承千百的猪油为何变得人人唯恐避之不及？揭开猪油的四大谣言！  unrelated
# 4                  剖析：香椿，为什么会致癌？            香椿含亚硝酸盐多吃会致癌？测完发现是谣言  disagreed

df_test = pd.read_csv("./datasets/test.csv")
df_test = df_test.loc[:, ["title1_zh", "title2_zh", "id"]]
df_test.columns = ["text_a", "text_b", "Id"]
df_test.to_csv("test.tsv", sep="\t", index=False)

print("预测样本数：", len(df_test))
# print(df_test.head())
# text_a                       text_b      Id
# 0  萨拉赫人气爆棚!埃及总统大选未参选获百万选票 现任总统压力山大  辟谣！里昂官方否认费基尔加盟利物浦，难道是价格没谈拢？  321187
# 1              萨达姆被捕后告诫美国的一句话，发人深思    10大最让美国人相信的荒诞谣言，如蜥蜴人掌控着美国  321190
# 2    萨达姆此项计划没有此国破坏的话，美国还会对伊拉克发动战争吗          萨达姆被捕后告诫美国的一句话，发人深思  321189
# 3              萨达姆被捕后告诫美国的一句话，发人深思  被绞刑处死的萨达姆是替身？他的此男人举动击破替身谣言！  321193
# 4              萨达姆被捕后告诫美国的一句话，发人深思         中国川贝枇杷膏在美国受到热捧？纯属谣言！  321191

"""
制作一个可以用来读取训练 / 测试集的Dataset， 这是你需要彻底了解的部分
此 Dataset 每次将 tsv 里的一个成对句子转换成 BERT 相容的格式，并返回 3 个 tensors：
- tokens_tensor：两个句子合并后的索引序列，包含 [CLS] 和 [SEP]
- segments_tensor：可以用来区别两个句子的 binary tensor
- label_tensor：将分类标签转换成类别索引的 tensor，如果是测试集测 返回 None
"""
from torch.utils.data import Dataset
import pysnooper
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

# 初始化一个专门读取训练样本的 Dataset，使用中文 BERT 断词
trainset = FakeNewsDataset("train", tokenizer=tokenizer)
# # 选择第一个样本
# sample_idx = 0
# # 将原始文本拿出来作比较
# text_a, text_b, label = trainset.df.iloc[sample_idx].values
# # 利用刚刚建立的 Dataset 取出转换后的 id tensors
# tokens_tensor, segments_tensor, label_tensor = trainset[sample_idx]
# # 将 tokens_tensor 还原成文本
# tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
# combined_text = "".join(tokens)
# 可以直接看输出结果
# print(f"""[原始文本]
# 句子 1：{text_a}
# 句子 2：{text_b}
# 分類  ：{label}
# --------------------
# [Dataset 返回的 tensors]
# tokens_tensor  ：{tokens_tensor}
# segments_tensor：{segments_tensor}
# label_tensor   ：{label_tensor}
# --------------------
# [还原 tokens_tensors]
# {combined_text}
# """)

"""
制作可以一次返回一个 mini-batch 的 DataLoader
这个 DataLoader 吃我们上面定义的 `FakeNewsDataset`，
返回训练 BERT 时会需要的 4 个 tensors：
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 这个函数的输入 `samples` 是一個 list，里面的每个 element 都是
# 刚刚定义的 `FakeNewsDataset` 返回的一个样本，每个样本都包含 3 tensors：
# - tokens_tensor
# - segments_tensor
# - label_tensor
# 它会对前两个 tensors 作 zero padding，并产生前面说明过的 masks_tensors
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
# 初始化一个每次返回 64 个训练样本的 DataLoader
# 利用 `collate_fn` 将 list of samples 合并成一个 mini-batch 是关键
BATCH_SIZE = 64
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                         collate_fn=create_mini_batch)

data = next(iter(trainloader))

tokens_tensors, segments_tensors, \
masks_tensors, label_ids = data
#
# print(f"""
# tokens_tensors.shape   = {tokens_tensors.shape}
# {tokens_tensors}
# ------------------------
# segments_tensors.shape = {segments_tensors.shape}
# {segments_tensors}
# ------------------------
# masks_tensors.shape    = {masks_tensors.shape}
# {masks_tensors}
# ------------------------
# label_ids.shape        = {label_ids.shape}
# {label_ids}
# """)
# 载入一个可以做中文多分类任务的模型，n_class = 3

from transformers import BertForSequenceClassification
NUM_LABELS = 3
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
# print(model.config)
# high-level 表示此模型里的 modules
# print("""
# name            module
# ----------------------""")
# for name, module in model.named_children():
#     if name == "bert":
#         for n, _ in module.named_children():
#             print(f"{name}:{n}")
#     else:
#         print("{:15} {}".format(name, module))


# class BertForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config, num_labels=2, ...):
#         super(BertForSequenceClassification, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config, ...)  # 載入預訓練 BERT
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         # 簡單 linear 層
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         ...
#
# def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, ...):
#     # BERT 輸入就是 tokens, segments, masks
#     outputs = self.bert(input_ids, token_type_ids, attention_mask, ...)
#     ...
#     pooled_output = self.dropout(pooled_output)
#     # 線性分類器將 dropout 後的 BERT repr. 轉成類別 logits
#     logits = self.classifier(pooled_output)
#
#     # 輸入有 labels 的話直接計算 Cross Entropy 回傳，方便！
#     if labels is not None:
#         loss_fct = CrossEntropyLoss()
#         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#         return loss
#     # 有要求回傳注意矩陣的話回傳
#     elif self.output_attentions:
#         return all_attentions, logits
#     # 回傳各類別的 logits
#     return logits
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

# 该模型跑在 GPU 上并取得训练集的分类准确率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)
# _, acc = get_predictions(model, trainloader, compute_acc=True)
# print("classification acc:", acc)

# def get_learnable_params(module):
#     return [p for p in module.parameters() if p.requires_grad]
#
# model_params = get_learnable_params(model)
# clf_params = get_learnable_params(model.classifier)
#
# print(f"""
# 整个分类模型的参数量：{sum(p.numel() for p in model_params)}
# 线性分类器的参数量：{sum(p.numel() for p in clf_params)}
# """)

# 训练模式
model.train()

# 使用 Adam Optim 更新整个分类模型的參數
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


EPOCHS = 8
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

# 建立測試集。這邊我們可以用跟訓練時不同的 batch_size，看你 GPU 多大
testset = FakeNewsDataset("test", tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=256,
                        collate_fn=create_mini_batch)

# 用分类模型预测测试集
predictions = get_predictions(model, testloader)

# 用来将预测的 label id 转换为 label 文字
index_map = {v: k for k, v in testset.label_map.items()}

# 生成 Kaggle 提交文件
df = pd.DataFrame({"Category": predictions.tolist()})
df['Category'] = df.Category.apply(lambda x: index_map[x])
df_pred = pd.concat([testset.df.loc[:, ["Id"]],
                     df.loc[:, 'Category']], axis=1)
df_pred.to_csv('./logs/bert_1_prec_training_samples.csv', index=False)
# 模型保存
PATH = "./logs/my_net.pth"
torch.save(model.state_dict(), PATH)
