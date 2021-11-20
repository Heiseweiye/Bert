#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: heiseweiye
# time: 2021/11/20 17:08
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: heiseweiye
# time: 2021/11/20 12:42
"""
這段程式碼載入已經訓練好的 masked 語言模型並對有 [MASK] 的句子做預測
"""
import torch
import random
import os
import pandas as pd
from torch.utils.data import Dataset
import pysnooper
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 确定要用的Bert基础模型
PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# 读取测试集数据并保存到test.tsv
df_test = pd.read_csv("./datasets/test.csv")
df_test = df_test.loc[:, ["title1_zh", "title2_zh", "id"]]
df_test.columns = ["text_a", "text_b", "Id"]
df_test.to_csv("test.tsv", sep="\t", index=False)
print("预测样本数：", len(df_test))

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

"""
制作可以一次返回一个 mini-batch 的 DataLoader
这个 DataLoader 吃我们上面定义的 `FakeNewsDataset`，
返回训练 BERT 时会需要的 4 个 tensors：
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""
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

# 利用本地模型进行预测
# 该模型跑在 GPU 上并取得训练集的分类准确率
PATH = 'logs/my_net.pth'
PRETRAINED_MODEL_NAME = 'bert-base-chinese'
NUM_LABELS = 3

# 加载模型参数
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
model.load_state_dict(torch.load(PATH))
print("模型加载成功！")

# 设置GPU还是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)

# 开启预测模式
model.eval()

# 建立测试集。这边我门可以用跟训练时不同的 batch_size，看你 GPU 多大
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
df_pred.to_csv('./logs/model_load_test_samples.csv', index=False)