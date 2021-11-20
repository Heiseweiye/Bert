import numpy as np
import torch 
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel

model_name = 'bert-base-chinese'
MODEL_PATH = './bert-base-chinese/'

# a. 通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
# b. 导入配置文件
model_config = BertConfig.from_pretrained(MODEL_PATH)
# 通过配置和路径导入模型
bert_model = BertModel.from_pretrained(MODEL_PATH)
a = "l Love You"
b = tokenizer.tokenize(a)
print(b)
print("测试成功")