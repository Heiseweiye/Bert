## Bert：使用pytorch+Bert实现虚假新闻对分类

---

任何新文章在发表前都必须经过内容真实性的测试。我们将新文章与数据库中的文章进行匹配。被认定为含有虚假新闻的文章，经人工审核后将被撤回。因此，过程的准确性和效率对于我们使平台安全、可靠和健康变得至关重要。

### 关于任务

任务：新闻文章分类

给定假新闻文章 A 的标题和即将发布的新闻文章 B 的标题，参与者被要求将 B 归入三个类别之一。

- 同意：B 和 A 谈论相同的假新闻
- 不同意：B 驳斥 A 中的假新闻
- 无关：B 与 A 无关

## 目录

1. [训练环境](#训练环境)
2. [训练数据集](#训练数据集)
3. [模型训练](#模型训练)
4. [模型保存与下载](#模型保存与下载)
5. [模型加载与预测](#模型加载与预测)
6. [参考](#参考)

## 训练环境

### 依赖包

python3.8

torch1.8.1+cu11.1（相关环境可在https://github.com/Heiseweiye/YOLOV3获得）

transformers

```shell
conda create -n 环境名 python==3.8
conda activate 环境名
# 下载好torch1.8.1+cu11.1然后切换到torch1.8.1+cu111的目录下
pip install 文件名
conda install transformers
```

### GPU

RTX3090

> tips：
>
> 本文使用RTX3090进行训练
>
> batchsize为256
>
> epoch为10
>
> 训练全部训练集
>
> 占用显存约20G
>
> 大概训练时间为2~3小时

## 数据集下载

训练集（26万个数据）：

链接：https://pan.baidu.com/s/1EfDgW-XYqPedTul9oS85Mg 
提取码：unyl

测试集（大概8万，忘记了）：

链接：https://pan.baidu.com/s/14fqkSNM9jc8LAcfhWDXduQ 
提取码：t5a6

Bert预训练模型无法自动下载的，可以手动下载，下载之后在Bert目录下解压即可：

链接：https://pan.baidu.com/s/127j2tToqaVQjbkyY5MR0uw 
提取码：9czo

数据集下载好统一放入datasets文件夹内

## 模型训练

1. 将本项目下载到本地，下载好相应的数据集

2. Bert的main函数可以调整3个参数：选取的训练集所占整个训练集的比例，batchsize，epoch

3. 根据自己的机器选择合适的参数值

4. 在项目文件下打开命令行，激活之前安装好的环境

5. 输入命令（0指定第一块GPU，如果有多块显卡，依次类推）

   ```shell
   CUDA_VISIBLE_DEVICES=0 python bert.py
   ```

6. 训练结束的模型参数保存在logs文件夹下

## 模型保存与下载

在进行预测之前，如果由于机器原因，无法训练自己的模型，也可以下载我训练好的模型进行预测：

Bert训练之后的模型，我已经训练好（约390MB）：

链接：https://pan.baidu.com/s/1zWao8lJadUnsVGe8oem8Cw 
提取码：qg48

把下载好的模型直接放在logs文件夹下面

## 模型加载与预测

在你已经训练好模型之后（或你已经下载好我的模型），直接运行model_load.py文件即可，最终的预测文件会保存为model_load_test_samples.csv

执行命令。此预测过程预计2~3分钟左右，耐心等待。

```shell
CUDA_VISIBLE_DEVICES=0 python model_load.py
```



## 参考

BERT视频：https://youtu.be/UYPa347-DdE

BERT实操：https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html



