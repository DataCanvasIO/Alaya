<div align="center">

  <img src="https://github.com/DataCanvasIO/Alaya/blob/main/pics/Alaya.png" width="100px">
  <h1>九章元识 | DataCanvas Alaya</h1>
  <a href="https://github.com/DataCanvasIO/Alaya/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-63b7a1"></a>
  <a href="https://github.com/DataCanvasIO/Alaya/blob/main/README_EN.md"><img src="https://img.shields.io/badge/Introduction-English-f6c844"></a>
  <a href="https://github.com/DataCanvasIO/Alaya/blob/main/README.md"><img src="https://img.shields.io/badge/%E7%AE%80%E4%BB%8B-%E4%B8%AD%E6%96%87-dc723c"></a>
  </br>
  <a href="https://huggingface.co/DataCanvas/Alaya-7B-Base"><img src="https://img.shields.io/badge/Base%20Model-%F0%9F%A4%97_Hugging_Face-a6e3de"></a>
  <a href="https://huggingface.co/DataCanvas/Alaya-7B-Chat"><img src="https://img.shields.io/badge/Chat%20Model-%F0%9F%A4%97_Hugging_Face-pink"></a>
</div>

&nbsp;

九章云极DataCanvas重磅发布的元识大模型Alaya，在自主整理的高品质多语言数据集上训练了1.5T+ tokens。  

首先在Hugging Face开源了<a href="https://huggingface.co/DataCanvas/Alaya-7B-Base">7B-Base</a>和<a href="https://huggingface.co/DataCanvas/Alaya-7B-Chat">7B-Chat</a>版本，模型表现业内领先，知识丰富且富有时效性，最新数据覆盖2023年10月的内容。Alaya-7B-Chat具备多轮对话、自我认知和偏见拒答的能力，能够完成知识问答、代码编写、信息提取、阅读理解、创意写作等多项语言任务。

# 🔗 目录
- [预训练](#-预训练)
  - 训练数据
  - 训练参数
  - Loss曲线
- [微调](#-微调)
  - 训练数据
- [使用Alaya](#-使用Alaya)
  - 依赖包安装
  - Chat Model推理方法
- [新闻](#-新闻)
- [声明与协议](#-声明)

&nbsp;

# 🧱 预训练
### 训练数据
Alaya使用了自研的大规模、多语言语料库，并采用文本去重和过滤这两种方法来控制数据品质。在文本去重上，我们采用了Fuzzy match 的策略，使用MinHash + LSH  筛选相似段落和文档，并结合编辑距离实现细致的文本去重。在针对低质量的文本数据过滤上，首先采用启发式的方法过滤掉部分不符合要求文本，然后训练了二分类器用于预测给定的网页文本是否适合，选择性丢弃一些数据。  

Alaya预训练数据中，英文语料占比约60%，中文语料占比约30%，代码语料占比约10%。为了更好地控制不同类型的语料参与训练的比例，对所有语料都做了分类，各个类型的语料token占比细节如下图所示：
<div align="center">
  <img src="https://github.com/DataCanvasIO/Alaya/blob/main/pics/Alaya%E5%90%84%E7%B1%BB%E5%88%AB%E8%AF%AD%E6%96%99%E5%88%86%E5%B8%83_ZH.png" width="400px">
  <img src="https://github.com/DataCanvasIO/Alaya/blob/main/pics/Alaya%E5%90%84%E8%AF%AD%E8%A8%80%E8%AF%AD%E6%96%99%E5%88%86%E5%B8%83_ZH.png" width="400px">
</div>

### 训练参数
训练Alaya的过程中，使用的超参如下：
| **Hidden Dimension**          | 4096                                                                  |
|:------------------------------|:----------------------------------------------------------------------|
| **Number of Attention Heads** | 32                                                                    |
| **Number of Layers**          | 32                                                                    |
| **Vocabulary Size**           | 60160                                                                 |
| **Optimizer**                 | Decoupled AdamW ($\beta_1$=0.9, $\beta_2$ =0.95, $\epsilon$ = 1.0e-8) |
| **Max Learning Rate**         | 1.2e-4                                                                |
| **Min Learning Rate**         | 1.2e-5                                                                |
| **Scheduler**                 | Cosine Decay with Warmup                                              |
| **Weight Decay**              | 1.0e-5                                                                |
| **Gradient Clip Norm**        | 0.3                                                                   |


### Loss
<div align="center">
  <img src="https://github.com/DataCanvasIO/Alaya/blob/main/pics/alaya_loss.png" width="600px">
</div>

&nbsp;
# 🔗 微调
### 训练数据
Alaya-Chat基于Alaya-7B进行有监督微调（SFT），微调数据量达500k+条，包含多领域的指令和对话数据。经过模型初筛和人工精筛，大幅提高微调数据品质，并且基于偏见语料对模型做了Red Teaming拒答微调。由于目前中文SFT语料多为机器翻译/大模型翻译而成，人工精筛可以进一步将其中不符合中文语法或使用习惯的劣质数据剔除。具体的微调数据分布如下图：

<div align="center">
  <img src="https://github.com/DataCanvasIO/Alaya/blob/main/pics/Alaya%E5%BE%AE%E8%B0%83%E6%95%B0%E6%8D%AE%E5%88%86%E5%B8%83_ZH.png" width="400px">
</div>

+ HHH(Helpful, Honest, Harmless)：问答模型最基础的属性就是为用户提供有帮助的、诚实的、无害健康的回答，我们精选了数万条3H对话数据。
+ 自我认知：模型对于自己的认知需要微调时提供相关信息，我们结合了人工编写和self-instruct两种方式，生成了3k+条多样的自我认知数据，从各个角度帮助模型学习Alaya的基本信息（i.e., 她的中文名、英文名...）。  
+ 偏见拒答：一定比例的Red Team数据可以减少模型的毒性输出，对于用户的错误引导做出拒答。我们使用了5k+偏见Red Team数据，模型拒绝回答不合理问题的能力显著增强。但需要注意的是，这并不能彻底杜绝毒性输出，对于用户强硬的洗脑，模型还是可能被误导。
+ 通用知识：我们使用了针对知识内容问答的数据集，增强模型作为知识助手的能力，让模型能够给出有效知识含量更高的回答。
+ 逻辑推断：CoT可以帮助模型提升推理能力，我们使用了中英双语的CoT数据数万条，同时也整理了代码、数学等领域的逻辑推理对话数据集。
+ 角色扮演：日常使用场景中，角色扮演可以对模型回答的风格、领域等细节进行限制，一定程度上增强问答模型的垂直领域灵活性。

科学设计的微调数据能够显著提升模型的问答能力，更加了解用户想要怎样的回答，提供更有效的帮助，并且不会对其在预训练阶段学习到的知识造成明显的负面影响。  


&nbsp;
# 💬 使用Alaya

### 依赖包安装
```shell
git clone https://github.com/DataCanvasIO/Alaya.git
pip install -r requirments.txt
```


### 推理方法
```
python -u inferrence.py <model_path> <input_file> <output_file>
```
+ ```model_path``` ：模型文件路径
+ ```input_file``` ：推断的输入```.txt```文件路径，每行为一个prompt
+ ```output_file``` ：输出的```.json```文件路径

&nbsp;
# 📰 新闻 
+ 2023年11月21日，九章云极举办开源Alaya-7B系列大模型发布会。  


&nbsp;
# 🛎 声明
Alaya训练过程中已经采取多种措施进行数据的筛选与过滤，尽可能保证数据的合法合规，但由于神经网络的黑盒性质，即使训练数据相对干净，模型还是可能生成一些错误的、不可预见的或难以干预的回答。请谨慎使用！  

请注意：
+ 请勿使用Alaya进行任何违反法律法规或是危害国家安全的活动  
+ 请勿恶意引导Alaya生成不合适的回答  
+ 请勿使用Alaya侵犯他人或团体的权益  
+ Alaya生成的文本不代表训练数据一定包含该信息，且不代表九章云极的立场

对于使用模型而导致的任何问题，九章云极将不承担任何责任。

### 联系我们  
如果您在使用的过程中发现任何问题，想要提供意见或建议，欢迎联系：sophia@zetyun.com。

&nbsp;
# 📃 协议
Alaya使用<a href="https://github.com/DataCanvasIO/Alaya/blob/main/LICENSE">Apache 2.0 Lisense</a>，开放模型权重，允许商业用途。如果您的项目引用了我们的Alaya，请标明出处，可以使用以下citation：
```
@misc{datacanvas2023alaya,
    author = {DataCanvas Ltd.},
    title = {alaya},
    year = {2023},
    howpublished = {\url{https://github.com/DataCanvasIO/Alaya}},
}
```

