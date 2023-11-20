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

DataCanvas has officially released the groundbreaking meta-awareness model, Alaya, which has been trained on a curated high-quality multi-lingual dataset, accumulating 1.5T+ tokens.   

Initially, the <a href="https://huggingface.co/DataCanvas/Alaya-7B-Base">7B-Base</a> and <a href="https://huggingface.co/DataCanvas/Alaya-7B-Chat">7B-Chat</a> versions were open-sourced on Hugging Face. The model demonstrates industry-leading performance, boasting rich and timely knowledge, with the latest data encompassing content up to October 2023. Alaya-7B-Chat possesses capabilities for multi-turn conversations, self-awareness, and bias refusal, enabling it to accomplish various language tasks such as knowledge-based question answering, code generation, information extraction, reading comprehension, and creative writing.

# Contents
+ [Pre-training](#-pre-training)
  + Training Data
  + Training Hyperparameters
  + Loss Curve 
+ [Fine-Tuning](#-fine-tuning)
  + Training Data
+ [Using Alaya](#-using-alaya)
  + Dependency Installation
  + Chat Model Inferrence
+ [News](#-news)
+ [Declaration&License](#-declaration)

&nbsp;
# 🧱 Pre-Training
### Training Data
Alaya utilizes a proprietary large-scale, multilingual corpus and employs two methods, namely text deduplication and data quality control, to manage data quality. For text deduplication, a strategy employing Fuzzy Match is adopted. This involves using MinHash + LSH to filter similar paragraphs and documents, coupled with edit distance for meticulous text deduplication. In addressing low-quality text data, a heuristic method is initially applied to filter out text that does not meet the required criteria. Subsequently, a binary classifier is trained to predict whether the given web page text is suitable, allowing for selective discarding of some data.

In the pre-training data for Alaya, English-language content constitutes approximately 60%, Chinese-language content makes up around 30%, and code-related content accounts for approximately 10%. To better control the proportions of different types of data participating in training, all data is categorized, and the token distribution details for each type are illustrated in the following figure:
<div align="center">
  <img src="https://github.com/DataCanvasIO/Alaya/blob/main/pics/Alaya%E5%90%84%E8%AF%AD%E8%A8%80%E8%AF%AD%E6%96%99%E5%88%86%E5%B8%83_EN.png" width="400px">
  <img src="https://github.com/DataCanvasIO/Alaya/blob/main/pics/Alaya%E5%90%84%E7%B1%BB%E5%88%AB%E8%AF%AD%E6%96%99%E5%88%86%E5%B8%83_EN.png" width="400px">
</div>


### Training Hyperparameters
The hyperparameters using when trainig Alaya are as follows:
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


### Loss Curve
<div align="center">
  <img src="https://github.com/DataCanvasIO/Alaya/blob/main/pics/alaya_loss.png" width="600px">
</div>

&nbsp;
# 🔗 Fine-Tuning
Alaya-Chat undergoes supervised fine-tuning (SFT) based on Alaya-7B, with a dataset comprising over 500,000 instances encompassing diverse instructions and dialogues across multiple domains. After initial model screening and meticulous human review, the quality of the fine-tuning data is significantly improved. Red-team fine-tuning is applied to the model using biased language data, enhancing its ability to refuse biased queries. Given that a considerable portion of Chinese SFT data is derived from machine translation or large model translation, human review further eliminates low-quality data that deviates from Chinese grammar or usage habits. The specific distribution of fine-tuning data is illustrated in the following figure:

<div align="center">
  <img src="https://github.com/DataCanvasIO/Alaya/blob/main/pics/Alaya%E5%BE%AE%E8%B0%83%E6%95%B0%E6%8D%AE%E5%88%86%E5%B8%83_EN.png" width="400px">
</div>

+ HHH (Helpful, Honest, Harmless): This category focuses on fundamental attributes for a question-answering model—providing helpful, honest, and harmless responses to users. Tens of thousands of 3H dialogue instances have been meticulously selected.
+ Self-awareness: To refine the model's self-awareness, providing relevant information when needed, a diverse set of over 3,000 instances has been generated using a combination of human-authored and self-instructed methods. This aids the model in learning basic information about Alaya, such as her Chinese name, English name, etc.
+ Bias Refusal: To reduce toxic outputs, Red-team data is used to train the model to refuse to respond to biased queries. With over 5,000 instances of biased Red Team data, the model's ability to reject unreasonable questions is significantly strengthened. It's important to note that this does not eliminate toxic outputs, and the model may still be misled by persistent attempts to manipulate it.
+ General Knowledge: A dataset focused on knowledge-based questions is utilized to enhance the model's capabilities as a knowledge assistant, enabling it to provide answers with higher knowledge content.
+ Logical Deduction: Using tens of thousands of bilingual CoT instances, the model's logical reasoning abilities are improved. Additionally, dialogue datasets for logical reasoning in fields such as code and mathematics are curated.
+ Role-playing: Simulating everyday scenarios, role-playing introduces constraints on the model's response style and domain details, enhancing the chat model's versatility on vertical industrial fields.

Fine-tuning with scientifically designed data significantly boosts the model's question-answering capabilities, improving its understanding of user expectations, providing more effective assistance, and minimizing negative impacts on the knowledge acquired during the pre-training phase.

&nbsp;
# 💬 Using Alaya
### Dependency Installation
```shell
git clone https://github.com/DataCanvasIO/Alaya.git
pip install -r requirments.txt
```

### Chat Model Inferrence
```
python -u inferrence.py <model_path> <input_file> <output_file>
```
+ ```model_path``` ：Path of model checkpoint path.
+ ```input_file``` ：Path of the input ```.txt``` file for inferrence, consisting of prompts line by line.
+ ```output_file``` ：Path of the output ```.json``` file.

&nbsp;
# 📰 News
+ On November 21, 2023, DataCanvas hosted the open-source release event for the Alaya-7B series.

&nbsp;
# 🛎 Declaration
During the training process of Alaya, various measures have been implemented to filter and screen data, aiming to ensure the legality and compliance of the data. However, due to the black-box nature of neural networks, even with relatively clean training data, the model may still generate erroneous, unpredictable, or difficult-to-intervene answers. Please use with caution!

Please note:
+ DO NOT use Alaya for any activities that violate laws and regulations or pose a threat to national security.
+ DO NOT maliciously guide Alaya to generate inappropriate responses.
+ DO NOT use Alaya to infringe upon the rights of individuals or groups.
+ The text generated by Alaya DOES NOT necessarily reflect information present in the training data and DOES NOT represent the stance of DataCanvas.

DataCanvas will not assume any responsibility for any issues arising from the use of the model.

### Contact Us

If you encounter any issues during use or wish to provide feedback or suggestions, please feel free to contact: sophia@zetyun.com .

&nbsp;
# 📃 License

Alaya is licensed under the <a href="https://github.com/DataCanvasIO/Alaya/blob/main/LICENSE">Apache 2.0 License</a>, and the model weights are open for commercial use. 

If your project references our Alaya model, please acknowledge the source and consider using the following citation:

```
@misc{datacanvas2023alaya,
    author = {DataCanvas Ltd.},
    title = {alaya},
    year = {2023},
    howpublished = {\url{https://github.com/DataCanvasIO/Alaya}},
}
```

