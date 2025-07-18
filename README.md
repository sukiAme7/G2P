# G2P
**Grapheme-to-Phoneme（G2P）转换**是**自然语言处理**和语音处理中的基础任务，其目标是：

> **将一个词的拼写（字母/字素）序列转换为其发音（音素）序列。**
## Dataset
数据集我已经完全划分好，放在data路径下，如果你想要重新划分，使用/utils/preprocess.py脚本即可
## Model
模型使用的Transformer,有两个版本，一个是model.py,另一个是transformer.py<br>
两个版本实现差异不大，我训练时具体使用的是model.py,但是另一个transformer.py中注释比较详细，可以对比阅读一下
## Training
训练可以直接运行trainer.py,但建议提前在config.py文件中配置一下GPU还是cuda,以及训练batch size大小以防训练OOM
```bash
python trainer.py
```
## Inference
推理需要先把训练得到的checkpoint放在model_save路径下，如果训练比较耗时可以使用我训练的模型文件<br>
下载路径：https://pan.baidu.com/s/1U7NMWEWC9T4vv7nLtOWNBw  提取码：wfsa <br>
然后直接运行inference.py
```bash
python inference.py
```
