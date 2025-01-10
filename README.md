
group代表分组   api代表apinet结构 lsk表示使用sa聚合  encoder代表编码器 c3s-loss代表c3sloss损失函数  bap代表BAP操作

| 训练文件 | 描述 | 准确率 |
| --- | --- | --- |
| train_group_api_lsk_encoder | 使用了encoder编码器 4个组 |  |
| train_group_c3s-loss-change_api_Res | 分两组仿apinet |  |
| train_group_c3s_api_lsk_encoder |  16个组 |  |
| train_group_c3s-loss-bap13 | 分了13组做bap |  |
| train_group_api_lsk_bap | 使用了bap进行预测 |  |

一般情况使用SGD就可以
SGD优化器 lr=0.001 每个GPU batch-size=16  150epoch 每60降0.1
AdamW优化器 lr=0.0001 每个GPU batch-size=16  150epoch 每60降0.1

数据增强代码在dataset.py中

```python
# 运行命令参考
python train_group_api_lsk_encoder.py
```
