## Aishell微调whisper-base


### First Experiment

我们提供了一个参考运行示例 `example/aishell/s0/run.sh` 在 aishell-1 数据集上，建议你一步步运行run.sh脚本，方便调试

```
cd example/aishell/s0
# 创建data/dev data/test data/train文件夹
bash run.sh --stage -1 --stop_stage -1
# 创建数据集
bash run.sh --stage 0 --stop_stage 0
# 训练
bash run.sh --stage 1 --stop_stage 1
# 测试
bash run.sh --stage 2 --stop_stage 2
```

未完待续。。。