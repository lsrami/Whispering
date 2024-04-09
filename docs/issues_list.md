## 已经问题清单


1. torchaudio>2.0需要关闭全局backend设置，在函数调用时指定backend为ffmpeg，[关联代码](../whispering/dataset/processor.py#L27)
2. 多语言多任务验证阶段只能以特定任务和语言选择best_checkpoint，[关联代码](../whispering/utils/executor.py#L188)
3. shard格式中的音频如果非wav格式，需要特殊处理，[关联代码](../whispering/dataset/processor.py#L97)
4. `data.list`中条数少于`num_works*rank`会有进程中断，[关联代码](../whispering/dataset/dataset.py#L72)
5. 使用`--cv_partition`指定多卡验证可能会有进程中断，[关联代码](../whispering/bin/train.py#L88)
