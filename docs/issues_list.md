## 已经问题清单


1. 训练阶段不支持多卡做验证，[关联代码](../whispering/bin/train.py#L176)
2. 多语言多任务验证阶段只能以特定任务和语言选择best_chckpoint，[关联代码](../whispering/utils/executor.py#L193)
3. shard格式中的音频如果非wav格式，需要特殊处理，[关联代码](../whispering/dataset/processor.py#L97)
3. `data.list`中条数少于`num_works*rank`会有进程中断，[关联代码](../whispering/dataset/dataset.py#L72)
