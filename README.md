# pairwise-saturator-oversample

## 领域

游戏、图形与多媒体

## 主题

饱和失真与过采样

## 说明

按采样率定倍、长期峰值与旁路淡化。

初始状态下 `media/saturator_oversample.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
