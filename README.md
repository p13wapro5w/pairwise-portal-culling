# pairwise-portal-culling

## 领域

游戏、图形与多媒体

## 主题

门户剔除

## 说明

门户视锥、递归深度与动态重建。

初始状态下 `render/portal_culling.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
