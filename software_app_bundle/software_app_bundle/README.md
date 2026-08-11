# 医学图像配准软件交付文件

这个文件夹包含的是**正式软件运行所需要的最小代码集合**。

它的目标不是保留全部实验过程，而是保留：

1. 原始图像输入  
2. `modified` 方法配准  
3. 配准后 3D 输出  
4. 配准后 `self` 切片输出  
5. 医生可直接使用的图形界面

---

## 1. 这个文件夹里包含什么

### Python 入口

- `doctor_registration_gui.py`
  医生版图形界面入口，推荐正式使用时直接运行它

- `one_click_registration.py`
  一步式后台主入口，负责串联完整流程

### Python 功能模块

- `step1.py`
  原始医学图像 -> 点云预处理

- `visualize_registration_result.py`
  配准结果 -> 3D 与切片输出

- `batch_prepare_experiments.py`
  这里主要保留了其中被一步式脚本复用的文件整理功能

### MATLAB 配准核心

- `main.m`
  正式软件使用的唯一 MATLAB 配准主函数，也就是 `modified`

- `KCenter.m`
  `main.m` 使用的点集压缩模块

- `SinkhornInit.m`
  初始刚体匹配核心

- `Sinkhorn.m`
  Sinkhorn 距离与流矩阵计算

- `distance.m`
  点集距离矩阵计算

- `Transport.m`
  Sinkhorn/OT 求解底层函数

---

## 2. 为什么只保留这些文件

因为正式软件目前只使用：

- 单病例输入
- `modified` 方法
- 自动预处理
- 自动生成配准后结果
- 自动生成 `self` 版本切片

所以以下类型的文件**没有放进来**：

- 批量实验脚本
- 对比算法脚本
- 误差分析脚本
- 实验性 MATLAB 版本
- 论文复现实验代码
- 数据集整理与评估用的附加脚本

这样做的好处是：

- 软件结构更干净
- 更适合交付给别人使用
- 不容易误用实验脚本
- 后续做图形界面封装更方便

---

## 3. 推荐启动方式

### 医生或普通用户

直接运行：

```bash
python doctor_registration_gui.py
```

然后在界面中：

1. 选择 `Moving` 图像
2. 选择 `Fixed` 图像
3. 选择模态
4. 选择输出文件夹
5. 点击“开始配准”

### 技术人员

如果需要命令行方式，可运行：

```bash
python one_click_registration.py \
  --moving-image /path/to/moving.nii.gz \
  --fixed-image /path/to/fixed.nii.gz \
  --moving-modality CT \
  --fixed-modality MRI \
  --output-dir /path/to/output_case \
  --overwrite
```

---

## 4. 输出结果

正式软件会输出：

- 配准后的 3D moving 图像
- 对应 fixed 参考 3D 图像
- 配准后 moving 的 `self` 切片图
- fixed 的 `self` 切片图
- 结果摘要 `summary.json`

---

## 5. 依赖

### Python

至少需要：

- `numpy`
- `scipy`
- `SimpleITK`
- `matplotlib`

### MATLAB

需要安装 MATLAB，并且命令行可调用：

```bash
matlab
```

如果 MATLAB 不在默认路径，可以在命令行入口中通过 `--matlab-bin` 指定。

---

## 6. 当前软件范围

这套交付文件当前适合做：

- 单病例医学图像刚体配准
- 多模态图像配准结果查看
- 临床演示
- 后续 GUI 或桌面程序封装

如果以后你要继续做：

- 批量实验
- 多算法比较
- 评估分析
- 论文复现

建议继续使用项目根目录中的完整代码，而不是只使用这个交付文件夹。
