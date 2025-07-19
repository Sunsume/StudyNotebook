# 张量Tensor

## 张量的创建

| 函数名                                                       | 函数作用                                          |
| ------------------------------------------------------------ | ------------------------------------------------- |
| torch.tensor(a)                                              | 把类型a创建成Tensor类型                           |
| torch.Tensor(ints)                                           | 创建ints指定形状的张量，默认dtype=float32         |
| torch.IntTensor(...)<br>torch.ShortTensor(...)<br/>torch.LongTensor(...)<br/>torch.FloatTensor(...)<br/>torch.DoubleTensor(...)<br/> | 使用具体类型的张量                                |
| torch.arange(start,end,step)                                 | 若有start,end则为遵循左闭右开原则，若单参数为长度 |
| torch.linspace((start,end,count)                             | 在指定区间按照元素个数生成,可取end                |
| torch.randn(row,col)                                         | 创建随机张量                                      |
| torch.random.manual_seed(n)                                  | 创建随机张量设置随机数种子为n                     |
| torch.random.initial_seed()                                  | 查看随机数种子值                                  |
| torch.ones(row,col)<br>torch.ones_like(Tensor)               | 创建全1张量                                       |
| torch.zeros(row,col)<br/>torch.zeros_like(Tensor)            | 创建全0张量                                       |
| torch.full([row,col],num)<br/>torch.zeros_like(tensor,num)   | 创建数值为num的张量                               |
| t1.type(torch.DoubleTensor)<br>t1.double()                   | 类型转换，有多种                                  |



## 张量数值计算

| 函数名                                                       | 函数作用                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| t1.add(nub)<br>t1.add_(nub)                                  | 张量加某常量，带_会改变原数据                                |
| t1.sub(nub)<br>t1.sub_(nub)                                  | 减法                                                         |
| t1.mul(nub)<br/>t1.mul_(nub)                                 | 乘法                                                         |
| t1.div(nub)<br/>t1.div_(nub)                                 | 除法                                                         |
| t1.neg(nub)<br/>t1.neg_(nub)                                 | 加负号                                                       |
| torch.mul(t1,t2)<br>t1*t2                                    | 阿达玛积:指的是矩阵对应位置的元素相乘                        |
| t1@t2<br>torch.mm(t1,t2)<br>torch.bmm(t1,t2)<br>torch.matmul(t1,t2) | 运算符 @ 用于进行两个矩阵的点乘运算<br> torch.mm 用于进行两个矩阵点乘运算, 要求输入的矩阵为2维<br/> torch.bmm 用于批量进行矩阵点乘运算, 要求输入的矩阵为3维 <br/>torch.matmul 对进行点乘运算的两矩阵形状没有限定. <br/> 对于输入都是二维的张量相当于 mm 运算. <br/>对于输入都是三维的张量相当于 bmm 运算 <br/>对数输入的 shape 不同的张量, 对应的最后几个维度必须符合矩阵运算规则 |

## 张量指定运算设备

PyTorch 默认会将张量创建在 CPU 控制的内存中, 即: 默认的运算设备为 CPU。我们也可以将张量创建在 GPU 上, 能够利用对于矩阵计算的优势加快模型训练。将张量移动到 GPU 上有两种方法: 1. 使用 cuda 方法 2. 直接在 GPU 上创建张量 3. 使用 to 方法指定设备

| 函数名             | 函数作用                   |
| ------------------ | -------------------------- |
| t1.device          | 查看张量t1的存储设备       |
| t1=t1.cuda()       | 将张量t1转到GPU上          |
| t1=t1.cpu()        | 将张量t1转到CPU上          |
| t1=t1.to('设备名') | 将张量t1转到指定存储设备上 |



## 张量类型转换

| 函数名                   | 函数作用                                                     |
| ------------------------ | ------------------------------------------------------------ |
| t2=t1.numpy()            | 将Tensor转ndarray，其中t1和t2共享内存                        |
| t2=torch.from_ numpy(t1) | 将ndarray转为Tensor                                          |
| t2=t1.item()             | 对于只有一个元素的张量，使用 item 方法将该值从张量中提取出来。 |
| t2.copy_(t1)             | 把t1copy到t2，直接修改目标张量，而不会创建新的张量。覆盖`tensor`的原始数据 |



## 张量拼接操作

| 函数名                       | 函数作用                                                     |
| ---------------------------- | ------------------------------------------------------------ |
| torch.cat((t1,t2,...),dim=0) | 把张量t1,t2,...拼接成一个张量，dim=0沿行方向拼接，dim=1沿列方向拼接，dim=2沿深度方向拼接 |
| torch.stack(tensors,dim=0)   | 沿**新维度**堆叠多个张量（创建新维度）                       |

### 1.PyTorch `torch.cat()` 方法详解：`dim` 参数的使用

`torch.cat()` 是 PyTorch 中最常用的张量拼接方法，而 `dim` 参数决定了拼接的方向。下面我将详细解析 `dim` 参数的使用方法，通过多维张量示例演示其工作机制。

#### 1.核心概念：`dim` 参数的含义

`dim` 参数指定了**沿着哪个维度进行拼接**：

- `dim=0`：沿行方向（垂直拼接）
- `dim=1`：沿列方向（水平拼接）
- `dim=2`：沿深度方向（通道方向）
- 以此类推，对于更高维张量

#### 2.基本规则

1. **所有非拼接维度必须完全相同**
2. **拼接维度的大小可以不同**
3. **输出张量的拼接维度尺寸 = 所有输入张量在该维度尺寸之和**

#### 3. dim参数使用总结表

| 数据类型      | `dim=0`            | `dim=1`                | `dim=2`          | `dim=3`  |
| ------------- | ------------------ | ---------------------- | ---------------- | -------- |
| 向量 (1D)     | 延长向量           | -                      | -                | -        |
| 矩阵 (2D)     | 垂直拼接（增加行） | 水平拼接（增加列）     | -                | -        |
| 图像特征 (3D) | 增加样本数         | 增加行数(高度)         | 增加列数（宽度） | -        |
| 图像批次 (4D) | 增加批次大小       | 增加通道数（色彩通道） | 增加高度         | 增加宽度 |

###  `2.torch.stack()` 及 `dim` 參數用法

#### 1. **torch.stack() 的核心功能**

- **將多個張量（tensors）沿著一個新的維度（dimension）堆疊**。
- 輸入的所有張量必須具有**完全相同的形狀（shape）**。
- 輸出張量會比輸入張量**多出一個維度**（新維度由 `dim` 參數指定）。

#### 2.语法

```python
torch.stack(tensors, dim=0, *, out=None)
```



#### 3.关键概念：`dim` 参数的作用

`dim` 決定新維度插入的位置：

- **dim=0**：新維度成為**最外層**（如將 `N` 個形狀為 `(A, B)` 的張量 → 堆疊為 `(N, A, B)`）。
- **dim=1**：新維度插入在**第 1 維**（原第 0 維之後）。
- **dim=-1(2也可以，-1和2的操作是一致的)**：新維度成為**最內層**（追加到最後）。

> **重要規則**：輸出張量的維度數 = 輸入張量維度數 + 1。
>
> eg:
>
> ```python
> import torch
> 
> t1=torch.tensor([[1,2],[3,4]])
> t2=torch.tensor([[5,6],[7,8]])
> #dim=0
> t3=torch.stack([t1,t2],dim=0)
> #dim=1
> t3=torch.stack([t1,t2],dim=1)
> #dim=-1（2也行）
> t3=torch.stack([t1,t2],dim=-1)
> ```
>
> ![20250718_112326.jpg](./张量img/20250718_112326.jpg)





## 张量索引操作

torch.tensor的索引操作与numpy的向量、矩阵的索引操作极为相似，几乎可以看做一样。熟悉numpy索引操作方式可以跳过本段

### 1.基本索引

##### 单元素 索引 arr[row,col] 

```python
import torch
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t1[0, 1])  # 索引0行1列的数值-》 输出 2
```

##### 切片索引 arr[start : end :step ,start : end : step ]

```python
print(t1[:, 1:3])  # 所有行的第 1-2 列 (左闭右开原则)
# 输出：
# [[2 3]
#  [5 6]]
```

### 2.高级索引

##### 列表索引

```python
print(arr[[0, 1], [0, 2]])  # 取 (0,0) 和 (1,2) 的元素
# 输出 [1, 6]
```

##### 布尔索引

```python
print(t1[arr > 3])  # 输出所有大于 3 的元素：[4, 5, 6]
```

##### 多维索引

```python
arr3d = torch.tensor([[[1, 2], [3, 4]], [[[5, 6], [7, 8]]])
print(arr3d[0, 1, 0])  # 输出 3 arr3d[z，x，y]
```

## 张量形状操作

| 函数名                   | 函数作用                                                     |
| ------------------------ | ------------------------------------------------------------ |
| t1.reshape(shape)        | **返回具有相同数据但新形状的新张量**.<br> **t1**: 要重塑的张量;**shape**:目标形状的元组（tuple) |
| t1.transpose(dim0, dim1) | **功能**：交换指定的两个维度 <br>**参数**：  `dim0`：第一个要交换的维度 `dim1`：第二个要交换的维度 |
| t1.permute(dims)         | **功能**：按指定顺序重新排列所有维度<br> **参数**：`*dims` - 新维度顺序的整数序列 |
| view                     | **改变张量的形状（shape）**，但**不改变数据内容**和**元素顺序** 要求张量的**元素总数不变**（即 `numel()` 不变） **必须内存连续**（或显式调用 `contiguous()`） |
| contigous                | **确保张量在内存中是连续的** 如果张量**原本不连续**，会**复制数据**并返回一个新的连续张量 如果张量**原本连续**，则**直接返回原张量** |
| squeeze                  | 函数用删除 shape 为 1 的维度                                 |
| unsqueeze                | 在每个维度添加 1, 以增加数据的形状                           |

### PyTorch 中的 `reshape()` 函数详解

#### 核心特性

1. **数据不变性**：只改变形状，不改变数据内容和顺序

2. **元素总数不变**：新形状的元素总数必须与原张量相同

3. **内存连续性**：尽可能返回视图（共享内存），必要时返回拷贝

4. **自动维度计算**：支持使用 `-1` 自动计算维度大小

5. **变化方式：**自左向右，自上向下，自前向后

   ![20250718_160452.jpg](./张量img/20250718_160452.jpg)

#### 参数详解

#### `shape` 参数规则

- 必须是**整数元组**（如 `(3, 4)`）
- 可以使用 `-1` 表示自动计算该维度大小
- 各维度大小的乘积必须等于原张量元素总数

#### 使用示例

##### 基础用法

```python
import torch

# 创建 1x12 张量
t = torch.arange(12)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# 重塑为 3x4 矩阵
reshaped = t.reshape(3, 4)
"""
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
"""
```

##### 自动维度计算（使用 -1）

```python
# 自动计算行数
auto_row = t.reshape(-1, 3)  # 4x3 矩阵

# 自动计算列数
auto_col = t.reshape(4, -1)  # 4x3 矩阵

# 自动计算中间维度
auto_mid = t.reshape(2, -1, 3)  # 2x2x3 张量
```

##### 增加/减少维度

```python
# 增加维度（从2D到3D）
matrix = torch.randn(4, 3)
tensor_3d = matrix.reshape(2, 2, 3)

# 减少维度（展平）
flattened = tensor_3d.reshape(-1)  # 1D向量
```

### PyTorch 中的 `transpose` 与 `permute` 函数详解

transpose 函数可以实现交换张量形状的指定维度, 例如: 一个张量的形状为 (2, 3, 4) 可以通过 transpose 函数把 3 和 4 进行交换, 将张量的形状变为 (2, 4, 3)



#### 1. `torch.transpose()` - 维度交换

##### 使用示例

```python
import torch

# 创建3维张量 (batch, channels, height)
x = torch.randn(2, 3, 4)  # 形状: [2, 3, 4]

# 交换维度0和1
y = x.transpose(0, 1)  # 形状变为 [3, 2, 4]

# 交换维度1和2
z = x.transpose(1, 2)  # 形状变为 [2, 4, 3]
```



##### 图像处理应用

```python
# 图像张量格式转换 (NCHW ↔ NHWC)
images = torch.randn(16, 3, 32, 32)  # NCHW格式

# 转换为NHWC格式
nhwc = images.transpose(1, 2).transpose(2, 3)  # 形状 [16, 32, 32, 3]

# 转换回NCHW格式
nchw = nhwc.transpose(2, 3).transpose(1, 2)  # 形状 [16, 3, 32, 32]
```

#### 2. `torch.permute()` - 维度重排

##### 使用示例

```python
# 创建4维张量 (batch, channels, height, width)
x = torch.randn(2, 3, 4, 5)  # 形状: [2, 3, 4, 5]

# 重排维度顺序
y = x.permute(0, 2, 3, 1)  # 形状变为 [2, 4, 5, 3]
z = x.permute(2, 0, 1, 3)  # 形状变为 [4, 2, 3, 5]

# 实现转置的等效操作
transpose_equivalent = x.permute(0, 1, 3, 2)  # 等价于 x.transpose(2, 3)
```

##### 序列数据处理

```python
# 时间序列数据 (batch, seq_len, features)
seq_data = torch.randn(10, 20, 50)  # [batch, seq, features]

# 转换为注意力机制需要的格式 (seq_len, batch, features)
attn_input = seq_data.permute(1, 0, 2)  # 形状 [20, 10, 50]
```

#####  关键区别对比

| 特性             | `transpose()`    | `permute()`          |
| ---------------- | ---------------- | -------------------- |
| **维度数量**     | 只能交换两个维度 | 可以重排任意多个维度 |
| **操作复杂度**   | 简单交换         | 完全重排             |
| **使用场景**     | 简单维度交换     | 复杂维度重排         |
| **参数数量**     | 固定两个参数     | 可变参数（维度数量） |
| **实现相同操作** | 需多次调用       | 单次调用完成         |
| **内存共享**     | 是（视图）       | 是（视图）           |

#### 3.**view() 函数**

##### **使用示例**

```python
import torch

# 原始张量 (6 个元素)
x = torch.arange(6)  # shape: [6]

# 改变形状为 2x3
y = x.view(2, 3)
"""
y = 
tensor([[0, 1, 2],
        [3, 4, 5]])
"""

# 改变形状为 3x2
z = x.view(3, 2)
"""
z = 
tensor([[0, 1],
        [2, 3],
        [4, 5]])
"""
```



##### **-1 自动计算维度**

如果某个维度不确定，可以用 `-1`，PyTorch 会自动计算：

```python
x = torch.arange(12)  # shape: [12]

# 自动计算行数，列数为 4
y = x.view(-1, 4)    # shape: [3, 4]
```

##### **view() 的限制**

- **必须内存连续**（否则会报错）
- **不能改变元素顺序**（`reshape()` 可以，但 `view()` 不行

#### 4.**contiguous() 函数**

##### **为什么需要 contiguous()？**

PyTorch 的某些操作（如 `transpose()`、`permute()`）会**改变张量的内存布局**，导致**非连续存储**：

```python
x = torch.arange(6).view(2, 3)  # shape: [2, 3]
y = x.transpose(0, 1)           # shape: [3, 2]

print(x.is_contiguous())  # True
print(y.is_contiguous())  # False（转置后不连续）
```

此时，如果直接 `y.view(6)` 会报错

##### **解决方案**

先 `contiguous()`，再 `view()`：

```python
y_contig = y.contiguous()  # 确保内存连续
y_reshaped = y_contig.view(6)  # 正确
```



#### 5.**view() vs reshape()**

| 函数           | 是否共享内存       | 是否要求连续 | 适用场景           |
| -------------- | ------------------ | ------------ | ------------------ |
| `view()`       | 是（视图）         | 必须连续     | **高效改变形状**   |
| `reshape()`    | 可能（视图或拷贝） | 不要求       | **更灵活**         |
| `contiguous()` | 可能拷贝           | 确保连续     | **修复非连续张量** |

1. **优先用 view()**（更高效，但要求连续）
2. **如果张量可能不连续，先用 contiguous()**
3. **不确定时用 reshape()（但可能牺牲性能）**

#### 6.squeeze 和 unsqueeze 函数的用法

```python
t1=torch.tensor(np.random.randint(0,10,[1,3,1,5]))

#1.squeeze使用
t2=t1.squeeze() #siez:[3,5]

#2.unsqueeze使用
t3=t2.unsqueeze(-1) #增加一个维度[3,5,1]
```

### 张量运算函数

计算均值、平方根、求和等等

```python
t1=torch.randint(0,10,[2,3],dtype=torch.float64)

#1.均值
t1.mean()#整体均值
t1.mean(dim=0)#按列
t1.mean(dim=1)#按行
#2.和
t1.sum()
...
#3.平方
t1.pow(2)
#4.平方根
t1.sqrt()
#等
```







