

# 自动微分模块

## 核心概念与机制

pyTorch 的自动微分模块 (`torch.autograd`) 是其核心组件，也是它成为流行深度学习框架的关键原因之一。它负责**自动计算张量（Tensor）操作的梯度**，极大地简化了神经网络的训练过程（尤其是反向传播）。

1. **计算图 (Computational Graph):**
   - 当你对 `Tensor` 进行操作（如加减乘除、矩阵乘法、激活函数等）时，`autograd` 会在后台**动态地构建一个计算图**。
   - 图中的节点：
     - **叶子节点 (Leaf Nodes):** 通常是用户直接创建的输入张量（例如，模型参数 `nn.Parameter`、输入数据）。
     - **中间节点 (Intermediate Nodes):** 由操作产生的张量。
     - **输出节点 (Output Node):** 最终的计算结果（例如，损失函数的值）。
   - 图中的边：代表张量之间的依赖关系和数据流动方向（正向传播）。
2. **张量属性 (requires_grad, grad, grad_fn):**
   - **requires_grad (布尔值):**
     - 如果一个 `Tensor` 的 `requires_grad=True`，则 `autograd` 会跟踪在其上执行的所有操作，并为其构建计算图。通常模型参数和输入数据的 `requires_grad` 会设为 `True`。
     - 默认情况下，新创建的张量 `requires_grad=False`。可以使用 `tensor.requires_grad_(True)` 或在创建时指定（如 `torch.tensor([...], requires_grad=True)`）来启用梯度跟踪。
   - **grad (属性):**
     - 当一个标量（通常是损失值）调用 `.backward()` 后，所有 `requires_grad=True` 的叶子节点的梯度会累积到这个属性中。
     - 它是一个与张量同形状的 `Tensor`，存储了损失函数对该张量的梯度 (`∂loss / ∂tensor`)。
     - 初始为 `None`，调用 `.backward()` 后才会被填充。需要**手动清零**（使用 `.grad.zero_()` 或优化器的 `optimizer.zero_grad()`），否则梯度会累加。
   - **grad_fn (属性):**
     - 指向创建该张量的 `Function` 对象。这个 `Function` 记录了生成该张量的操作以及反向传播时计算梯度所需的函数。
     - 叶子节点的 `grad_fn` 为 `None`（因为它们不是操作的结果）。
3. **Function 类:**
   - 计算图中的每个操作（节点）都对应一个 `torch.autograd.Function` 的子类实例。
   - 每个 `Function` 知道如何：
     - **正向计算 (Forward):** 执行实际的操作并产生输出。
     - **反向传播 (Backward):** 给定输出张量的梯度，计算输入张量的梯度（链式法则的具体实现）。这个函数定义了 `∂output / ∂input`。

## 工作流程（训练循环的核心）

1. **前向传播 (Forward Pass):**

   - 输入数据通过网络，执行各种操作（`Tensor` 操作）。
   - `autograd` **动态构建计算图**，记录操作序列和依赖关系。
   - 最终计算得到损失函数的值（一个标量 `loss`）。

2. **反向传播 (Backward Pass):**

   - 在损失值 `loss`（标量）上调用 `.backward()` 方法。
   - `autograd` **沿着计算图反向遍历**：
     - 从 `loss` 对应的 `grad_fn` 开始。
     - 调用每个 `Function` 的 `backward()` 方法，传入**该节点输出张量的梯度**（对于 `loss` 节点，初始梯度是 `1`，因为 `∂loss/∂loss = 1`）。
     - `Function.backward()` 计算并返回该操作**所有输入张量**的梯度 (`∂loss / ∂input`)。
     - 这些梯度通过链式法则传播到更早的节点（作为那些节点 `grad_fn.backward()` 的输入梯度）。
   - **累积梯度：** 当反向传播到达 `requires_grad=True` 的**叶子节点**（通常是模型参数）时，计算出的梯度 (`∂loss / ∂parameter`) 会**累积**到该张量的 `.grad` 属性中（不会覆盖，会累加！）。

3. **参数更新:**

   - 使用优化器 (`torch.optim`) 根据存储在 `.grad` 属性中的梯度更新模型参数。

4. **梯度清零 (Crucial!):**

   - 在下一次 `loss.backward()` 调用**之前**，必须将模型参数的 `.grad` 属性清零。否则梯度会不断累加，导致错误更新。

     ```python
     optimizer.zero_grad()  # 清空所有优化器管理的参数的.grad
     # 或者手动：
     # for param in model.parameters():
     #     param.grad = None  # 或 param.grad.zero_()
     ```

## 关键特性与优势

1. **动态图 (Define-by-Run):**
   - PyTorch 的计算图是在**代码执行时动态构建**的。这与 TensorFlow 1.x 的静态图（先定义图后执行）形成对比。
   - **优势：**
     - 更灵活：可以使用 Python 的控制流（if, for, while）、调试工具（如 pdb）等。
     - 更直观：模型结构直接由代码执行路径决定，易于理解和修改。
2. **按需反向传播:**
   - 只有在标量输出上调用 `.backward()` 时才会触发反向传播计算。
   - 可以只对图的特定部分进行反向传播（通过 `retain_graph` 参数或自定义 `Function`）。
3. **梯度累加:**
   - 默认行为是累加梯度（`.grad += new_gradients`）。这在需要累积多个 mini-batch 的梯度时（如模拟大 batch size）很有用，但通常需要使用 `zero_grad()` 来重置。
4. **分离计算 (detach()) 和停止梯度跟踪 (no_grad()):**
   - **tensor.detach():** 返回一个内容相同但 `requires_grad=False` 的新张量，将其从当前计算图中分离出来。用于防止某些计算被跟踪（例如，固定预训练网络的一部分）。
   - **torch.no_grad():** 上下文管理器。在该上下文中进行的所有操作都不会被 `autograd` 跟踪（`requires_grad` 被忽略）。用于**推理**（计算预测值）或在评估指标时，**节省内存和计算资源**。

## 示例

### eg1:计算y=x^2+20在x=10点的梯度

```python
import torch

#初始化一个x=10的Tesor
x = torch.tensor(10, requires_grad=True, dtype=torch.float64)
#定义损失函数
f=x**2+20
#反向传播计算梯度
f.backward()
#计算f在x=10的梯度
print(x.grad)

```

### eg2:多向量梯度计算

```python
#多向量梯度计算
# y = x1 ** 2 + x2 ** 2 + x1*x2
x1=torch.tensor([10,20],requires_grad=True,dtype=torch.float64)
x2=torch.tensor([30,40],requires_grad=True,dtype=torch.float64)
x3=torch.tensor([[1],[1]],requires_grad=True,dtype=torch.float64)
y1=x1**2+x2**2+x1*x2 #(100,400)+(900,1600)+(300,800)=(1300,2800)

#y和x3点乘
y2=y1@x3

y2.backward()

print(x1.grad)#  --->y=(x11 ** 2 + x21 ** 2 + x11*x21)+(x12 ** 2 + x22 ** 2 + x12*x22)   对于x11-> 2x11+x21=50    x12->  2x12+x22=80
print(x2.grad)#  --->y=(x11 ** 2 + x21 ** 2 + x11*x21)+(x12 ** 2 + x22 ** 2 + x12*x22)   对于x21-> 2x21+x11=70    x12->  2x22+x12=100
```

### eg3.控制不计算梯度

```python
#控制不计算梯度
#1.第一种方法：对代码装饰
x=torch.tensor(10,requires_grad=True,dtype=torch.float64)
with torch.no_grad():
    x=x**2+20
#2.对函数装饰
@torch.no_grad()
def funccc(x):
    return x**2+20
#3.
torch.set_grad_enabled(False)
y=x**2+20
```

### eg4.梯度清理使用

```python
for _ in range(100):
    y1 = x ** 2 + 20
    y2 = y1.mean()
    if x.grad is not None:
        x.grad.data.zero_()
    y2.backward()
    print(x.grad)
```

## 注意

当对设置 requires_grad=True 的张量使用 numpy 函数进行转换时, 会出现如下报错:

Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

此时, 需要先使用 detach 函数将张量进行分离, 再使用 numpy 函数.

注意: detach 之后会产生一个新的张量, 新的张量作为叶子结点，并且该张量和原来的张量共享数据, 但是分离后的张量不需要计算梯度。

```python
 x = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)
 x.detach().numpy()
```

