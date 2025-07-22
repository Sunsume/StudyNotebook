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