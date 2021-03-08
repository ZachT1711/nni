使用 Retiarii 进行神经网络架构搜索（实验性）
==============================================================================================================

`Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__ 是一个支持神经体系架构搜索和超参数调优的新框架。 它允许用户以高度的灵活性表达各种搜索空间，重用许多前沿搜索算法，并利用系统级优化来加速搜索过程。 该框架提供了以下全新的用户体验。

* 搜索空间可以直接在用户模型代码中表示。 调优空间可以通过定义模型来表示。
* 在 Experiment 中，神经架构候选项和超参数候选项得到了更友好的支持。
* Experiment 可以直接从 Python 代码启动。

NNI 正在把 `之前 NAS 框架 <../Overview.rst>`__ *迁移至Retiarii框架。 因此，此功能仍然是实验性的。 NNI 建议用户尝试新的框架，并提供有价值的反馈来改进它。 旧框架目前仍受支持。*

.. contents::

有两个步骤来开始神经架构搜索任务的 Experiment。 首先，定义要探索的模型空间。 其次，选择一种搜索方法来探索您定义的模型空间。

定义模型空间
-----------------------

模型空间是由用户定义的，用来表达用户想要探索、认为包含性能良好模型的一组模型。 在这个框架中，模型空间由两部分组成：基本模型和基本模型上可能的突变。

定义基本模型
^^^^^^^^^^^^^^^^^

定义基本模型与定义 PyTorch（或 TensorFlow）模型几乎相同， 只有两个小区别。

* 对于 PyTorch 模块（例如 ``nn.Conv2d``, ``nn.ReLU``），将代码 ``import torch.nn as nn`` 替换为 ``import nni.retiarii.nn.pytorch as nn`` 。
* Some **user-defined** modules should be decorated with ``@basic_unit``. 例如，``LayerChoice`` 中使用的用户定义模块应该被修饰。 Users can refer to `here <#serialize-module>`__ for detailed usage instruction of ``@basic_unit``.

下面是定义基本模型的一个简单的示例，它与定义 PyTorch 模型几乎相同。

.. code-block:: python

  import torch.nn.functional as F
  import nni.retiarii.nn.pytorch as nn

  class MyModule(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv = nn.Conv2d(32, 1, 5)
      self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
      return self.pool(self.conv(x))

  class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.mymodule = MyModule()
    def forward(self, x):
      return F.relu(self.mymodule(x))

可参考 :githublink:`Darts 基本模型 <test/retiarii_test/darts/darts_model.py>` 和 :githublink:`Mnasnet 基本模型 <test/retiarii_test/mnasnet/base_mnasnet.py>` 获取更复杂的示例。

定义模型突变
^^^^^^^^^^^^^^^^^^^^^^

基本模型只是一个具体模型，而不是模型空间。 我们为用户提供 API 和原语，用于把基本模型变形成包含多个模型的模型空间。

**以内联方式表示突变**

为了易于使用和向后兼容，我们提供了一些 API，供用户在定义基本模型后轻松表达可能的突变。 API 可以像 PyTorch 模块一样使用。

* ``nn.LayerChoice``， 它允许用户放置多个候选操作（例如，PyTorch 模块），在每个探索的模型中选择其中一个。 *Note that if the candidate is a user-defined module, it should be decorated as `serialize module <#serialize-module>`__. 在下面的例子中，``ops.PoolBN`` 和 ``ops.SepConv`` 应该被修饰。

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # 在 `__init__` 中声明
    self.layer = nn.LayerChoice([
      ops.PoolBN('max', channels, 3, stride, 1),
      ops.SepConv(channels, channels, 3, stride, 1),
      nn.Identity()
    ]))
    # 在 `forward` 函数中调用
    out = self.layer(x)

* ``nn.InputChoice``， 它主要用于选择（或尝试）不同的连接。 它会从设置的几个张量中，选择 ``n_chosen`` 个张量。

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # 在 `__init__` 中声明
    self.input_switch = nn.InputChoice(n_chosen=1)
    # 在 `forward` 函数中调用，三者选一
    out = self.input_switch([tensor1, tensor2, tensor3])

* ``nn.ValueChoice``， 它用于从一些候选值中选择一个值。 It can only be used as input argument of the modules in ``nn.modules`` and ``@basic_unit`` decorated user-defined modules.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # 在 `__init__` 中使用
    self.conv = nn.Conv2d(XX, XX, kernel_size=nn.ValueChoice([1, 3, 5])
    self.op = MyOp(nn.ValueChoice([0, 1], nn.ValueChoice([-1, 1]))

详细的 API 描述和使用说明在 `这里 <./ApiReference.rst>`__。 使用这些 API 的示例在 :githublink:`Darts base model <test/retiarii_test/darts/darts_model.py>`。

**用 Mutator 表示突变**

尽管内联突变易于使用，但其表达能力有限，无法表达某些模型空间。 为了提高表达能力和灵活性，我们提供了编写 *Mutator* 的原语，方便用户更灵活地修改基本模型。 Mutator 位于基础模型之上，因此具有编辑模型的全部能力。

用户可以按以下方式实例化多个 Mutator，这些 Mutator 将依次依次应用于基本模型来对新模型进行采样。

.. code-block:: python

  applied_mutators = []
  applied_mutators.append(BlockMutator('mutable_0'))
  applied_mutators.append(BlockMutator('mutable_1'))

``BlockMutator`` 由用户定义，表示如何对基本模型进行突变。 用户定义的 Mutator 应该继承 ``Mutator`` 类，并在成员函数 ``mutate`` 中实现突变逻辑。

.. code-block:: python

  from nni.retiarii import Mutator
  class BlockMutator(Mutator):
    def __init__(self, target: str, candidates: List):
        super(BlockMutator, self).__init__()
        self.target = target
        self.candidate_op_list = candidates

    def mutate(self, model):
      nodes = model.get_nodes_by_label(self.target)
      for node in nodes:
        chosen_op = self.choice(self.candidate_op_list)
        node.update_operation(chosen_op.type, chosen_op.params)

``mutate`` 的输入是基本模型的 graph IR（请参考 `这里 <./ApiReference.rst>`__ 获取 IR 的格式和 API），用户可以使用其成员函数（例如， ``get_nodes_by_label``，``update_operation``）对图进行变异。 变异操作可以与 API ``self.choice`` 相结合，以表示一组可能的突变。 在上面的示例中，节点的操作可以更改为 ``candidate_op_list`` 中的任何操作。

使用占位符使突变更容易：``nn.Placeholder``。 如果要更改模型的子图或节点，可以在此模型中定义一个占位符来表示子图或节点。 然后，使用 Mutator 对这个占位符进行变异，使其成为真正的模块。

.. code-block:: python

  ph = nn.Placeholder(
    label='mutable_0',
    kernel_size_options=[1, 3, 5],
    n_layer_options=[1, 2, 3, 4],
    exp_ratio=exp_ratio,
    stride=stride
  )

``label`` is used by mutator to identify this placeholder. The other parameters are the information that are required by mutator. They can be accessed from ``node.operation.parameters`` as a dict, it could include any information that users want to put to pass it to user defined mutator. 完整的示例代码在 :githublink:`Mnasnet base model <test/retiarii_test/mnasnet/base_mnasnet.py>`。

探索定义的模型空间
------------------------------------------

在模型空间被定义之后，是时候探索这个模型空间了。 Users can choose proper search and model evaluator to explore the model space.

Create an Evaluator and Exploration Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**经典搜索方法：**
In this approach, model evaluator is for training and testing each explored model, while strategy is for sampling the models. Both evaluator and strategy are required to explore the model space. We recommend PyTorch-Lightning to write the full evaluation process.

**Oneshot（权重共享）探索方法：**
In this approach, users only need a oneshot trainer, because this trainer takes charge of both search, training and testing.

In the following table, we listed the available evaluators and strategies.

.. list-table::
  :header-rows: 1
  :widths: auto

  * - Evaluator
    - Strategy
    - Oneshot Trainer
  * - 分类
    - TPEStrategy
    - DartsTrainer
  * - 回归
    - Random
    - EnasTrainer
  * - 
    - GridSearch
    - ProxylessTrainer
  * - 
    - RegularizedEvolution
    - SinglePathTrainer (RandomTrainer)

使用说明和 API 文档在 `这里 <./ApiReference>`__。

Here is a simple example of using evaluator and strategy.

.. code-block:: python

  import nni.retiarii.evaluator.pytorch.lightning as pl
  from nni.retiarii import serialize
  from torchvision import transforms

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
  test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)
  lightning = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=10)

.. Note:: For NNI to capture the dataset and dataloader and distribute it across different runs, please wrap your dataset with ``serialize`` and use ``pl.DataLoader`` instead of ``torch.utils.data.DataLoader``. See ``basic_unit`` section below for details.

Users can refer to `API reference <./ApiReference.rst>`__ on detailed usage of evaluator. 参考 "`此文档 <./WriteTrainer.rst>`__" 编写一个新的 Trainer，参考 `此文档 <./WriteStrategy.rst>`__ 编写一个新的 Strategy。

发起 Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

上述内容准备就绪之后，就可以发起 Experiment 以进行模型搜索了。 NNI 设计了统一的接口来发起 Experiment， 示例如下：

.. code-block:: python

  exp = RetiariiExperiment(base_model, trainer, applied_mutators, simple_strategy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnasnet_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 10
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)

此代码发起了一个 NNI Experiment， 注意，如果使用内联突变，``applied_mutators`` 应为 ``None``。

一个简单 MNIST 示例的完整代码在 :githublink:`这里 <test/retiarii_test/mnist/test.py>`。

可视化 Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^

用户可以像可视化普通的超参数调优 Experiment 一样可视化他们的 Experiment。 例如，在浏览器里打开 ``localhost::8081``，8081 是在 ``exp.run`` 里设置的端口。 参考 `这里 <../../Tutorial/WebUI.rst>`__ 了解更多细节。 如果用户使用的是 Oneshot Trainer，可以参考 `这里 <../Visualization.rst>`__ 去可视化他们的 Experiment。

导出 Experiment 中发现的最佳模型
---------------------------------------------------------------------

如果您使用的是\ *经典搜索方法*，那么您可以从 WebUI 中找到最好的模型。

如果您使用的是 *Oneshot（权重共享）搜索方法*，则可以使用 ``exp.export_top_models`` 导出 Experiment 中发现的几个最佳模型。

高级功能和常见问题
--------------------------------

.. _serialize-module:

**Serialize Module**

To understand the decorator ``basic_unit``, we first briefly explain how our framework works: it converts user-defined model to a graph representation (called graph IR), each instantiated module is converted to a subgraph. 然后将用户定义的突变应用于图上以生成新的图， 并将每个新图转换回 PyTorch 代码执行。 ``@basic_unit`` here means the module will not be converted to a subgraph but is converted to a single graph node. 也就是说，该模块将不再展开。 在以下情况下，用户应该/可以修饰自定义的模块类：

* 当模块类由于某些实现问题无法成功转换为子图时。 For example, currently our framework does not support adhoc loop, if there is adhoc loop in a module's forward, this class should be decorated as serializeble module. 下面的 ``MyModule`` 应该被修饰：

  .. code-block:: python

    @basic_unit
    class MyModule(nn.Module):
      def __init__(self):
        ...
      def forward(self, x):
        for i in range(10): # <- adhoc loop
          ...

* The candidate ops in ``LayerChoice`` should be decorated as serializable module. 例如，在 ``self.op = nn.LayerChoice([Op1(...), Op2(...), Op3(...)])``中，如果 ``Op1``, ``Op2``, ``Op3`` 是用户自定义的模块，则应该被修饰。
* When users want to use ``ValueChoice`` in a module's input argument, the module should be decorated as serializable module. 例如，在 ``self.conv = MyConv(kernel_size=nn.ValueChoice([1, 3, 5]))`` 中，``MyConv`` 应该被修饰。
* If no mutation is targeted on a module, this module *can be* decorated as a serializable module.
