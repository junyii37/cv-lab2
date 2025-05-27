# README

本项目为计算机视觉课程的期中项目，由两个子任务构成。本文档将分别介绍如何配置环境，如何准备模型和数据，如何训练以及测试模型。

项目结构如下所示：

```
cv-lab2
├── task1
│   ├── logs	# 不同参数下的训练日志
│   ├── results		# best model的测试结果
│   ├── dataset_idx.pkl		# 数据集划分策略
│   ├── eval.py
│   ├── train.py
│   └── requirements.txt
└── task2
    └── mmdetection
```

---

### Task1 微调在ImageNet上预训练的卷积神经网络实现Caltech-101分类

#### 1.1 环境配置

(建议为两个Task创建不同环境，以避免可能的版本不兼容情况)

```
pip install -r task1/requirements
```



#### 1.2 模型与数据集准备

##### 1.2.1 数据集准备

Task1用到的数据集为Caltech-101.zip，准备步骤如下：

（1）从https://drive.google.com/file/d/17UPyjHq6O8DhnkyzzDnwxRDRdH3rxa2P/view?usp=drive_link下载`caltech-101.zip`。

（2）解压缩后放在`cv-lab2/task1/data`目录下（需新建目录`data`）。

##### 1.2.2 模型准备

从https://drive.google.com/file/d/1D2eiLp069wnVMpno9AKH6YfkDTDi_brw/view?usp=drive_link下载`best_model.pth`，直接放在`cv-lab2/task1`目录下



准备完成后，文件结构应如下所示：

```
task1
├── logs	# 不同参数下的训练日志
├── results		# best model的测试结果
├── data
│   └── caltech-101
├── dataset_idx.pkl		# 训练时划分的数据集对应的索引
├── best_model.pth
├── eval.py
├── train.py
└── requirements.txt
```



#### 1.3 训练及测试模型

##### 1.3.1 训练

如果想要训练模型，可以在项目根目录下输入：

```
cd task1
python train.py
```

即可以用默认配置训练resnet模型。

注意：如果在windows系统上运行，需要修改如下参数，否则可能发生进程堵塞。

```
python train.py --num_workers 0
```



其他参数可以通过如下命令查看：

```
python train.py -h
```

使用示例：

1. 调整batch_size为64：

   ```
   python train.py --batch_size 64
   ```

2. 使用更大的学习率：

   ```
   python train.py --lr 0.1
   ```

3. 调整训练时长为5个epoch：

   ```
   python train.py --epoch 5
   ```

4. 禁用预训练模型：

   ```
   python train.py --unpretrained
   ```



训练得到的最佳模型将位于`task1/best_model.pth`，同时在`logs`目录下会新增该次训练时间戳对应的文件夹，其中包含（1）训练配置文件`cfg.json`（2）测试结果`result.txt`（3）tensorboard可视化文件



##### 1.3.2 测试

如果想要测试现有模型，可以在项目根目录下输入：

```
cd task1
python eval.py
```

即可以评估训练好的resnet模型在测试集上的效果。

注意：如果在windows系统上运行，需要修改如下参数，否则可能发生进程堵塞。

```
python eval.py --num_workers 0
```



其他参数可以通过如下命令查看：

```
python eval.py -h
```

使用示例：

1. 加载不同位置的模型：

   ```
   python eval.py --model_path {MODEL_PATH}
   ```

2. 改变输出目录：

   ```
   python eval.py --output_dir {OUTPUT_DIR}
   ```

---

### Task2 在VOC数据集上训练并测试模型Mask R-CNN和Sparse R-CNN

#### 2.1  环境配置

仿照mmdetection的帮助文档进行环境配置：

**Step 0.** 安装pytorch

**Step 1.** 使用MIM安装MMEngine和MMCV

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 2.** 使用MIM安装MMEngine和MMCV

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

**Step3.** 验证安装

```
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```

```
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```

你会在当前文件夹中的 `outputs/vis` 文件夹中看到一个新的图像 `demo.jpg`，图像中包含有网络预测的检测框。



#### 2.2 模型与数据集准备

##### 2.2.1 数据集准备

Task2所用的数据集为VOC2012，准备步骤如下：

（1）从https://drive.google.com/file/d/11wnTQeAacWj7YbQSsuWuwPZlnymZL458/view?usp=drive_link下载`coco.zip`文件；

（2）解压缩后放到`cv-lab2/task2/mmdetection/data`目录下（需新建`data`目录）

##### 2.2.2 模型准备

**Mask R-CNN**：https://drive.google.com/file/d/1wOElxm27xSn9ufXxO2AI-oq3xhW-sUxk/view?usp=drive_link

**Sparse R-CNN**：https://drive.google.com/file/d/12deFQk-vcMPOoEFXMCu--mrg3XnwJ8VR/view?usp=drive_link

下载后将两个模型文件放到`cv-lab2/task2/mmdetection/models`目录下（需新建`models`目录）



#### 2.3 训练及测试模型

##### 2.3.1 训练

首先移动到mmdetection目录下：

```
cd task2/mmdetection
```

训练Mask R-CNN模型：

```
python tools/train.py configs/voc2012_mask_rcnn_r50_fpn.py
```

训练Sparse R-CNN模型：

```
python tools/train.py configs/voc2012_sparse_rcnn_r50_fpn_1x_coco.py
```

训练好的模型和训练日志将位于`mmdetection/work_dirs`里面对应的模型目录下



##### 2.3.2 测试

测试下载的Mask R-CNN模型：

```
python tools/test.py configs/voc2012_mask_rcnn_fpn.py models/mask_rcnn.pth
```

测试下载的Sparse R-CNN模型：

```
python tools/test.py configs/voc2012_sparse_rcnn_r50_fpn_1x_coco.py models/sparse_rcnn.pth
```

测试结果和日志同样位于`mmdetection/work_dirs`里面对应的模型目录下



##### 2.3.3 demo

此外，如果只想可视化模型在少数几张图片上的效果，可以采取如下步骤：

（1）在`mmdetection/demo`目录下新建`samples`目录，将图片置于其中。

（2）输入如下指令：

Mask R-CNN：

```
python demo/image_demo.py demo/samples configs/voc2012_mask_rcnn_fpn.py \ 
--weights models/mask_rcnn.pth \
--out-dir mask_rcnn_outputs
```

Sparse R-CNN:

```
python demo/image_demo.py demo/samples configs/voc2012_sparse_rcnn_r50_fpn_1x_coco.py \ 
--weights models/sparse_rcnn.pth \
--out-dir sparse_rcnn_outputs
```

即可在对应的`out-dir`目录下看到可视化结果。

