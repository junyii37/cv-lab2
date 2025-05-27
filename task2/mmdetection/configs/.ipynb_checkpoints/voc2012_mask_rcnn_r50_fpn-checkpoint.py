_base_ = [
    '_base_/models/mask-rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance_1.py',
    '_base_/schedules/schedule_1x.py', '_base_/default_runtime.py'
]

'''
# 1. 数据集类型和类别列表
dataset_type = 'CocoDataset'  # 以离线 COCO 格式为准
data_root = 'data/VOC2012_coco/'  
classes = ('aeroplane','bicycle','bird','boat','bottle',
           'bus','car','cat','chair','cow',
           'diningtable','dog','horse','motorbike','person',
           'pottedplant','sheep','sofa','train','tvmonitor')

# 2. 数据加载
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train2012.json',
        img_prefix=data_root + 'train2012/'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2012.json',
        img_prefix=data_root + 'val2012/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2012.json',
        img_prefix=data_root + 'val2012/')
)  # 保留原 schedule 与 runtime，且无需再导入 coco_instance.py 中的路径

# 3. 模型类别数修正
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)
    )
)
'''