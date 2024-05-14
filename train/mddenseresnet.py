from fastai.vision.all import *
from fastai.callback.fp16 import *
from models.MDNet import *

print(torch.cuda.is_available())

path = Path('./data/img/undersample_500')
path.ls()

"""## Dataset & Training"""

# Data Block
data_block = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(280, min_scale=0.50)
)

# Batch
dls = data_block.dataloaders(path, bs=8)

# Models : MDResNet-18/34, MDDenseNet-121 / 169 and MDDenseResNet
model = MDDenseResNet(ResidualDenseLayer, [6, 12, 24, 16], num_classes=9, bn_size=8)

dls.train.show_batch(max_n=5, nrows=1, unique=True)

# Learner with Model-ResNet-18/34/50
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=[accuracy, error_rate]).to_fp16()

# Strategy 1 : Find the optimal learning rate and Train
#learn.lr_find()

lr_min = 2e-4
learn.fit_one_cycle(50, slice(lr_min))

# Best Acc : 0.6500 (17 epoch)
learn.recorder.plot_loss()
#learn.recorder.plot_metrics()