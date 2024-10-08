from fastai.vision.all import *
from fastai.callback.fp16 import *
from models.MDNet import *

print(torch.cuda.is_available())

path = Path('./data/img/malwarePix_small')
path.ls()

"""## Dataset & Training"""

# Data Block
data_block = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(1024, min_scale=0.50)
)

# Batch
dls = data_block.dataloaders(path, bs=32)

# Models : MDResNet-18/34, MDDenseNet-121 / 169
model = MDResNet18(pretrained=True)

dls.train.show_batch(max_n=5, nrows=1, unique=True)

# Learner with Model-ResNet-18/34/50
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=[accuracy, error_rate]).to_fp16()

# Strategy : Find the optimal learning rate and Train
#learn.lr_find()

lr_min = 3e-4
learn.fit_one_cycle(30, slice(lr_min))

# Best Acc : 0.8863 (28 epoch)
learn.recorder.plot_loss()
#learn.recorder.plot_metrics()
