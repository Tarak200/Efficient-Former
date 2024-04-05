import torch, time
import os
import argparse
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from util import *
from models import *

parser = argparse.ArgumentParser('Comparing models', add_help=False)
parser.add_argument('--data-path', default='3classData', type=str, help='dataset path')
parser.add_argument('--model-path', default='new_model/model.pth', type=str, help='model path')
parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--pin-mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
args = parser.parse_args()

root = os.path.join(args.data_path, 'train')

t = []
size = int((256 / 224) * args.input_size)
t.append(
    # to maintain same ratio w.r.t. 224 images
    transforms.Resize(size, interpolation=3),
)
t.append(transforms.CenterCrop(args.input_size))
t.append(transforms.ToTensor())
t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
dataset_test = datasets.ImageFolder(root, transform=transforms.Compose(t))
device = torch.device('cuda')


sampler_val = torch.utils.data.SequentialSampler(dataset_test)

data_loader_val = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
)



from transformers import  MobileNetV2ForImageClassification, AutoImageProcessor

mobileNet_model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
image_processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")

mobileNet_model.eval()
mobileNet_model.to(device)

start_time = time.time()

with torch.no_grad():
    for inputs, _ in data_loader_val:
        # inputs = image_processor(inputs, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = mobileNet_model(inputs).logits

end_time = time.time()
total_time = end_time - start_time

print(f"MobileNetV2")
print(f"Total inference time: {total_time} seconds")
print(f"Average inference time per sample: {total_time / len(data_loader_val.dataset)} seconds")



from transformers import  PoolFormerForImageClassification

Poolformermodel = PoolFormerForImageClassification.from_pretrained("sail/poolformer_s12")
Poolformermodel.to(device)
Poolformermodel.eval()


start_time = time.time()

with torch.no_grad():
    for inputs, _ in data_loader_val:
        # inputs = image_processor(inputs, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = Poolformermodel(inputs).logits

end_time = time.time()
total_time = end_time - start_time

print(f"PoolFormer-S12")
print(f"Total inference time: {total_time} seconds")
print(f"Average inference time per sample: {total_time / len(data_loader_val.dataset)} seconds")



Efficientformer_model = torch.load(args.model_path)
Efficientformer_model.eval()
Efficientformer_model.to(device)

start_time = time.time()
with torch.no_grad():
    for inputs, _ in data_loader_val:
        inputs = inputs.to(device)
        outputs = Efficientformer_model(inputs)
        
end_time = time.time()
total_time = end_time - start_time

# print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
print(f"EfficientFormer")
print(f"Total inference time: {total_time} seconds")
print(f"Average inference time per sample: {total_time / len(data_loader_val.dataset)} seconds")
