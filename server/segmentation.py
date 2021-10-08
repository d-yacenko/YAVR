import sys, os
sys.path.insert(0, '../vision/')
import torchvision
from torchvision import models
import torch
from PIL import Image
from torchvision import transforms
import time

H,W = 720,1280
USE_CUDA=True
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#os.environ["CUDA_LAUNCH_BLOCKING"] = '0'
# models
#model = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
#model = models.segmentation.deeplabv3_resnet50(pretrained=1).eval()
model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=1).eval()
if USE_CUDA and torch.cuda.is_available():
    print('cuda is available')
    model.to('cuda',non_blocking=True)

# transforms
preprocess = transforms.Compose([
    transforms.Resize(540,interpolation=Image.NEAREST),
    #transforms.Resize(540,interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
postprocess = transforms.Compose([
    transforms.Resize(H,interpolation=Image.NEAREST),
    #transforms.Resize(H,interpolation=Image.BICUBIC),
])

def process(image_path):
    #start0 = time.time()
    input_image = Image.open(image_path)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    # move the input and model to GPU for speed if available
    if USE_CUDA and torch.cuda.is_available():
        input_batch = input_batch.to('cuda',non_blocking=True)
    with torch.no_grad():
        output = model(input_batch)['out'][0].half()
    #output_predictions = output.argmax(0)
    #pred=output_predictions.cpu()
    torch.cuda.synchronize()
    #start1 = time.time()
    output_predictions = output.data.argmax(0).to('cpu',non_blocking=True)
    pred=output_predictions #.argmax(0)
    input_batch.detach()
    output.detach()
    del input_batch
    del output
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    #end1 = time.time()
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    # plot the semantic segmentation predictions of 21 classes in each color
    #print(output_predictions.shape)
    r = Image.fromarray(pred.byte().numpy()).resize(input_image.size)
    r.putpalette(colors)
    image=postprocess(r)
    head, tail = os.path.split(image_path)
    image.convert('RGB').save(head+"/out"+tail,"JPEG")
    #end0 = time.time()
    #print(f"Runtime of the program is {end0 - start0}")
    #print(f"Runtime of the program is {end1 - start1}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('usage: segmentation.py file_to_mask.jpg')
        sys.exit(0)
    else:
        process(sys.argv[1])
