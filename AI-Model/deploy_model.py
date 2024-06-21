import numpy as np 
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class_dict = {
    0:'bishop', 1:'black-bishop', 2:'black-king', 3:'black-knight', 4:'black-pawn', 
    5:'black-queen', 6:'black-rook', 7:'white-bishop', 8:'white-king', 9:'white-knight', 
    10:'white-pawn', 11:'white-queen', 12:'white-rook'}

# Load a model
model = YOLO("./best.pt")  # load a custom model


device = "cuda" if torch.cuda.is_available() else "cpu"

resize = transforms.Compose(
                [ transforms.Resize((640,640)), transforms.ToTensor()])      
im = Image.open(
    "./cfc306bf86176b92ffc1afbb98d7896f_jpg.rf.effd71a5dcd98ec0f24072af5f7c0a31.jpg") 
img = resize(im) 
img = img.unsqueeze(0) # add fake batch dimension
img = img.to(device)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

img_outs = model(im)[0]

fig, ax = plt.subplots()
ax.imshow(im)
boxes = img_outs.boxes.xywh.cpu()
for i, img_out in enumerate(img_outs.boxes.data.tolist()):
    x1, y1, x2, y2, score, class_id = img_out

    x, y, w, h = boxes[i]
    if score < 0.2: continue

    rect = patches.Rectangle((int(x-(w/2)), int(y-(h/2))), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.text(int(x-(w/2)), int(y-(h/2)), class_dict[class_id], fontsize='xx-small')

plt.show()