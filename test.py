# from segment_anything import SamPredictor, sam_model_registry
# from PIL import Image

# # Replace 'dog3.jpg' with the correct path to your image file
# image_path = 'dog3.jpg'

# # Load the image using PIL
# image = Image.open(image_path).convert("RGB")

# # Continue with the rest of your code

# sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
# predictor = SamPredictor(sam)
# predictor.set_image(image)
# masks, _, _ = predictor.predict(points)

import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

image = cv2.imread('dog3.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


predictor.set_image(image)
# single background point 
input_point = np.array([[1000, 375]])
input_label = np.array([0])
# multiple points
# input_point = np.array([[500, 375], [1000, 375]])
# input_label = np.array([1, 0])

# mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
# masks, _, _ = predictor.predict( point_coords=input_point,point_labels=input_label)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

print(masks.shape)
print(scores)
# print(logits)



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    



for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  
  
