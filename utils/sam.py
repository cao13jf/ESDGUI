import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# segment anything
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils



def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
    
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    print(len(sorted_anns))
    # ax = plt.gca()
    # ax.set_autoscale_on(False)
    polygons = []
    color = []
    overlay = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        #ax.imshow(np.dstack((img, m*0.35)))
        
        overlay.append(np.dstack((img, m*0.35)))

    return overlay

def full_sam_predict(image_path, device=torch.device('cpu')):

    sam_checkpoint = '/home/a/Desktop/ESDGUI-Update/runs/sam_vit_h_4b8939.pth'
    model_type = 'vit_h'
    image_path = image_path
    #output_dir = '../sam_output'
    device = device

    
    # initialize SAM
    print('loading SAM model')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    print('load over')
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print('generate over')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, None, fx=0.8,fy=0.8, interpolation=cv2.INTER_AREA)
    masks = mask_generator.generate(image)
    

    decode_mask = []
    for m in masks:
        #print(m["segmentation"])
        #
        # color_map = np.random.randint(0, 255, size=(256, 1, 3)).astype(np.uint8)
        # 创建一个随机颜色的图像
        color_image = np.zeros(image.shape, dtype=np.uint8)
        color_image[:] = np.random.randint(1, 254, size=(3), dtype=np.uint8)
        binary_mask = np.where(m["segmentation"], 255, 0).astype(np.uint8)
        color_mask = cv2.bitwise_and(color_image, color_image, mask=binary_mask)
        decode_mask.append(color_mask)
    


    for m in decode_mask:
        
        image = cv2.addWeighted(image, 1.0, m, 0.5, -1.0)
    
    return masks, image

def prompt_sam_predict(image_path, input_box, image_dim, device=torch.device('cpu')):
    sam_checkpoint = '/home/a/Desktop/ESDGUI-Update/runs/sam_vit_h_4b8939.pth'
    model_type = 'vit_h'
    image_path = image_path
    #output_dir = '../sam_output'
    device = device

    
    # initialize SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_dim, interpolation = cv2.INTER_AREA)
    input_point = np.array([[int(0.5 * (input_box[0] + input_box[2])), int(0.5 * (input_box[1] + input_box[3]))]])
    input_label = np.array([0])
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
                                    point_coords=input_point,
                                    point_labels=input_label,
                                    box=input_box,
                                    multimask_output=False,
                                )
    m = masks[0]
    #print(m["segmentation"])
    #
    # color_map = np.random.randint(0, 255, size=(256, 1, 3)).astype(np.uint8)
    # 创建一个随机颜色的图像
    color_image = np.zeros(image.shape, dtype=np.uint8)
    color_image[:] = np.random.randint(1, 254, size=(3), dtype=np.uint8)
    binary_mask = np.where(m, 255, 0).astype(np.uint8)
    color_mask = cv2.bitwise_and(color_image, color_image, mask=binary_mask)
        
    image = cv2.addWeighted(image, 1.0, color_mask, 0.5, -1.0)
    
    return masks, image