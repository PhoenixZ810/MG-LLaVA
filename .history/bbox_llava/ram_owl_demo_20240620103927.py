import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

image_path = 'recognize-anything/images/demo/demo1.jpg'
pth_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/recognize-anything-plus-model/ram_plus_swin_large_14m.pth'
image_size = 384

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = get_transform(image_size=image_size)
model = ram_plus(pretrained=pth_path,
                 image_size=image_size,
                 vit='swin_l')
model.eval()
model = model.to(device)
orig_image = Image.open(image_path)
image = transform(orig_image).unsqueeze(0).to(device)

res = inference(image, model)
tags = res[0].replace(' |', ',')
print("Image Tags: ", tags)

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from mmengine.visualization import Visualizer
import numpy as np
import cv2

tags = tags.lower()
tags = tags.strip()
tags = tags.split(',')
print("Image Tags: ", tags)
tags = tags.split(',')
tags = [tag.strip() for tag in tags]
prompt_tags = ['a photo of ' + tag for tag in tags]

path = '/mnt/hwfile/mm_dev/zhaoxiangyu/models--google--owlv2-large-patch14-ensemble/snapshots/d638f16c163f70a8b6bd643b2ddbfc8be2c34807/'
processor = Owlv2Processor.from_pretrained(pretrained_model_name_or_path=path)
model = Owlv2ForObjectDetection.from_pretrained(pretrained_model_name_or_path=path)

orig_image = orig_image.convert("RGB")
texts = [prompt_tags]
inputs = processor(text=texts, images=orig_image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
max_wh = max(orig_image.size)
target_sizes = torch.Tensor([[max_wh, max_wh]])
print(target_sizes)
# target_sizes = torch.Tensor([image.shape[::2]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(outputs=outputs, threshold=0.2, target_sizes=target_sizes)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

visualizer = Visualizer()
visualizer.set_image(np.array(orig_image))

# Print detected objects and rescaled box coordinates
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

    visualizer.draw_bboxes(np.array(box).reshape(-1, 4))
    visualizer.draw_texts(str({round(score.item(), 3)}), positions=np.array(box[:2]).reshape(-1, 2))

drawn_img = visualizer.get_image()
cv2.imwrite("owlv2.jpg", drawn_img[..., ::-1])
