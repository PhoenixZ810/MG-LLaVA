import torch
import torch.nn as nn

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from transformers import Owlv2Processor, Owlv2ForObjectDetection


class box_generator(nn.Module):
    def __init__(self, ram_path, owl_path, image_size=384):
        super().__init__()
        self.ram_transform = get_transform(image_size=image_size)
        ram_model = ram_plus(pretrained=ram_path, image_size=image_size, vit='swin_l')
        ram_model.cuda()
        ram_model.eval()
        self.ram_model = ram_model

        self.owl_processor = Owlv2Processor.from_pretrained(pretrained_model_name_or_path=owl_path)
        owl_model = Owlv2ForObjectDetection.from_pretrained(pretrained_model_name_or_path=owl_path)
        owl_model.cuda()
        owl_model.eval()
        self.owl_model = owl_model

    def forward(self, image_file):
        curr_dict = {}
        orig_image = Image.open(image_file)
        with torch.no_grad():
            image = self.ram_transform(orig_image).unsqueeze(0).cuda()
            res = inference(image, self.ram_model)
        tags = res[0].replace(' |', ',')
        tags = tags.lower()
        tags = tags.strip()
        tags = tags.split(',')
        tags = [tag.strip() for tag in tags]
        prompt_tags = ['a photo of ' + tag for tag in tags]

        orig_image = orig_image.convert("RGB")
        texts = [prompt_tags]
        with torch.no_grad():
            inputs = self.owl_processor(text=texts, images=orig_image, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.owl_model(**inputs)

        max_wh = max(orig_image.size)
        target_sizes = torch.Tensor([[max_wh, max_wh]])
        results = self.owl_processor.post_process_object_detection(
            outputs=outputs, threshold=0.2, target_sizes=target_sizes
        )

        index = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[index]
        boxes, scores, labels = results[index]["boxes"], results[index]["scores"], results[index]["labels"]

        boxes = boxes.int().cpu().numpy().tolist()
        scores = scores.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        labels = [tags[label] for label in labels]

        curr_dict['boxes'] = boxes
        curr_dict['scores'] = scores
        curr_dict['labels'] = labels

        return curr_dict
