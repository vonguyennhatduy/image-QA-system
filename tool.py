from langchain_core.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Đang tải model Image Captioning...")
blip_model_name = "Salesforce/blip-image-captioning-large"
blip_processor = BlipProcessor.from_pretrained(blip_model_name)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(device)

print("Đang tải model Object Detection...")
detr_model_name = "facebook/detr-resnet-50"
detr_processor = DetrImageProcessor.from_pretrained(detr_model_name)
detr_model = DetrForObjectDetection.from_pretrained(detr_model_name).to(device)


class ImageCaptionTool(BaseTool):
    name: str = "Image captioner"
    description: str = (
        "Use this tool when given the path to an image that you would like to be described. "
        "It will return a simple caption describing the image."
    )

    def _run(self, img_path: str) -> str:
        img_path = img_path.strip().strip('"').strip("'")
        image = Image.open(img_path).convert('RGB')

        # Sử dụng model đã được load sẵn ở trên
        inputs = blip_processor(image, return_tensors='pt').to(device)
        output = blip_model.generate(**inputs, max_new_tokens=20)

        caption = blip_processor.decode(output[0], skip_special_tokens=True)
    
        return caption

    async def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name: str = "Object detector"
    description: str = (
        "Use this tool when given the path to an image that you would like to detect objects. "
        "It will return a list of all detected objects. Each element in the list in the format: "
        "[x1, y1, x2, y2] class_name confidence_score."
    )

    def _run(self, img_path: str) -> str:
        img_path = img_path.strip().strip('"').strip("'")
        image = Image.open(img_path).convert('RGB')

        # Sử dụng model đã được load sẵn ở trên
        inputs = detr_processor(images=image, return_tensors="pt").to(device)
        outputs = detr_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(detr_model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        if not detections:
            return "No objects detected with high confidence."

        return detections

    async def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")