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


###########################################


def get_image_caption(image_path):
    """
    Generates a short caption for the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string representing the caption for the image.
    """
    image = Image.open(image_path).convert('RGB')

    inputs = blip_processor(image, return_tensors='pt').to(device)
    output = blip_model.generate(**inputs, max_new_tokens=20)

    caption = blip_processor.decode(output[0], skip_special_tokens=True)

    return caption


def detect_objects(image_path):
    """
    Detects objects in the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string with all the detected objects. Each object as '[x1, y1, x2, y2] class_name confidence_score'.
    """
    image = Image.open(image_path).convert('RGB')

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


if __name__ == '__main__':
    image_path = 'tokyo.jpg'

    try:
        print("\n--- Bắt đầu nhận diện vật thể ---")
        detections = detect_objects(image_path)
        print(detections)

        print("\n--- Bắt đầu tạo chú thích ---")
        caption = get_image_caption(image_path)
        print(caption)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy ảnh tại đường dẫn {image_path}")