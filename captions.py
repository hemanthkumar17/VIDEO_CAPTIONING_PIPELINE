import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def get_model():
  processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
  model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
  return processor, model

def image_caption(processor, model, image_folder,image_path):
  raw_image = Image.open("/content/"+image_folder+"-opencv/"+ image_path)
  inputs = processor(raw_image, return_tensors="pt").to("cuda")
  out = model.generate(**inputs)
  return processor.decode(out[0], skip_special_tokens=True)

def get_captions(final_frames, processor, model):
  # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
  # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
  
  output = {}
  for k, v in final_frames.items():
    image_folder = k
    caption =""
    for k1, v1 in v.items():
      caption = caption + image_caption(processor, model, image_folder, v1) +". "
    output[image_folder] = caption
  return output