import json
import os
import boto3
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("rsc/model", local_files_only=True)
model = AutoModelForImageClassification.from_pretrained("rsc/model", local_files_only=True)
s3 = boto3.client('s3')
print("Model loaded successfully")

# processor = AutoImageProcessor.from_pretrained(model_dir)
# model = AutoModelForImageClassification.from_pretrained(model_dir)
#for local test
# model_dir = "rsc\model"
# processor = AutoImageProcessor.from_pretrained(model_dir)
# model = AutoModelForImageClassification.from_pretrained(model_dir)
# image = Image.open("test.jpg").convert("RGB")
# input = processor(images=image, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(**input)
#     logits = outputs.logits
#     predicted_class_idx = logits.argmax(-1).item()
#     label = model.config.id2label[predicted_class_idx]
#     print(label)

def lambda_handler(event, context):
    print("Received event: ", event)
    record = event['Records'][0]['s3']
    bucket = record['bucket']['name']
    key = record['object']['key']

    image_obj = s3.get_object(Bucket=bucket, Key=key)
    image = Image.open(BytesIO(image_obj['Body'].read())).convert("RGB")
    input = processor(images=image, return_tensors="pt")
    print("Input prepared successfully")
    with torch.no_grad():
        outputs = model(**input)
    logits = outputs.logits
    print("Model inference completed successfully")
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]

    result = {
        "input_file": key,
        "predicted_label": label,
    }

    result_key = key.replace("uploads/", "results/").replace(".jpg", ".json")
    print("Result key: ", result_key)

    s3.put_object(
        Bucket=bucket,
        Key=result_key,
        Body=json.dumps(result).encode('utf-8'),
        ContentType='application/json'
    )
    return {
        'statusCode': 200,
        'body': json.dumps('Image processed successfully!')
    }