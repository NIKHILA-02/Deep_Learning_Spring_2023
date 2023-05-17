# Imports
import json
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# Loading data tokens
data = []

# Functions to process data tokens
with open('Flickr8k.token.txt', 'r') as f:
    for line in f:
        line = line.strip()
        parts = line.split('#')
        filename = parts[0].strip()
        caption = ' '.join(parts[1:]).strip()
        caption = caption.split("\t")[1]
        
        image_data = {
            "filename": filename,
            "caption": caption
        }
        data.append(image_data)

# data to json
with open('example.json', 'w') as f:
    json.dump(data, f)



# loading pretrained processor for tokenization
dataset = []
processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-base', add_special_tokens=True)

# Tokenizing images and their captions
for image_data in data:
    image_filename = os.path.join('Flicker8k_Dataset', image_data['filename'])
    image = Image.open(image_filename).convert('RGB')

    prompt = 'Describe the content of the image'
    answer = image_data['caption']
    inputs = processor(image, answer, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    dataset.append((image, inputs, answer))






# splitting dataset
train_set = dataset[:6000]
test_set = dataset[6000:]


# Function to preprocess data
def preprocess_dataset(dataset):
    return [(inputs, processor.tokenizer.encode(answer, return_tensors='pt')) for image, inputs, answer in dataset]


# preprocessing data
train_set = preprocess_dataset(train_set)
test_set = preprocess_dataset(test_set)


print(train_set[0])

# init blip pretrained model
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")



EPOCHS = 10
# running the train model
for epoch in range(EPOCHS):
    train_loss = 0.0

    # Training loop
    for inputs, labels in train_set:
        with tf.GradientTape() as tape:
            outputs = blip_model(inputs, training=True)
            loss = SparseCategoricalCrossentropy(labels, outputs)
            grads = tape.gradient(loss, blip_model.trainable_variables)
            Adam(lr=1e-5).apply_gradients(zip(grads, blip_model.trainable_variables))
            train_loss = loss.numpy() + train_loss


# Testing model
test_loss = 0.0
for inputs, labels in test_set:
    outputs = blip_model(inputs, training=False)
    loss = SparseCategoricalCrossentropy(labels, outputs)
    test_loss = loss.numpy() + test_loss





