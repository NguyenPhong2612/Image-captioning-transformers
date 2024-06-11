# Project name
Image Captiong with Transformers

## [1]. Overview
My project focuses on image captioning, aiming to automatically generate descriptive captions for images using deep learning techniques. By combining computer vision with natural language processing, we strive to bridge the gap between visual content and textual descriptions.

You can access demo [here](https://huggingface.co/spaces/windy2612/ImageCaptioning)
## [2]. Code structure

-`Save/Weight` : Directory containing weight of pretrained model.

-`Save/text_data.npy` : Directory contain the caption.

-`Save/vocabulary.npy` : Directory contain the words that uses to predict caption.

-`app.py` : Script for deploy model.

-`make_dataset.py` : Script generate dataset.

-`model.py` : Script for define model architecture.

-`predict.py` : Script for generate caption for new images.

-`training.py` : Script for training.

## [3].Dataset
Project uses flickr8k dataset. The project uses the Flickr8k dataset. This is a dataset consisting of over 8000 images, with each image having 5 sample captions.

You can access the dataset from [here](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## [4].Model architecture
![Model architecture](Save/Stuff/Model.jpg)


The model architecture is inspired by the Transformers architecture, consisting of two parts: an encoder and a decoder.

### [4.1]Encoder
![Encoder architecture](Save/Stuff/Encoder.jpg)

The image above illustrates the overall architecture of the encoder block.
Step 1: A batch of images is passed through the ImageEmbedding to extract feature data.

Step 2: The features extracted in step 1 are then passed through one or more EncoderLayers to perform "self-attention".

Finally, we will receive a tensor that will be passed to the decoder as an input.

![ImageEmbedding architecture](Save/Stuff/ImageEmbedding.jpg)

The ImageEmbedding block uses the EfficientNet model to extract data from images. Using EfficientNet reduces computational costs.The output of ImageEmbedding wil feed to EncoderLayer as input.


![EncoderLayer architecture](Save/Stuff/EncoderLayer.jpg)

Each EncoderLayer integrates a MultiheadAttention network, Add&Norm, and FeedForward. The encoder consists of multiple stacked EncoderLayers, where the output of one layer becomes the input to the next.

Finally, the output of the encoder is a tensor used as input for the decoder block.

### [4.2]Decoder
![Decoder archietecture](Save/Stuff/Decoder.jpg)


Step 1: Pass the captions through the CaptionEmbedding block.

Step 2: After embedding the captions, we combine them with the encoder output through one or more DecoderLayer blocks.

Step 3 : Passed the output of step 2 to the classifier block. Finally we receive the result, and we passed the result as the input of decoder at the next timestep.


![CaptionEmbedding architecture](Save/Stuff/CaptionEmbedding.jpg)

The CaptionEmbeding block is tasked embedding the captions. 
It not only embeds the words in the caption but also combines positional embeddings of the words in the caption to ensure that words further apart have smaller correlation.The output of CaptionEmbedding wil feed to DecoderLayer as input.


![DecoderLayer architecture](Save/Stuff/DecoderLayer.jpg)

The decoder layer consists of three blocks: (MaskedMultiheadAttention + AddNorm), (MultiheadAttention + AddNorm), and (Feedforward + AddNorm).

Step 1: The first input is passed through the MaskedMultiheadAttention block for "self-attention". Using a mask ensures that words in the sentence can only "pay attention" to words before them. The output of the masked multihead attention is then passed through AddNorm for normalization.

Step 2: The output from step 1 is then inputted into the MultiheadAttention block to perform "cross-attention" with the encoder output. The tensor after "cross-attention" is then passed through AddNorm.

Step 3: The output from step 2 is passed through the Feedforward block to transform the logic, and then passed through AddNorm again.

Finally, we obtain a tensor that can be passed to the next DecoderLayer block or to the classifier block for classification.

## [5].Demo

After training, the model achieved an accuracy of 74.83% on the train dataset and 72.86% on the valid dataset. This is considered a fairly good result.

Below are some demos of the model:
![First](Save/Stuff/First.jpg)
![Second](Save/Stuff/Second.jpg)
![Third](Save/Stuff/Third.jpg)

# Thanks for your attention.







