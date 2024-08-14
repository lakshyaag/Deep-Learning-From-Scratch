# Contrastive Language-Image Pretraining (CLIP)

This project implements the CLIP (Contrastive Language-Image Pretraining) model, which is designed to learn visual concepts from natural language supervision. The reference paper can be found [here](https://arxiv.org/abs/2103.00020).

## Project Structure

- **data.py**: Contains the dataset class for loading image-text pairs.
- **config.py**: Configuration file with hyperparameters and model settings.
- **models.py**: Contains the definition of the CLIP model and its components.
- **trainer.py**: Script for training the CLIP model.
- **inference.py**: Script for running inference with the trained CLIP model.

## Key Components

### Model Architecture

The CLIP model consists of two main encoders:

1. **Image Encoder**: Uses a ResNet-50 model to encode images into feature vectors.
2. **Text Encoder**: Uses a DistilBERT model to encode text into feature vectors.

Both encoders are followed by projection heads that map the features into a common embedding space.

## Usage

### Dataset

I used the [Flickr8k dataset](https://github.com/goodwillyoga/Flickr8k_dataset) for the training, validation, and testing process. Since each image contains 5 captions, I randomly sample one caption for each image to create the image-text pairs in the dataset class.

### Training

The training process involves optimizing a contrastive loss that brings the embeddings of matching image-text pairs closer together while pushing apart the embeddings of non-matching pairs.

To train the model, run the `trainer.py` script:

```bash
python trainer.py
```

### Inference

The inference script allows for generating image embeddings and finding the best matching images for a given text query.

To run inference with the trained model, use the `inference.py` script:

```bash
python inference.py --weights /path/to/weights.pt --n 8
```

## References

- [CLIP: Contrastive Language-Image Pretraining](https://arxiv.org/abs/2103.00020)
- [OpenAI-CLIP on GitHub](https://github.com/moein-shariatnia/OpenAI-CLIP)
- [Implementing CLIP with PyTorch Lightning](https://wandb.ai/manan-goel/coco-clip/reports/Implementing-CLIP-With-PyTorch-Lightning--VmlldzoyMzg4Njk1)
