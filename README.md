# Damage State Prediction using Transfer Learning with ResNet18

This repository contains a project that utilizes **transfer learning** with a pre-trained **ResNet18** model to classify the damage state of objects into three categories. The dataset initially consisted of 300 images, which were augmented to increase the number of training samples and improve model performance. The model achieved an **accuracy of 91% on the test set**.

## Project Details:
- **Model**: ResNet18 (Pre-trained on ImageNet)
- **Task**: Damage state classification into 3 categories: 0-minor, 1-moderate, 2-severe
- **Accuracy**: 94.0% on the test set
- **Dataset**: : Dataset from [Abhijit85/InsuranceClaimImages](https://huggingface.co/datasets/Abhijit85/InsuranceClaimImages) on Hugging Face,300 images initially (80% for training, 20% for testing)
- **Data Augmentation**: Applied to increase training sample diversity
- **Transfer Learning**: Fine-tuned the convolutional layers with a small learning rate, and fully connected layers were trained from scratch.

<p align="center">
	<img src="https://raw.githubusercontent.com/danalejosolerma/fine-tuning/main/images/confusion.png" width="450" height="400" />
</p>

## Features:
- **Efficient Training**: Leveraged pre-trained ResNet18 to improve accuracy with limited data.
- **Data Augmentation**: Techniques like random flips and rotations were applied to enhance model generalization.
- **Reproducibility**: All training and testing steps are included for easy reproduction.

## Requirements:
- PyTorch
- torchvision
- PIL
- NumPy
