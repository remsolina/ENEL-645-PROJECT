# Facial Expression Recognition

This project aims to develop a deep learning model for facial expression recognition (FER) using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The goal is to accurately classify facial expressions into seven categories: angry, disgust, fear, happy, sad, surprise, and neutral.

## Dataset

The dataset used in this project consists of grayscale facial images divided into "train" and "test" folders. The "train" folder is further split into training and validation sets using stratified sampling. Data augmentation techniques, such as random horizontal flipping and rotations, are applied to enhance the model's robustness.

## Approaches

The approach explored for this project was:

1. **CNN-RNN Model**: A hybrid model combining CNN layers for spatial feature extraction and RNN layers (LSTM) for capturing temporal dependencies. The CNN part consists of two convolutional layers with 32 and 64 filters, followed by ReLU activation and max pooling. The extracted features are then passed through a two-layer LSTM with 128 hidden units and dropout regularization. Finally, a fully connected layer with softmax activation produces the class probabilities.


## Results

The table below summarizes the performance of the model on the test set:

| Approach                 | Accuracy | Weighted F1-Score |
|--------------------------|----------|-------------------|
| CNN-RNN                  | 0.58     | 0.57              |

The CNN-RNN model achieved an accuracy of 58% and a weighted F1-score of 0.57.

## Limitations and Future Work

The current model the CNN-RNN struggle to achieve high accuracy on the facial expression recognition task, it  only reaches an accuracy of 58%, indicating significant room for improvement.

To enhance the performance of the facial expression recognition system, the following directions can be explored:

- Investigating advanced regularization techniques to mitigate overfitting, such as L1/L2 regularization, dropout, or data augmentation which we did but model was taking too long to train 
- Exploring other CNN architectures specifically designed for FER, such as the EmotionNet or FERNet.
- Incorporating temporal information from video sequences to capture the dynamics of facial expressions over time.
- Expanding the dataset with more diverse and balanced samples to improve the model's generalization ability.
- Conducting cross-dataset evaluations to assess the model's performance on unseen data from different domains.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Pandas


## License

This project is for ENEL645 group 10

## Acknowledgments

We would like to thank the creators of the facial expression dataset used in this project and the open-source community for providing valuable resources and pre-trained models.

If you have any questions or suggestions, please feel free to contact us.

Happy expression recognition!
