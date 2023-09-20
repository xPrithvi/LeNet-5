# LeNet-5 on Jupyter Notebook
[LeNet-5](/Lecun98.pdf), a convolution neural network (CNN), for digit recognition was replicated in PyTorch and trained on the [MNIST dataset]([LeNet-5/Lecun98.pdf](https://www.kaggle.com/datasets/hojjatk/mnist-dataset))
on Jupyter Notebook with feature visualisation via a gradient-based method. The rectified linear unit (ReLU) was used instead of the hyperbolic tangent function used in the orginal paper. An accuracy of over 97% on the test dataset was 
achieved by model "LeNet-JFW4E".

## Future Improvements

- Hyperparameter tuning via Bayesian optimization.
- Implementation of the segmenter to allow for the extraction of multiple digits from noisy images.
