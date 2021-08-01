# Conditional Generative Adversarial Network (cGAN)

```sh
pip install -r requirements.txt
```

## Task Description [![1_Text_Embed](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
Train a Generative Adversarial Network, with architecture of your choice, for
generating FashionMNIST data. The model must be conditional: at the end of
training, you must be able to input the index of one of the FashionMNIST’s classes
that you wish to generate. Important: subset FashionMNIST to two classes:
T-shirt/top (0) vs. Pullover (2). 

## Loss Function of Generator and Discriminator
Write the mathematical formula of the loss functions for Generator and Discriminator.
![Alt text](images/loss_function.png?raw=true "Loss function")

## Training history(loss)
Plot the loss curves for Generator and Discriminator, and one example of generated images for
each class, paired with a corresponding one of the same class from FashionMNIST.
![Alt text](images/loss_history.png?raw=true "Loss history")
![Alt text](images/accuracy.png?raw=true "Accuracy history")

## Prediction
Generate 1k images from the Generator and sample 1k real images from
FashionMNIST, uniformly at random. What’s the accuracy of the Discriminator on this
new dataset? 


## CLI tool
We need to specify the model and the class we wish to generate.
```python
python3 generate.py modes/cgan_generator.h5 pullover
```