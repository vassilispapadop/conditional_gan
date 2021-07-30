# Conditional Generative Adversarial Network (cGAN)

```python
pip install -r requirements.txt
```

## Task Description
Train a Generative Adversarial Network, with architecture of your choice, for
generating FashionMNIST data. The model must be conditional: at the end of
training, you must be able to input the index of one of the FashionMNIST’s classes
that you wish to generate. Important: subset FashionMNIST to two classes:
T-shirt/top (0) vs. Pullover (2). 

## Loss Function of Generator and Discriminator
Write the mathematical formula of the loss functions for Generator and Discriminator.

## Training history(loss)
Plot the loss curves for Generator and Discriminator, and one example of generated images for
each class, paired with a corresponding one of the same class from FashionMNIST.

## Prediction
Generate 1k images from the Generator and sample 1k real images from
FashionMNIST, uniformly at random. What’s the accuracy of the Discriminator on this
new dataset? 

##
Question: how easy/hard is this task for the Discriminator? Why?

## Api with Flask
Finally, build the command line interface to call the Generator. It must receive one
string indicating the class of clothing to generate among “t-shirt” or “pullover”, and
return an image in the FashionMNIST format as response. Let the script fail
gracefully if any error is encountered. Make sure the s