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
The standard MinMax loss function of GAN is the following. The generator tries to minimize the function while the discrimitator to maximize it.

### Standard Loss
![Alt text](images/loss_function.png?raw=true "Loss function")
* D(x) is the discriminator's estimate of the probability that real data instance x is real.
* Ex is the expected value over all real data instances.
* G(z) is the generator's output when given noise z.
* D(G(z)) is the discriminator's estimate of the probability that a fake instance is real.
* Ez is the expected value over all random inputs to the generator (in effect, the expected value over all generated fake instances G(z)).
* The formula derives from the cross-entropy between the real and generated distributions.

### Conditional Loss
The conditioning is usually done by feeding the information y into both the discriminator and the generator, as an additional input layer to it. The only difference between them is that a conditional probability is used for both the generator and the discriminator, instead of the regular one
![Alt text](images/loss_function_cgan.png?raw=true "Conditional Loss function")

## Training history(loss)
Plot the loss curves for Generator and Discriminator, and one example of generated images for
each class, paired with a corresponding one of the same class from FashionMNIST.
![Alt text](images/loss_history.png?raw=true "Loss history")
![Alt text](images/accuracy.png?raw=true "Accuracy history")

## Prediction
![Alt text](images/fake_vs_real_pullover.png?raw=true "T-shirt generated")
![Alt text](images/fake_vs_real_tshirt.png?raw=true "Pullover generated")

Generate 1k images from the Generator and sample 1k real images from
FashionMNIST, uniformly at random. What’s the accuracy of the Discriminator on this
new dataset? 
![Alt text](images/accuracy_on_new_dataset.png?raw=true "Synthetic dataset accuracy")
As the generator improves with training, the discriminator performance gets worse because the discriminator can't easily tell the difference between real and fake. If the generator succeeds perfectly, then the discriminator has a 50% accuracy.

## CLI tool
In order to use the model a cli tool is developed *generate.py*. We need to specify the model and the class we wish to generate.
The tool simply checks the number of arguments and the validity, loads the model and outputs the generated image into a pre-defined directory.
Usage:
```python
python3 generate.py models/cgan_generator.h5 pullover
```

## Future work

* Latent Space Size. Arbitrarily is set to *100*. Experimentation of Latent space size on the impact it has on the quality of generated images.

* Embedding Size. Arbitrarily is set to *50*. Experimentation of Embedding space size on the impact it has on the quality of generated images.


## References

[1] Generative Adversarial Networks (GANs). [link](https://arxiv.org/pdf/1406.2661.pdf)

[2] Are conditional GANs explicitly conditional?. [link](https://arxiv.org/pdf/2106.15011.pdf)

[3] Conditional GAN. [link](https://keras.io/examples/generative/conditional_gan/)

[4] GAN training. [link](https://developers.google.com/machine-learning/gan/training)