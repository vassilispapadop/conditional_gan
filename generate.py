
import os
import sys
from typing import List, Dict
import keras
from keras.models import load_model
import numpy as np
from matplotlib import pyplot
from datetime import datetime

SUPPORTED_LABELS = {
    't-shirt': 0,
    'pullover': 1
}

OUTPUT_FOLDER = 'generated_images'

def main() -> None:
    '''Entrypoint of program.'''
    # reads arguments
    try:
        print('Reading command line arguments.')
        args: Dict[str,str] = read_args()

        # checks integrity of arguments
        print('Checking validity of arguments.')
        check_args_validity(args)

        if not os.path.exists(OUTPUT_FOLDER):
            print('Creating output directory')
            os.makedirs(OUTPUT_FOLDER)

        # loads trained keras model
        print('Loading trained keras model.')
        model = load_keras_model(args['model_path'])
        # generates images and saves it to folder
        print(f'Generating image: {args["label"]}')
        generate_image(model=model, class_key = args['label'])
    except FileNotFoundError as Fe:
        print(Fe)
    except Exception as e:
        print(e)



def read_args() -> Dict[str,str]:
    '''Check for valid CLI arguments and return them in dictionary.'''
    if len(sys.argv) != 3:
        print("Usage: python generate.py [model] [label]")
        exit()

    return {
        'model_path': sys.argv[1],
        'label': sys.argv[2].lower()
    }

def check_args_validity(args):
    '''
    Checks for validity of arguments. 
    @model_path, checks whether the model exists and can be loaded into keras.
    @label, checks whether it is included in supported labels/classes.
    '''
    
    model_path = args['model_path']
    label = args['label']

    if not os.path.isfile(model_path):
        print(f'Warning: the model {model_path} does not exist. Please provide a valid model')
        exit()

    if label not in SUPPORTED_LABELS.keys():
        print(f'Warning: Passed argument {label} is not supported. Supported labels are {SUPPORTED_LABELS.keys()}')
        exit()


def load_keras_model(model_path) -> keras.models:
    '''Loads keras model'''
    model = load_model(model_path)
    return model


def generate_latent_points_by_class(latent_dim, n_samples, target_class) -> List:
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = np.empty(n_samples)
    labels.fill(target_class)
  
    return [z_input, labels]

def save_image(generated, title) -> None:
    # plot generated image
    pyplot.subplot(1, 1, 1)
    pyplot.axis('off')
    pyplot.title(title)
    pyplot.imshow(generated[0, :, :, 0], cmap='gray_r')

    file_name = f'{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}_{title}.png'
    file_path = os.path.join(OUTPUT_FOLDER, file_name)
    pyplot.savefig(file_path)
    print(f'Image saved at: {file_path}')



def generate_image(model, class_key) -> None:
    # generate latent points
    latent_points, labels = generate_latent_points_by_class(latent_dim=100, 
                                                            n_samples=1, 
                                                            target_class=SUPPORTED_LABELS[class_key])
    X  = model.predict([latent_points, labels])
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    save_image(X, class_key)

if __name__ == "__main__":
    main()