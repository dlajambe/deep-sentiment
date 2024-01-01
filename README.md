# Deep Sentiment

Using deep learning to determine the sentiment of movie reviews.

## Description

This project involves the creation and evaluation of a deep neural network to classify the sentiment of movie reviews as positive or negative. The IMDB movie reviews dataset was used to train and evaluate the model and is available in the `app/data` directory.

## Getting Started

### Dependencies

To install, simply clone the repository onto your machine and use your virtual environment creation tool to recreate the environment with the provided requirements.txt file. If you have Anaconda or Miniconda, you can instead use the environment.yml to reproduce the environment.

### Installing

To install, simply clone the repository onto your machine and use your virtual environment creation tool to recreate the environment with the provided `requirements.txt` file. If you have Anaconda or Miniconda, you can instead use the `environment.yml` to reproduce the environment.

### Configuration

The project includes a configuration file, `hyperparameters.py`, which can be used to modify the model architecture, training, and evaluation characteristics. These parameters can have a large impact on model performance and can be experimented with the optimize the 

### Executing program

The application can be started locally through the command line. With your virtual environment activated, navigate to the `app/` directory and execute the following command to train the model: 
```
python train_model.py
```

### Updating the environment files

Any time the application's virtual environment is modified, the environment files must be updated to reflect the change.

To update the `requirements.txt` file, run the following command:
```
pip list --format=freeze > requirements.txt
```
To update the `environment.yml` file, run the following command:
```
conda env export > environment.yml
```

## Authors

David LaJambe

## Acknowledgements

The dataset used in this project was originally created by the authors of the [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/) paper.

## License

This project is licensed under the Apache License version 2.0 - see the LICENSE.md file for details