# Sentiment-Analysis-with-Spacy-v3.0
Basic sentiment analysis to start with if you are a fanboy of spacy.

# Dataset
Dataset can be downloaded from kaggle here:
https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp

Make sure to extract and place it inside the data folder.
For custom datasets, you need to change the way the data is loaded in the load function. If you have any problems, feel free to reach out at venkyr91193@gmail.com.

# Installation
Use the command:
    >>> pip3 install -r requirements.pip

or install the requirements from the file by yourself.

# FastAPI
I have created a basic implementation of a server based fastAPI file if you want to use.
If you want to create a docker server, you can launch this api.py file inside the docker so that you can use your docker as a server. Change the body as per your requirements.

You will get the output in the format:
{
    'emotion': str
    'score': float
}

Feel free to reach out at venkyr91193@gmail.com if you need any help.
