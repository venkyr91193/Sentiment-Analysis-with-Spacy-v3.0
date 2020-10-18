import argparse
import os
import random
from pathlib import Path

import pandas as pd
import spacy
from spacy.language import Language
from spacy.training import Example
from spacy.util import compounding, decaying, minibatch

# more lower the dropout, the more overfitting
# dropout
dropout = decaying(0.5, 1e-4)


# needed to add components in the new v3.0 release
@Language.factory("my_component")
def create_my_component(nlp, name):
    return MyComponent(nlp)

class MyComponent:
    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, doc):
        return doc

def load_data():
    """
    Function: to load the data
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
    df_train = pd.read_csv(os.path.join(data_path,'train.txt'),header=None,sep=';')
    df_test = pd.read_csv(os.path.join(data_path,'test.txt'),header=None,sep=';')
    df_val = pd.read_csv(os.path.join(data_path,'val.txt'),header=None,sep=';')
    labels = list(set(df_train[1].values))

    def format_data(df: pd.DataFrame):
        """
        Function: to format the data to train spacy textcat
        """
        data = list()
        for index,sent in enumerate(df[0].values):
            temp_dict = dict()
            for label in labels:
                if label == df[1].values[index]:
                    temp_dict[label] = 1
                else:
                    temp_dict[label] = 0
            data.append((sent,{"cats": temp_dict}))
        return data

    train_data = format_data(df_train)
    test_data = format_data(df_test)
    val_data = format_data(df_val)
    return train_data,test_data,val_data,labels

def evaluate(tokenizer, textcat, data):
    texts = [dat[0] for dat in data]
    cats = [dat[1]["cats"] for dat in data]
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

def train(args):
    """
    Function: to train the textcat
    """
    # load data and get the labels set
    train_data,test_data,val_data,LABELS = load_data()

    # declare the nlp object
    # length of language ISO == 2
    if len(args.model_name) == 2:
        nlp = spacy.blank(args.model_name)
    else:
        nlp = spacy.load(args.model_name)

    # adding textcat in pipe
    if "textcat" not in nlp.pipe_names:
        # default config={"exclusive_classes": True}
        nlp.add_pipe("textcat")
    textcat = nlp.get_pipe("textcat")

    # adding labels
    for label in LABELS:
        textcat.add_label(label)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(*other_pipes):
        # use resume_training() if pretrained model is used
        # length of a language iso == 2
        if len(args.model_name) == 2:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()
         # start training
        for iteration in range(args.n_iter):
            print(f"Iteration : {iteration}")
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(16.0, 32.0, 1.05))
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = []
                for text, annotation in batch:
                    examples.append(
                        Example.from_dict(nlp.make_doc(text), annotation)
                    )
                # Updating the weights
                nlp.update(
                    examples, sgd=optimizer, drop=next(dropout), losses=losses
                )
            # evaluate after each batch
            scores = evaluate(nlp.tokenizer, textcat, val_data)

            print(f"Losses: {losses}")
            print(f"Scores: {scores}")
    scores = evaluate(nlp.tokenizer, textcat, test_data)
    print(f"Test accuracy: {scores}")
    # save the model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"model")
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(Path(model_path))
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # better to use a pre-trained model rather than a blank one
    # visit https://nightly.spacy.io/models/en for choices and languages
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="en",
        help="Input model name",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        required=False,
        default=5,
        help="No of iterations to train.",
    )
    args = parser.parse_args()
    train(args)
