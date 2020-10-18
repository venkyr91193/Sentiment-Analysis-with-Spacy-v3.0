import spacy
from spacy.util import compounding, decaying, minibatch
from spacy.language import Language
from spacy.training import Example
import random

@Language.factory("my_component")
def create_my_component(nlp, name):
    return MyComponent(nlp)

class MyComponent:
    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, doc):
        return doc

# more lower the dropout, the more overfitting
# dropout
dropout = decaying(0.5, 1e-4)

nlp = spacy.blank('en')
nlp.add_pipe("textcat")
# true by default
# config={"exclusive_classes": True}

examples = []
data = []
for ex in data:
   examples.append(Example.from_dict(nlp.make_doc(ex[0]),ex[1]))

# the next line or add the labels yourself
nlp.initialize(lambda: examples)

# adding labels custom without examples
nlp.get_pipe("textcat").add_label("some_labels")

nlp.begin_training()

# some train data
TRAIN_DATA = []

for i in range(20):
    random.shuffle(TRAIN_DATA)
    for batch in minibatch(TRAIN_DATA):
        examples = []
        for text, annots in batch:
            examples.append(Example.from_dict(nlp.make_doc(text), annots))
        nlp.update(examples)

pass