from ipywidgets import widgets
from IPython.display import display
from SA import model2, map_text_to_indices
import torch

#todo a predict for the provided sentence using our trained model
#hints:
# - you need to convert the input sentence to token ids, using the same mapping as for the training.
#   Can you re-use something to accomplish this?
# - since we only have a single input, we don't need batching nor a dataloader
# . We don't need the gradients from the model
sentence_widget = widgets.Text(
    value="This movie is terrible",
    placeholder="Type something",
    description="Sentence:",
    disabled=False,
)
display(sentence_widget)
def predict_sentence(sentence:str):
    token_ids = map_text_to_indices(sentence)
    token_ids = torch.tensor([token_ids],dtype=torch.long)
    with torch.no_grad():
        predictions = model2(sentence)
    return predictions

sentence = sentence_widget.value
predictions = predict_sentence(sentence)
display(predictions)
