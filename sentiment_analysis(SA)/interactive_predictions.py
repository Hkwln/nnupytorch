from ipywidgets import widgets
from IPython.display import display

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
# Get the input sentence from the widget
sentence = sentence_widget.value