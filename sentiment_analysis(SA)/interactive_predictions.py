from ipywidgets import widgets
from IPython.display import display
from SA import model2, map_text_to_indices
import torch


#here we change the char input sentence into tokens and finaly into a torch tensor and feed into the model
#input: sentence in char
#output: tensor (for example: tensor([[ 0.5330, -0.3757]])) 
#        which represents the sentiment of the input sentence
def predict_sentence(sentence:str):
    token_ids = map_text_to_indices(sentence)
    token_ids = torch.tensor([token_ids],dtype=torch.long)
    with torch.no_grad():
        predictions = model2(token_ids)
    return predictions

#here we define a function where we have as a input a tensor like tensor([[ 0.5330, -0.3757]]), which represents the sentiment of the input sentence; 
#input: tensor (for example: tensor([[ 0.5330, -0.3757]]))
#output:translate the tensor into positive, negative, neutral / probability
def get_sentiment(tensor):
    #softmax function:
    #compute exponentials of the logits
    exp = torch.exp(tensor)
    #compute the sum of the logits to normalize
    total = exp.sum()
    #normalize the exponentials to get the probabilities
    probs = exp / total
    #define if the text is positiv, negativ or neutral
    #positiv: [[under 50%][over 50%]]
    #negative: [[over 50%][under 50%]]
    #neutral:  both under or both over 50%
    for bad,good in probs:
        if bad <0.4:
            print("positive")
        elif bad >0.6:
            print("negative")
        else: print("neutral")

    return probs


# change the while True function into a a function with recursion
def interactive_part():
    sentence = input("give me an input sentence:")
    if sentence == ("info"):
        print(f"here you can classify your sentence, you can quit by typing Ende or type quit \n")
    if not sentence == ("Ende") or  sentence == ("quit"):
        #validate the input sentence
        if not sentence.strip():
            raise ValueError("Inputsentence cannot be empty.")
        if not sentence == ("info"):
            predictions = predict_sentence(sentence)
            answer = get_sentiment(predictions)
            display(answer)
            label = input("is the prediction correct?y/n:")
            if label == "y":
                
                if display(answer)=="positive":
                    label = 0
                else: label = 1
            elif label == "n":
                
                if display(answer) =="positive":
                    label = 1
                else: label = 0
            else: 
                raise ValueError("You have to type y/n and not some other bullshit")
        interactive_part()


