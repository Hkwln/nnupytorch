from ipywidgets import widgets
from IPython.display import display
from SA import model2, map_text_to_indices, path, prepare_dataset
import torch
from datasets import Features, Value, ClassLabel, Dataset

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
    for bad,good in probs:
        if bad <0.5:
            print("positive")
        elif bad >0.5:
            print("negative")
        else: print("neutral")

    return probs

# change the while True function into a a function with recursion
def interactive_part(sentence = None, label = None):
    if sentence is None or label is None:
        sentence = []
        label = []
    ipsentence = input("give me an input sentence:")
    
    if ipsentence == ("info"):
        print(f"here you can classify your sentence, you can quit by typing Ende or type quit \n")
    if not ipsentence == ("Ende") or  ipsentence == ("quit"):
        #validate the input sentence
        if not ipsentence.strip():
            raise ValueError("Inputsentence cannot be empty.")
        if not ipsentence == ("info"):
            
            predictions = predict_sentence(ipsentence) 
            answer = display(get_sentiment(predictions))
            while True:
                correction = input("what was the correct answer?p/n/nope:")
                if correction == ("p"):
                    iplabel = 1
                    break
                elif correction == ("n"):
                    iplabel = 0
                    break
                elif correction  == ("nope"):
                    break
                #here we add the current sentence and the label both to an array
                label.append(iplabel)
                sentence.append(ipsentence)
        return interactive_part(sentence, label)
    #termination condition
    elif ipsentence in ("Ende", "quit"):
    #now i want to save the feautures:(idx, sentence and labels) num:(rows) 
    #this should be the same data structure as the data loaded from the hugginface_hub
        
        features = Features({
            "idx": Value("int32"),
            "sentence": Value("string"),
            "label": ClassLabel(names=["negative","positive"])
        })
        data = {
            "idx": list(range(len(sentence))),
            "sentence": sentence,
            "label": label
        }
        dataset = Dataset.from_dict(data, features=features)
        return dataset
    else:
        raise ValueError("This should not happen in any way, how did you do that?")
#now i want to add my newly self created dataset to the old dataset
#step 1: preprocessing the dataset
#step 2: adding the additional dataset to the preprocessed_data.pt
def update_preprocessed_data(new_dataset):
    #get the preprocessed data
    preprocessed = torch.load(path, weights_only= True)
    #prepare the new data
    new_data = prepare_dataset(new_dataset)
    #add the new data to the old preprocessed
    preprocessed["dataset_train_tokenized"].extend(new_data)
    torch.save(preprocessed, path)

#usage:

dataset = interactive_part()
print("final dataset:", dataset)
update_preprocessed_data(dataset)
print("the preprocessed_dataset has been updated")