import os

dataset_folder = os.path.join(os.getcwd(), 'dataset')
data = os.listdir(dataset_folder)

# Filter out only text documents
text_file = [file for file in data if file.endswith(".txt")]

all_text = ""
for file_name in text_file:
    with open(os.path.join(dataset_folder, file_name), 'r', encoding='CP1252') as file:
        all_text += file.read()

# Split the text into words and filter out words shorter than 3 characters
words = all_text.split()
filtered_words = [word for word in words if len(word) > 3]

# Count occurrences of each word without using additional libraries
word_counts = {}
for word in filtered_words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

# Sort the words by count in descending order
sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

# Display the top five used words
print("Top five used words:")
for i in range(min(5, len(sorted_word_counts))):
    word, count = sorted_word_counts[i]
    print(f"{word}: {count}")





