from sortAlgorithm import filtered_words, all_text

#Given a list of words:
#Filter Select only the numbers that start with a L
#Map: encrypt the chars to a number and sqare all even ones, decrypt later
#Sort: sort the squared numbers
# use filter() and map() sorted() with lambda 
chars = sorted(list(set(all_text)))

words_that_start_with_L = list(set(filter(lambda word: word.startswith("L"), filtered_words)))

print(words_that_start_with_L)


char_to_int = {chars:i for i ,chars in enumerate(chars)}
int_to_char = {i: chars for chars, i in enumerate(chars)}
encode = lambda s:[char_to_int[c] for c in s]
decode = lambda a: ''.join([int_to_char[d] for d in a])
numbers = encode(chars)

gerade_nummer = [numbers for numbers in numbers if numbers %2 ==0]

filtered_numbers = list(map(lambda i: i ** i, filter(lambda number: number % 2 == 0, numbers)))
print(f"here are the encoded chars:",numbers)
print(f"here are only even numbers",filtered_numbers)