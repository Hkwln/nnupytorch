## steps to make a working opical character recognizer:
- image thresholding
- countour formation
- detecting individual characters and forming a Region of image (roi) around detected characters
- passing the roi though our trained model and finding out the predicted character
- saving/appending this character as a string, which is basically your output.


## plan:
- build a cnn which recognizes every character
- make inputing more than one char possible so that it can translate simple senteces correctly
- build a neural network that basically works like a spell checker
  - spell checker more sure than the cnn, change the char
- make a detection of chars  and set a roi for to make the project complete
