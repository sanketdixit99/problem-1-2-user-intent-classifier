data:

- The dataset used for training is generated using chatGPT for demonstration
- It contains 600 query and label pairs
- It has 4 unique labels ("VIDEO", "SEGMENT", "CLIP", "FRAME")
- There are 150 samples corresponding to each label


architecture:

- pretrained distilbert-base-uncased is used as text encoder
- The classification head is added to the cls token (1st token)
- The classification head uses softmax activation at the end to convert the logits to probability distribution on the four labels.
