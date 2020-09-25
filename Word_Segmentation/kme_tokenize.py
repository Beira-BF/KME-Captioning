import tensorflow as tf
import json
import numpy as np


with open('files/vocab.json', 'r') as f:
  CHAR_INDICES = json.load(f)

look_back = 10

file_path='save_models/best_model.hdf5'
best_model = tf.keras.models.load_model(file_path)

def preprocessing_text(raw_text):
    """
    take unseen (testing) text and encode it with CHAR_DICT
    //It's like create_dataset() but not return label
    return preprocessed text
    """
    X = []
    data = [CHAR_INDICES['<pad>']] * look_back
    for char in raw_text:
        char = char if char in CHAR_INDICES else '<unk>'  # check char in dictionary
        data = data[1:] + [CHAR_INDICES[char]]  # X data
        X.append(data)
    return np.array(X)

def predict(preprocessed_text):
    pred = best_model.predict(preprocessed_text)
    class_ = tf.argmax(pred, axis=-1).numpy()
    
    return class_

def word_tokenize(text):
    preprocessed_text = preprocessing_text(text)
    class_ = predict(preprocessed_text)
    class_ = np.append(class_, 1)

    cut_indexs = [i for i, value in enumerate(class_) if value == 1]
    words = [text[cut_indexs[i]:cut_indexs[i+1]] for i in range(len(cut_indexs)-1)]
    
    join_word = '|'.join(words)
    
    return words, join_word