import tensorflow as tf
import numpy as np
import os
import requests

class Segmentation():
    def __init__(self):
        url = 'https://firebasestorage.googleapis.com/v0/b/ysc-kme-25095.appspot.com/o/best_model.hdf5?alt=media&token=124965a7-56ae-43c8-ac90-b99220dde13d'
        r = requests.get(url, allow_redirects=True)
        with open('my_model.hdf5', 'wb') as f_hdf5, requests.Session() as req:
            f_hdf5.write(r.content)

            resp = req.get('https://firebasestorage.googleapis.com/v0/b/ysc-kme-25095.appspot.com/o/vocab.json?alt=media&token=34178ef2-9e62-491f-9fc5-bf2e30a33635')
            self.CHAR_INDICES = resp.json()
        
        self.model = tf.keras.models.load_model('my_model.hdf5')
        os.remove('my_model.hdf5')            
        
        self.look_back = 5
        
    def create_dataset(self, text):
        """
        take text with label (text that being defined where to cut ('|')) 
        and encode text and make label
        return preprocessed text & preprocessed label
        """
        X, y = [], []
        text = '|' + text
        data = [self.CHAR_INDICES['<pad>']] * self.look_back
        for i in range(1, len(text)):
            current_char = text[i]
            before_char = text[i-1]

            if current_char == '|':
                continue
            data = data[1:] + [self.CHAR_INDICES[current_char]]  # X data

            target = 1 if before_char == '|' else 0  # y data
            X.append(data)
            y.append(target)
        
        return np.array(X), tf.one_hot(y, 2)
        
    def preprocessing_text(self, raw_text):
        """
        take unseen (testing) text and encode it with CHAR_DICT
        //It's like create_dataset() but not return label
        return preprocessed text
        """
        X = []
        data = [self.CHAR_INDICES['<pad>']] * self.look_back
        for char in raw_text:
            char = char if char in self.CHAR_INDICES else '<unk>'  # check char in dictionary
            data = data[1:] + [self.CHAR_INDICES[char]]  # X data
            X.append(data)
        return np.array(X)

    def predict(self, preprocessed_text):
        pred = self.model.predict(preprocessed_text)
        class_ = tf.argmax(pred, axis=-1).numpy()
        
        return class_

    def word_segmentation(self, text):
        preprocessed_text = self.preprocessing_text(text)
        class_ = self.predict(preprocessed_text)
        class_[0] = 1
        class_ = np.append(class_, 1)

        cut_indexs = [i for i, value in enumerate(class_) if value == 1]
        words = [text[cut_indexs[i]:cut_indexs[i+1]] for i in range(len(cut_indexs)-1)]
        
        join_word = '|'.join(words)
        
        return words, join_word

class Tokenizer:
    def __init__(self):
        self.word2index = {"<pad>": 0, "<start>": 1,"<end>": 2, "<unk>": 3}
        self.word2count = {"<pad>": 0, "<start>": 0,"<end>": 0, "<unk>": 0}
        self.index2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentences = 0
        self.kme_segment = Segmentation()
    
    def add_word(self, word):
        if word not in self.word2index:

            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:

            # Word exists; increase word count
            self.word2count[word] += 1
    
    def fit_on_text(self, sentences):
        for sentence in sentences:

            # Model predict 
            segmented_text, _ = self.kme_segment.word_segmentation(sentence)

            # Add <start> at start and <end> at end of each sentences
            segmented_text = np.concatenate((['<start>'], segmented_text, ['<end>']))

            for word in segmented_text:
                self.add_word(word)       
    
    def text_to_sequences(self, sentences, method_pad=None):
        sequences_arr = []
        for sentence in sentences:

            # Model predict 
            segmented_text, _ = self.kme_segment.word_segmentation(sentence)
            tokenize_text_arr = []

            for word in segmented_text:
                try:
                    tokenize_text_arr.append(self.word2index[word])
                except KeyError:

                    # use <unk> key for word that never met 
                    tokenize_text_arr.append(self.word2index['<unk>'])

            sequences_arr.append(tokenize_text_arr)
        if method_pad:
            # Make zero padding by Maximum length of sentence
            sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences_arr, padding=method_pad)
            return sequences_padded
        return sequences_arr
    
    def sequences_to_text(self, sequences):
        texts_arr = []
        for sequence in sequences:
            index_arr = []
            for index in sequence:
                index_arr.append(self.index2word[index])
            texts_arr.append(index_arr)
        return texts_arr
                