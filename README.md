# YSC21-AI_KME
AI for generate IUPAC "Organic Compound" name from bond-line structural picture

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ltxA8A_e9ulmUyrG9iPwxBVMcqPH3XGy?usp=sharing/)


___

## Prepare training dataset
#### What to have
1. CHAR_INDICES: dictionary with key is character [string], value is number [int] *(use to preprocess text to number)*

2. Dict_cut: Input text with determined (by '|' ) where to be cut (use for create label)
```
Cyclo|prop|ane| |Non|a|-|1|,|8|-|di|yne| 
|1|,|3|-|di|chlorocyclo|hex|ane| |Hept|a|-|1|,|5|-|di|ene
```

3. Dict: Input text (raw) (use for train model)
```
Cyclopropane Nona-1,8-diyne 1,3-dichlorocyclohexane Hepta-1,5-diene
```

## Create dataset
1. Make JSON value as array of chemical name (dataset, dataset_cut)
2. Split array for **training dataset (90%)** (dataset_train, dataset_cut_train) and **validation dataset (10%)** (dataset_val, dataset_cut_val) 
3. Join item in each array together into text
4. Create dataset using **create_dataset function** that **take dataset_cut** then return **X_train**  (size: [text_length, look_back]) (dataset_cut that have been cut '|') and **label** (position where to cut 1 = cut, 0 = not cut)
5. Use tf.data.Dataset.from_tensor_slices((X, y)).batch_size(128) to make data easy to be train

## Create Model
1. We use 1x**Embedding** layer, 1x**Bidirection LSTM** layer, **Dense** Layer
2. Compiled **model optimizer = Adam**, **loss_function = Categorical Crossentropy** (becase we classify 2 label output 1 = cut, 0 = not cut) **call_back = [EarlyStopping, ModelCheckpoint]**
*Early stopping : Stop train model if validation_loss is being increase*
*ModelCheckpoint : Save model that has minimum validation_loss*

## After Train Model
- The output of the model is array (size: [batch_size, 2] determined which position to be cut (value = 0 -> not cut ; 1 -> cut))
```
[1 1 1 1 1 0 1 1 0 0 0 0 1 1 1 1 1 0 0 0 1 0 1 0 0 0 1 0 0]
```
- Tokenize dataset (text which **didn't** determined where to be cut) with label (output from model) using **word_tokenize function** that return **array of text that has been cut**
``` 
['1', ',', '2', '-', 'di', 'h', 'ydrox', 'y', '-', '2', '-', 'meth', 'yl', 'prop', 'ane']
```