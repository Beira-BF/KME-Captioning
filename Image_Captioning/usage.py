import tensorflow as tf
from tensorflow.keras.models import load_model
import efficientnet.tfkeras as efn
from nami.AI.kme_tokenize import Tokenizer
import json

def config_device():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
      tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
      # Visible devices must be set before GPUs have been initialized
      print(e)

def get_tokenizer():
  tokenizer = Tokenizer()

  with open('tokenizer/index2word.json', 'r') as f1:
    tokenizer.index2word = json.load(f1)
    tokenizer.index2word = {int(k):v for k,v in tokenizer.index2word.items()}
  with open('tokenizer/longest_sentences.json', 'r') as f2:
    tokenizer.longest_sentences = json.load(f2)

  with open('tokenizer/num_words.json', 'r') as f3:
    tokenizer.num_words = json.load(f3)

  with open('tokenizer/word2count.json', 'r') as f4:
    tokenizer.word2count = json.load(f4)

  with open('tokenizer/word2index.json', 'r') as f5:
    tokenizer.word2index = json.load(f5)

  return tokenizer

def get_model(mode: str):
  # if mode == 'bahdanau'
  encoder = load_model('./Bahdanau_model/Encoder_model.h5', compile=False)
  decoder = load_model('./Bahdanau_model/Decoder_model.h5', compile=False)
  
  
  return encoder, decoder

def preprocess_input(image):
  image = efn.preprocess_input(image)
  image = tf.cast( tf.reshape(1, 300, 300, 3), tf.float32 )
  return  image

def reset_state(batch_size):
  return tf.zeros((batch_size, 512))

def caption_image(image, max_length = 64):
  attention_plot = np.zeros((max_length, 100))

  hidden_state = reset_state(batch_size = 1)
  features = encoder( image, training=False )
  dec_input = tf.expand_dims([tokenizer.word2index['<start>']], 0)
  result = []

  for i in range(max_length): 
    predictions, hidden_state, attention_weights = decoder([features, dec_input, hidden_state], training=False)
    attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
    predicted_id = np.argmax(predictions, axis=-1)[0]
    result.append(tokenizer.index2word[predicted_id])

    if tokenizer.index2word[predicted_id] == '<end>':
      return  result, attention_plot

    dec_input = tf.expand_dims([predicted_id], 0)

  attention_plot = attention_plot[:len(result), :]
  return  result, attention_plot

if __name__ == "__main__":
  config_device()
  encoder, decoder = get_model(mode = 'bahdanau')
  tokenizer = get_tokenizer()
  image_pre = preprocess_input(image)
  
  result, _ = caption_image(image_pre)
  print(''.join(result))
  plt.imshow( image )
  plt.show()

