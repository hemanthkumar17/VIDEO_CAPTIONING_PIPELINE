from nltk.translate import meteor
from nltk import word_tokenize
import json
import nltk
nltk.download('punkt')
nltk.download('wordnet')

def get_score(data, bard_captions):
  output = {}
  for key in bard_captions:
    input_captions = ""
    for i in data[key]['annotations']:
      input_captions = input_captions + i['sentence']+'. '
    output[key] = meteor([word_tokenize(input_captions)],word_tokenize(bard_captions[key]))
  return output