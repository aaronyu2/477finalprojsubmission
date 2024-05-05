# important: run this file from the root directory
import prompts
from prompts import PatentData
import random
from PIL import Image as PILImage

class Augmentation:
  def __call__(self, data: PatentData):
    return data.text

class StringAugmentation(Augmentation):
  def __call__(self, data:PatentData):
    return data.text

class StringRandomWordDeletion(StringAugmentation):
  # check for double spaces, but shouldn't really matter
  deletion_probability: int
  def __call__(self, data:PatentData):
    lines = data.text.split("\n")
    res = []
    for line in lines:
      words = line.split()
      for i in range(len(words)):
        if random.random() < self.deletion_probability:
          words[i] = ''
      res.append(' '.join(words))
    return '\n'.join(res)

class EasyDataAugmentation(StringAugmentation):
  # CITATION: https://arxiv.org/pdf/1901.11196
  alpha_deletion: int
  stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']
  def synonym_replacement_in_one(words, n):
    stop_words = set(stopwords.words('english'))
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = set()
        for syn in wordnet.synsets(random_word): 
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if random_word in synonyms:
            synonyms.remove(random_word)
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: 
            break
    
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

  def contains_nonalphanumeric_character(word):
    for c in word:
      if not c.isalnum():
        return True
    return False
  
  def random_deletion(self, words):
    if len(words) == 1: return words
    new_words = []
    for word in words:
      if random.uniform(0, 1) > p: new_words.append(word)
        
    if len(new_words) == 0:
      rand_int = random.randint(0, len(words)-1)
      return [words[rand_int]]
    return new_words
  
  def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
      random_idx_1 = random.randint(0, len(new_words)-1)
      random_idx_2 = random_idx_1
      attempts = 0
      while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        attempts += 1
        if attempts > 2: break
      if attempts <= 2:
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words
  
  def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
      rand_int = random.randint(0, len(new_words)-1)
      new_words.insert(rand_int, words[rand_int])
    return new_words
  
  def __call__(self, data:PatentData):
    lines = data.text.split("\n")
    res = []
    for line in lines:
      words = line.split()
      new_words = self.random_deletion(words)
      new_words = self.random_swap(new_words, 1)
      new_words = self.random_insertion(new_words, 1)
      res.append(' '.join(new_words))
    return '\n'.join(res)

class ImageAugmentation(Augmentation):
  def __call__(self, data:PatentData):
    return data.image
  
class RandomImageCrop(ImageAugmentation):
  def __call__(self, data:PatentData):
    image = data.image
    width, height = image.size
    x = random.randint(0, width//2)
    y = random.randint(0, height//2)
    return image.crop((x, y, width, height))
