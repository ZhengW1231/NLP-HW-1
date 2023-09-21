# Name(s):
# Netid(s):
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
from collections import defaultdict
from nltk import classify
from nltk import download
from nltk import pos_tag
import numpy as np
from collections import Counter
from nltk.classify.maxent import MaxentClassifier
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class HMM: 

  def __init__(self, documents, labels, vocab, all_tags, k_t, k_e, k_s, smoothing_func): 
    """
    Initializes HMM based on the following properties.

    Input:
      documents: List[List[String]], dataset of sentences to train model
      labels: List[List[String]], NER labels corresponding the sentences to train model
      vocab: List[String], dataset vocabulary
      all_tags: List[String], all possible NER tags 
      k_t: Float, add-k parameter to smooth transition probabilities
      k_e: Float, add-k parameter to smooth emission probabilities
      k_s: Float, add-k parameter to smooth starting state probabilities
      smoothing_func: (Float, Dict<key Tuple[String, String] : value Float>, List[str]) -> 
      Dict<key Tuple[String, String] : value Float>
    """
    self.documents = documents
    self.labels = labels
    self.vocab = vocab
    self.all_tags = all_tags
    self.k_t = k_t
    self.k_e = k_e
    self.k_s = k_s
    self.smoothing_func = smoothing_func
    self.emission_matrix = self.build_emission_matrix()
    self.transition_matrix = self.build_transition_matrix()
    self.start_state_probs = self.get_start_state_probs()


  def build_transition_matrix(self):
    """
    Returns the transition probabilities as a dictionary mapping all possible
    (tag_{i-1}, tag_i) tuple pairs to their corresponding smoothed 
    log probabilities: log[P(tag_i | tag_{i-1})]. 
    
    Note: Consider all possible tags. This consists of everything in 'all_tags', but also 'qf' our end token.
    Use the `smoothing_func` and `k_t` fields to perform smoothing.

    Output: 
      transition_matrix: Dict<key Tuple[String, String] : value Float>
    """
    # YOUR CODE HERE
    transition_matrix = {}
    unique_tags = set(self.all_tags + ["qf"])  # Include 'qf' for the end token
    # Count transition occurrences
    transition_counts = Counter()
    for labels in self.labels:
        for i in range(len(labels) - 1):
            transition_counts[(labels[i], labels[i + 1])] += 1
    # Calculate smoothed transition probabilities using apply_smoothing
    smoothed_transition_probs = self.smoothing_func(self.k_t, transition_counts, unique_tags)
    # Fill the transition_matrix with smoothed probabilities
    for tag1 in unique_tags:
        for tag2 in unique_tags:
            transition_matrix[(tag1, tag2)] = smoothed_transition_probs.get((tag1, tag2), float('-inf'))
    return transition_matrix
    raise NotImplementedError()


  def build_emission_matrix(self): 
    """
    Returns the emission probabilities as a dictionary, mapping all possible 
    (tag, token) tuple pairs to their corresponding smoothed log probabilities: 
    log[P(token | tag)]. 
    
    Note: Consider all possible tokens from the list `vocab` and all tags from 
    the list `all_tags`. Use the `smoothing_func` and `k_e` fields to perform smoothing.
  
    Output:
      emission_matrix: Dict<key Tuple[String, String] : value Float>
      Its size should be len(vocab) * len(all_tags).
    """
    # YOUR CODE HERE
    emission_matrix = {}

    # Count emission occurrences
    emission_counts = Counter()
    for labels, tokens in zip(self.labels, self.documents):
        for label, token in zip(labels, tokens):
            emission_counts[(label, token)] += 1

    # Calculate smoothed emission probabilities using apply_smoothing
    smoothed_emission_probs = self.smoothing_func(self.k_e, emission_counts, self.vocab)

    # Fill the emission_matrix with smoothed probabilities
    for tag in self.all_tags:
        for token in self.vocab:
            emission_matrix[(tag, token)] = smoothed_emission_probs.get((tag, token), float('-inf'))

    return emission_matrix
    raise NotImplementedError()


  def get_start_state_probs(self):
    """
    Returns the starting state probabilities as a dictionary, mapping all possible 
    tags to their corresponding smoothed log probabilities. Use `k_s` smoothing
    parameter to manually perform smoothing.
    
    Note: Do NOT use the `smoothing_func` function within this method since 
    `smoothing_func` is designed to smooth state-observation counts. Manually
    implement smoothing here.

    Output:
      start_state_probs: Dict<key String : value Float>
    """
    # YOUR CODE HERE 
    start_state_probs = {}

    # Count starting state occurrences
    start_counts = Counter(label[0] for label in self.labels)

    # Calculate smoothed starting state probabilities manually
    total_start_count = len(self.labels)
    for tag in self.all_tags:
        count = start_counts.get(tag[0], 0)
        smoothed_prob = (count + self.k_s) / (total_start_count + (self.k_s * len(self.all_tags)))
        start_state_probs[tag] = np.log(smoothed_prob)

    return start_state_probs
    raise NotImplementedError()


  def get_trellis_arc(self, predicted_tag, previous_tag, document, i): 
    """
    Returns the trellis arc used by the Viterbi algorithm for the label 
    `predicted_tag` conditioned on the `previous_tag` and `document` at index `i`.
    
    For HMM, this would be the sum of the smoothed log emission probabilities and 
    log transition probabilities: 
    log[P(predicted_tag | previous_tag))] + log[P(document[i] | predicted_tag)].
    
    Note: Treat unseen tokens as an <unk> token.
    Note: Make sure to handle the case where we are dealing with the first word. Is there a transition probability for this case?
    Note: Make sure to handle the case where the predicted tag is an end token. Is there an emission probability for this case?
  
    Input: 
      predicted_tag: String, predicted tag for token at index `i` in `document`
      previous_tag: String, previous tag for token at index `i` - 1
      document: List[String]
      i: Int, index of the `document` to compute probabilities 
    Output: 
      result: Float
    """
    # YOUR CODE HERE 
    if i == 0:
        # Handle the first word, where there is no transition probability
        transition_prob = 0.0
    else:
        # Calculate the log transition probability from previous_tag to predicted_tag
        transition_prob = self.transition_matrix.get((previous_tag, predicted_tag), float('-inf'))

    # Handle unseen tokens as <unk> token
    token = document[i] if document[i] in self.vocab else "<unk>"

    # Calculate the log emission probability for the token and predicted_tag
    emission_prob = self.emission_matrix.get((predicted_tag, token), float('-inf'))

    # Calculate the trellis arc, which is the sum of transition and emission probabilities
    result = transition_prob + emission_prob

    return result
    raise NotImplementedError()

 

################################################################################
################################################################################



class MEMM: 

  def __init__(self, documents, labels): 
    """
    Initializes MEMM based on the following properties.

    Input:
      documents: List[List[String]], dataset of sentences to train model
      labels: List[List[String]], NER labels corresponding the sentences to train model
    """
    self.documents = documents
    self.labels = labels
    self.classifier = self.generate_classifier()


  def extract_features_token(self, document, i, previous_tag):
    """
    Returns a feature dictionary for the token at document[i].

    Input: 
      document: List[String], representing the document at hand
      i: Int, representing the index of the token of interest
      previous_tag: string, previous tag for token at index `i` - 1

    Output: 
      features_dict: Dict<key String: value Any>, Dictionaries of features 
                    (e.g: {'Is_CAP':'True', . . .})
    """
    features_dict = {}

    # YOUR CODE HERE 
    ### TODO: ADD FEATURES
    # Extract features for the current token
    word = document[i]
    prev_word = document[i - 1] if i > 0 else ''
    next_word = document[i + 1] if i < len(document) - 1 else ''

    # features :
    # 'Part of Speech' 
    # features_dict['part_of_speech'] = nltk.pos_tag([word])[0][1]

    # 'A certain character (e.g: '.') in the token' 
    # features_dict['contains_period'] = '.' in word

    # 'Number of vowels' 
    # vowels = ['a', 'e', 'i', 'o', 'u']
    # features_dict['num_vowels'] = sum(1 for char in word if char.lower() in vowels)

    # 'Previous Tag' 
    features_dict['previous_tag'] = previous_tag

    # 'Current Word' 
    features_dict['current_word'] = word

    # 'Previous Word' 
    features_dict['previous_word'] = prev_word

    # 'Verb' 
    features_dict['is_verb'] = True if nltk.pos_tag([word])[0][1].startswith('V') else False

    # 'Capitalization' 
    features_dict['is_capitalized'] = word[0].isupper()

    # 'Length of current token' 
    # features_dict['token_length'] = len(word)

    # 'Special characters (e.g: digits)' 
    features_dict['has_special_char'] = any(char.isdigit() for char in word)

    # 'Token frequency in the whole set of documents or current document' 
    # features_dict['token_frequency'] = document.count(word)

    return features_dict
    raise NotImplementedError()


  def generate_classifier(self):
    """
    Returns a trained MaxEnt classifier for the MEMM model on the featurized tokens.
    Use `extract_features_token` to extract features per token.

    Output: 
      classifier: nltk.classify.maxent.MaxentClassifier 
    """
    # YOUR CODE HERE 
    featuresets = []
    for i, document in enumerate(self.documents):
            previous_tag = 'O'  # Initialize previous tag as 'O' for the first token
            for j in range(len(document)):
                features = self.extract_features_token(document, j, previous_tag)
                featuresets.append((features, self.labels[i][j]))
                previous_tag = self.labels[i][j]  # Update previous tag

        # Train the MaxEnt classifier
    classifier = MaxentClassifier.train(featuresets, trace=0, max_iter=10)

    return classifier
    raise NotImplementedError()


  def get_trellis_arc(self, predicted_tag, previous_tag, document, i): 
    """
    Returns the trellis arc used by the Viterbi algorithm for the label 
    `predicted_tag` conditioned on the features of the token of `document` at 
    index `i`.
    
    For MEMM, this would be the log classifier output log[P(predicted_tag | features_i)].
  
    Input: 
      predicted_tag: string, predicted tag for token at index `i` in `document`
      previous_tag: string, previous tag for token at index `i` - 1
      document: string
      i: index of the `document` to compute probabilities 
    Output: 
      result: Float
    """
    # YOUR CODE HERE 
    features = self.extract_features_token(document, i, previous_tag)

    # Calculate the log probability of the predicted tag given the features
    log_prob = self.classifier.prob_classify(features).logprob(predicted_tag)
    return log_prob
    raise NotImplementedError()
