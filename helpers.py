# Name(s):
# Netid(s):
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import numpy as np 
import math
from collections import Counter, defaultdict

def handle_unknown_words(t, documents): 
    """
    Replaces tokens in the given documents with <unk> (unknown) tokens if they occur 
    less frequently than a certain threshold, based on the provided parameter 't'. 
    Tokens are ordered first by frequency then alphabetically, so the tokens 
    replaced are the least frequent tokens and earliest alphabetically.
    
    Input:
        t (float):
            A value between 0 and 1 representing the threshold for token frequency.
            The int(t * total_unique_tokens) least frequent tokens will be replaced.
        documents (list of lists):
            A list of documents, where each document is represented as a list of tokens.
    Output:
        new_documents (list of lists):
            A list of processed documents where the int(t * total_unique_tokens) least
            frequent tokens have been replaced with <unk> tokens and no other changes. 
        vocab (list):
            A list of tokens representing the vocabulary, including both the most common tokens
            and the <unk> token.
    Example:
    t = 0.3
    documents = [["apple", "banana", "apple", "orange"],
                 ["apple", "cherry", "banana", "banana"],
                 ["cherry", "apple", "banana"]]
    new_documents, vocab = handle_unknown_words(t, documents)
    # new_documents:
    # [['apple', 'banana', 'apple', '<unk>'],
    #  ['apple', 'cherry', 'banana', 'banana'],
    #  ['cherry', 'apple', 'banana']]
    # vocab: ['banana', 'apple', 'cherry', '<unk>']
    """
    # YOUR CODE HERE
        # Flatten the list of lists into a single list of tokens
    all_tokens = [token for doc in documents for token in doc]
    
    # Count the frequency of each token
    token_counts = Counter(all_tokens)
    
    # Determine the tokens to be replaced with <unk>
    tokens_to_replace = set()
    total_tokens = len(set(all_tokens))
    
    for token, count in token_counts.items():
        if count / total_tokens < t:
            tokens_to_replace.add(token)
    
    # Replace tokens with <unk> in the documents
    new_documents = []
    for doc in documents:
        replaced_tokens = ["<unk>" if token in tokens_to_replace else token for token in doc]
        new_documents.append(replaced_tokens)
    
    vocab = set([voc for doc in new_documents for voc in doc])
    
    return new_documents, vocab

    raise NotImplementedError()


def apply_smoothing(k, observation_counts, unique_obs):
    """
    Apply add-k smoothing to state-observation counts and return the log smoothed observation 
    probabilities log[P(observation | state)].

    Input:
        k (float): 
            A float number to add to each count (the k in add-k smoothing)
            Observation here can be either an NER tag or a word, 
            depending on if you are applying_smoothing to transition_matrix or emission_matrix
        observation_counts (Dict[Tuple[str, str], float]): 
            A dictionary containing observation counts for each state.
            Keys are state-observation pairs and values are numbers of occurrences of the key.
            Keys should contain  all possible combinations of (state, observation) pairs. 
            i.e. if a `(NER tag, word)` doesn't appear in the training data, you should still include it as `observation_counts[(NER tag, word)]=0`
        unique_obs (List[str]):
            A list of string containing all the unique observation in the dataset. 
            If you are applying smoothing to the transition matrix, unique_obs contains all the possible NER tags in the dataset.
            If you are applying smoothing to the emission matrix, unique_obs contains the vocabulary in the dataset

    Output:
        Dict<key Tuple[String, String]: value Float>
            A dictionary containing log smoothed observation **probabilities** for each state.
            Keys are state-observation pairs and values are the log smoothed 
            probability of occurrences of the key.
            The output should be the same size as observation_counts.

    Note that the function will be applied to both transition_matrix and emission_matrix. 
    """
    # YOUR CODE HERE 
    expected_log_smoothed_probs = {}

    for label in set(label for label, _ in observation_counts.keys()):
        # Filter observation counts for the current label
        label_observation_counts = {obs: count for (l, obs), count in observation_counts.items() if l == label}

        # Calculate the total count of observations for the current label
        total_label_observations = sum(label_observation_counts.values())

        for obs in unique_obs:
            count_obs = label_observation_counts.get(obs, 0)

            # Calculate the smoothed probability using Laplace smoothing
            smoothed_prob = (count_obs + k) / (total_label_observations + (k * len(unique_obs)))

            # Calculate the logarithm of the smoothed probability
            log_smoothed_prob = np.log(smoothed_prob)

            expected_log_smoothed_probs[(label, obs)] = log_smoothed_prob

    return expected_log_smoothed_probs


    raise NotImplemented()
