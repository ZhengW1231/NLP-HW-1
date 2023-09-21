# Name(s):
# Netid(s):
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import numpy as np

def viterbi(model, observation, tags):
  """
  Returns the model's predicted tag sequence for a particular observation.
  Use `get_trellis_arc` method to obtain model scores at each iteration.

  Input: 
    model: HMM/MEMM model
    observation: List[String]
    tags: List[String]
  Output:
    predictions: List[String]
  """
  # YOUR CODE HERE 
  n = len(observation)  # Length of the observation
  num_tags = len(tags)  # Number of possible tags

  # Initialize the Viterbi trellis and backpointer matrices
  trellis = np.zeros((n, num_tags))
  backpointers = np.zeros((n, num_tags), dtype=int)

  # Initialize the first column of the trellis with the start state probabilities
  # for i in range(num_tags):
  #     trellis[0][i] = model.start_state_probs[tags[i]] + model.get_trellis_arc(tags[i], '<s>', observation, 0)

  for i in range(num_tags):
    trellis[0][i] = model.get_trellis_arc(tags[i], '<s>', observation, 0)

  # Fill in the trellis and backpointers
  for t in range(1, n):
      for j in range(num_tags):
          max_prob = float('-inf')
          best_prev_tag = None
          for i in range(num_tags):
              score = trellis[t - 1][i] + model.get_trellis_arc(tags[j], tags[i], observation, t)
              if score > max_prob:
                  max_prob = score
                  best_prev_tag = i
          trellis[t][j] = max_prob
          backpointers[t][j] = best_prev_tag

  # Find the best sequence of tags using backpointers
  best_last_tag = np.argmax(trellis[n - 1])
  best_tag_sequence = [tags[best_last_tag]]

  for t in range(n - 1, 0, -1):
      best_last_tag = backpointers[t][best_last_tag]
      best_tag_sequence.insert(0, tags[best_last_tag])

  return best_tag_sequence

  raise NotImplementedError()