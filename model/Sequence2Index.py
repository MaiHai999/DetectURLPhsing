
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class Sequence2Index:
    def __init__(self):
        self.lable = ["benign", "phishing", "defacement", "malware"]

    def create_character_vocabulary(self):
        # Define the character vocabulary based on the provided rules
        character_vocabulary = {}
        index_counter = 1  # Start index from 1

        # Lowercase letters
        for char in 'abcdefghijklmnopqrstuvwxyz':
            character_vocabulary[char] = index_counter
            index_counter += 1

        # Uppercase letters
        for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            character_vocabulary[char] = index_counter
            index_counter += 1

      # Digits
        for char in '0123456789':
            character_vocabulary[char] = index_counter
            index_counter += 1

        # Special characters
        special_characters = ',;.!?:”’/\\|_@#$%^&*~`+-=<>()[]{}:'
        for char in special_characters:
            character_vocabulary[char] = index_counter
            index_counter += 1

        return character_vocabulary


    def char_to_index(self,character):
        character_vocabulary = self.create_character_vocabulary()
        return character_vocabulary.get(character, 95)  # -1 for unknown characters

    def sequence_to_index(self, input_sequence , max_len):
        indices = [self.char_to_index(char) for char in input_sequence]
        padded_indices = pad_sequences([indices], maxlen=max_len, padding='post', value=0)[0]
        return np.array(padded_indices)
