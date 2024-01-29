import re
import random
import itertools
import nltk
import pytrec_eval

nltk.download("wordnet")
from nltk.corpus import wordnet


def preprocess_corpus(file_path: str) -> list:
    """
    This function creates a spelling error list from a given .txt corpus
    @param corpus_path: the path to the corpus
    @param filename: the name of the corpus file
    @return: a list containing the spelling errors and true spellings
    """

    # initialize with an empty dataframe
    arr = []

    # current correct word
    cur_correct = ""

    idx = 0
    # read file line by line
    with open(file_path, "r") as file:
        for line in file:
            # match the correct words which start with $
            if re.match(r"^\$", line):
                cur_correct = line.strip().replace("$", "")
            # add if an incorrect match found
            else:
                arr.append([line.strip(), cur_correct])
                idx += 1

    # return the dataframe
    return arr


def reduce_spelling_errors_corpus_length(df_arg, approx_size):
    letter_words_dict = {}
    for pair in df_arg:
        starting_letter = pair[0][0].lower()
        if starting_letter not in letter_words_dict and starting_letter.isalpha():
            letter_words_dict[starting_letter] = []
        if starting_letter.isalpha():
            letter_words_dict[starting_letter].append(pair)

    words_per_letter = approx_size // len(letter_words_dict)

    reduced_df = []
    for letter, words in letter_words_dict.items():
        chosen_words = random.sample(words, min(words_per_letter, len(words)))
        reduced_df.extend(chosen_words)

    return reduced_df


def calculate_med(str1: str, str2: str) -> int:
    """
    This function calculates the minimum edit distance between two strings, uses cost two for substitution, also called as the Levenshtein distance
    @param str1: the first string (source)
    @param str2: the second string (destination)
    @return: the minimum edit distance between the two strings
    """
    # length of source string
    n = len(str1)

    # length of destination string
    m = len(str2)

    # dynamic programming matrix
    dp = [[0 for x in range(m + 1)] for x in range(n + 1)]

    # initializing the rows and columns of the dp matrix
    for i in range(m + 1):
        dp[0][i] = i

    for j in range(n + 1):
        dp[j][0] = j

    # calculate the minimum edit distance iteratively
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # condition where equal character, no substitution cost
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = min(
                    (dp[i - 1][j] + 1), (dp[i - 1][j - 1]), (dp[i][j - 1] + 1)
                )

            # condition where characters arent equal
            else:
                dp[i][j] = min(
                    (dp[i - 1][j] + 1), (dp[i - 1][j - 1] + 2), (dp[i][j - 1] + 1)
                )

    # return the minimum edit distance
    return dp[n][m]


# function to get the top k words according to the mde algorithm
def get_top_k_words(training_example: list) -> list:
    """
    This function returns the top k words according to the med algorithm

    @param training_example: a single row of the list
    @return: the top k words
    """
    # get al the words in the wordnet dictionary
    words = wordnet.words()
    # convert the incorrect word to a lower case letter for consistency
    incorrect = training_example[0].lower()
    # convert the correct word to a lower case for more consistency
    correct = training_example[1].lower()
    # create a dictionary for results
    results = {}
    # final dictionary containing values
    final_dict = {"cor": correct, "inc": incorrect}
    # loop through the dictionary
    for word in words:
        # calculate the minimum edit distance for the current word in the dictionary
        med = calculate_med(incorrect, word)
        # add to the results dictionary
        results[word] = med

    # sort the results dictionary based on the edit distance
    results = dict(sorted(results.items(), key=lambda w: w[1]))

    # get top 1, 5 and 10 values
    k_vals = [1, 5, 10]

    for val in k_vals:
        # get a slice of the first k word values
        w_temp = dict(itertools.islice(results.items(), val))

        # list to store the top k words
        top_k = list()

        for w, m in w_temp.items():
            top_k.append(w)

        final_dict["k = " + str(val)] = top_k

    return final_dict


def s_at_K_for_every_incorrect_token(final_results):
    temp_arr = []
    final_dict = {}
    for result in final_results:
        correct_w = result.get("cor")
        incorrect_w = result.get("inc")
        k1 = result.get("k = 1")
        k5 = result.get("k = 5")
        k10 = result.get("k = 10")
        if k1 and k5 and k10:
            temp_arr.append(1 if correct_w in k1 else 0)
            temp_arr.append(1 if correct_w in k5 else 0)
            temp_arr.append(1 if correct_w in k10 else 0)

        final_dict[incorrect_w] = temp_arr
        temp_arr = []

    return final_dict


def compute_avg_at_K(succ_values):
    k1_arr = [values[0] for values in succ_values.values()]
    k5_arr = [values[1] for values in succ_values.values()]
    k10_arr = [values[2] for values in succ_values.values()]

    combined_arr = [["s@1", k1_arr], ["s@5", k5_arr], ["s@10", k10_arr]]
    agg_dict = {}
    for arr in combined_arr:
        agg_dict[arr[0]] = pytrec_eval.compute_aggregated_measure("s@1", arr[1])

    return agg_dict
