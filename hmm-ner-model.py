import pandas as pd
import csv
import more_itertools as mit
import collections as col
import numpy as np


def read_corpus(corpus_file):
    """
    read in tsv corpus file and return as dataframe

    restructures the data using restructure_data()-function
    """
    df = pd.read_csv(corpus_file, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE,
                     header=None, names=['instance_index',
                                         'token', 'label',
                                         'alternative_label'])
    df = restructure_data(df)
    return df


def restructure_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    get rid of metadata rows and transfer document
    boarders to new column by using numbers for each instance

    Parameters:
        df (pd.DataFrame): corpus

    Returns:
        df (pd.DataFrame): modified corpus dataframe

    """
    # initialize document counter
    doc_number = 0

    # add a new column with document numbers
    doc_numbers = []
    for i, row in df.iterrows():
        if row['instance_index'].startswith('#'):
            doc_number += 1
        doc_numbers.append(doc_number)
    df['instance'] = doc_numbers

    # remove the metadata column
    df = df[df['instance_index'].str.startswith('#') == False]
    return df

# function is not used for this exercise
def save_new_df(df: pd.DataFrame):
    """
    Saves dataframe to tab-separated csv file

    Parameters:
        df: pd.DataFrame
    """
    df.to_csv('modified_data.csv', index=False, sep='\t')


def initial_probability(df: pd.DataFrame) -> dict[str, float]:
    """
    calculate initial probabilities of a BIO-labelled corpus

    Parameters:
        df (pd.DataFrame): Dataframe containing tokens and their corresponding IOB-Labels as well as the position of
        each token/label in the corresponding instance, starting with 1

    Returns:
        initial_probabilities (dict): Dictionary with the initial probability of each IOB-label
            keys: labels
            values: initial probabilities
    """
    # initialize dictionary to save values to
    label_count = {'B': 0, 'I': 0, 'O': 0}
    # filter dataframe so that only first label of each instance is shown
    initial_labels = df[df['instance_index'] == '1']['label']
    # count occurrences of labels
    for label in initial_labels:
        if label.startswith('B'):
            label_count['B'] += 1
        elif label.startswith('I'):
            label_count['I'] += 1
        elif label.startswith('O'):
            label_count['O'] += 1
    # get total amount of labels to calculate relative frequencies
    total = len(initial_labels)
    initial_probabilities = {}
    for label, count in label_count.items():
        initial_probabilities[label] = round(count / total, 2)
    return initial_probabilities


def observed_probabilities(df: pd.DataFrame):
    """
    calculate observed probabilities of tokens in IOB-labeled corpus

    Parameters:
        df (pd.DataFrame): IOB-labeled corpus

    Returns:
        observed_prob_dict (dict): Dictionary containing observed frequencies for each token in corpus
            keys: tokens
            values: dict with each label as key and observed probabilities as values
        most_10_B_tokens (dict): Dictionary containing top 10 tokens for B-label
            keys: tokens
            values: dict with each label as key and observed probabilities as values
        most_10_I_tokens (dict): Dictionary containing top 10 tokens for I-label
            keys: tokens
            values: dicts with each label as key and observed probabilities as values
        most_10_O_tokens (dict): Dictionary containing top 10 tokens for O-label
            keys: tokens
            values: dicts with each label as key and observed probabilities as values
    """
    observed_prob_dict = {}
    # get the absolute frequency of labels for each token
    for (index, elem) in df.iterrows():
        if elem['token'] not in observed_prob_dict:
            token_label_dict = {'B':0, 'I':0, 'O':0}
            token_label_dict[elem['label'][0]] = 1
            observed_prob_dict[elem['token']] = token_label_dict
        else:
            observed_prob_dict[elem['token']][elem['label'][0]] += 1
    # get the relative frequency of labels for each token
    for (key,value) in observed_prob_dict.items():
        for (label, count) in value.items():
            value[label] = count/sum(value.values())
    # get top 10 tokens for each label
    most_10_B_tokens = sorted(observed_prob_dict.items(), key=lambda i: i[1]['B'], reverse=True)[:10]
    most_10_I_tokens = sorted(observed_prob_dict.items(), key=lambda i: i[1]['I'], reverse=True)[:10]
    most_10_O_tokens = sorted(observed_prob_dict.items(), key=lambda i: i[1]['O'], reverse=True)[:10]
    return observed_prob_dict, most_10_B_tokens, most_10_I_tokens, most_10_O_tokens


def transition_probability(df):
    """calculate transition probabilities of a BIO-labelled corpus

    Parameters:
    df (pd.DataFrame): Dataframe containing tokens and their corresponding IOB-Labels as well as the position of
        each token/label in the corresponding instance, starting with 1

    Returns:
     transition_prob_dict (dict): Dictionary with the transition probability of each IOB-label
            keys: labels
            values: list of transition probabilities(in the order of 'B I O')
    """
    next_token_dict = {'B':[],'I':[],'O':[]}
    peek_df_iterrows = mit.peekable(df.iterrows())
    # for each token add its next token to the next_token_dict
    for (index, elem) in peek_df_iterrows:
        if index< df.shape[0]-1:
            next_label = peek_df_iterrows.peek()[1]['label']
            next_index = peek_df_iterrows.peek()[1]['instance_index']
            if next_index != '1': # get rid of the situation, if it is the last token in a sequence
                next_token_dict[elem['label'][0]].append(next_label[0])
    # count the labels in the next_token_dict
    B_to = dict(col.Counter(next_token_dict['B']))
    I_to = dict(col.Counter(next_token_dict['I']))
    O_to = dict(col.Counter(next_token_dict['O']))
    # make the count result to a list
    B_to_list = [B_to['B'], B_to['I'], B_to['O']]
    I_to_list = [I_to['B'], I_to['I'], I_to['O']]
    O_to_list = [O_to['B'], 0, O_to['O']]
    # calculate the transition probability for each label
    B_to_prob = {'B':round(B_to['B']/sum(B_to_list),2), 
                 'I':round(B_to['I']/sum(B_to_list),2), 
                 'O':round(B_to['O']/sum(B_to_list),2)}
    I_to_prob = {'B':round(I_to['B']/sum(I_to_list),2), 
                 'I':round(I_to['I']/sum(I_to_list),2), 
                 'O':round(I_to['O']/sum(I_to_list),2)}
    O_to_prob = {'B':round(O_to['B']/sum(O_to_list),2), 
                 'I':0.00, 
                 'O':round(O_to['O']/sum(O_to_list),2)}
    transition_prob_dict = {'B': B_to_prob, 'I': I_to_prob, 'O': O_to_prob}
    return transition_prob_dict

def get_top10_instances(df):
    """get top 10 instance from corpus

    Parameters:
    df (pd.DataFrame): Dataframe containing tokens and their corresponding IOB-Labels as well as the position of
        each token/label in the corresponding instance, starting with 1

    Returns:
    instance_dict (dict): Dictionary with the top 10 instances
            keys: instance numbers
            values: lists of token list and label list
    """
    instance_dict = {}
    for (index, elem) in df.iterrows():
        if int(elem['instance']) < 11:
            if elem['instance'] not in instance_dict:
                instance_dict[elem['instance']] = [[],[]]
                instance_dict[elem['instance']][0].append(elem['token'])
                instance_dict[elem['instance']][1].append(elem['label'])
            else:
                instance_dict[elem['instance']][0].append(elem['token'])
                instance_dict[elem['instance']][1].append(elem['label'])
    return instance_dict

def calculate_10_instances_gold_probability(df):
    """calculate the gold probability for top 10 instance

    Parameters:
    df (pd.DataFrame): Dataframe containing tokens and their corresponding IOB-Labels as well as the position of
        each token/label in the corresponding instance, starting with 1
    """
    initial_prob_dict = initial_probability(df)
    transition_prob_dict = transition_probability(df)
    observed_prob_dict = observed_probabilities(df)
    # get the top 10 instances 
    instance_dict = get_top10_instances(df)
    # calculate the probability of label and token for each instance
    for (key,value) in instance_dict.items():
        # calculate the initial probability
        initial_prob = initial_prob_dict[value[1][0][0]]
        transition_prob_list = []
        observed_prob_list = []
        token_label = zip(value[0],value[1])
        peek_token_label = mit.peekable(enumerate(token_label))
        # calculate the transition und observed probability for each token and label
        for count,elem in peek_token_label:
            if count < len(value[0])-1:
                current_label = elem[1][0]
                next_label = peek_token_label.peek()[1][1][0]
                current_token = elem[0]
                transition_prob_list.append(transition_prob_dict[current_label][next_label])
                observed_prob_list.append(observed_prob_dict[0][current_token][current_label])
        # get the total transition and observed probability
        transition_prob = np.prod(transition_prob_list)
        observed_prob = np.prod(observed_prob_list)
        # get the gold probability for one instance
        gold_prob = initial_prob * transition_prob * observed_prob
        # get the token and label sequences
        token_sentence = ' '.join(value[0])
        label_sentence = ' '.join(value[1])
        # print the result for one instance
        print(f'instance{key}:\n {token_sentence}\n {label_sentence}\n gold probability: {gold_prob}')


if __name__ == '__main__':
    # read in data as pandas dataframe
    data = read_corpus('NER-de-train.tsv')
    
    # initial probability
    print('\n-----------------------------------Initial Probabilities---------------------------------------------------')
    print(initial_probability(data))
    # observed probability: top 10 tokens for each label
    print('\n-----------------------------------Observed Probabilities--------------------------------------------------')
    print('top 10 B-label tokens:\n', observed_probabilities(data)[1],'\n')
    print('top 10 I-label tokens:\n', observed_probabilities(data)[2],'\n')
    print('top 10 O-label tokens:\n', observed_probabilities(data)[3],'\n')
    # transition probability
    print('\n-----------------------------------Transition Probabilities------------------------------------------------')
    print(transition_probability(data))

    # calculate_10_instances_gold_probability
    dev_data = read_corpus('NER-de-dev.tsv')
    print('\n-----------------------------------Calculate_10_instances_gold_probability---------------------------------')
    calculate_10_instances_gold_probability(dev_data)
