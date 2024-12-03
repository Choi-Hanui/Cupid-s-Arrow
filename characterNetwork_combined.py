# -*- coding: utf-8 -*-
"""
Created on Tues Oct 16 23:33:04 2018

@author: Ken Huang
"""

import codecs
import os
import spacy
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from afinn import Afinn
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import gender_guesser.detector as gender

def flatten(input_list):
    '''
    A function to flatten complex list.
    :param input_list: The list to be flatten
    :return: the flattened list.
    '''

    flat_list = []
    for i in input_list:
        if type(i) == list:
            flat_list += flatten(i)
        else:
            flat_list += [i]

    return flat_list


def common_words(path):
    '''
    A function to read-in the top common words from external .txt document.
    :param path: The path where the common words info is stored.
    :return: A set of the top common words.
    '''

    with codecs.open(path) as f:
        words = f.read()
        words = json.loads(words)

    return set(words)


def read_novel(book_name, path):
    '''
    A function to read-in the novel text from given path.
    :param book_name: The name of the novel.
    :param path: The path where the novel text file is stored.
    :return: the novel text.
    '''

    book_list = os.listdir(path)
    book_list = [i for i in book_list if i.find(book_name) >= 0]
    novel = ''
    for i in book_list:
        with codecs.open(path / i, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read().replace('\r', ' ').replace('\n', ' ').replace("\'", "'")
        novel += ' ' + data

    return novel


def name_entity_recognition(sentence, sentence_list):
    '''
    A function to retrieve name entities in a sentence.
    :param sentence: the sentence to retrieve names from.
    :return: a name entity list of the sentence.
    '''
    doc = nlp(sentence)
    context_info = {}
    # retrieve person and organization's name from the sentence
    name_entity = [x for x in doc.ents if x.label_ in ['PERSON']]
    # convert all names to lowercase and remove 's in names
    name_entity = [str(x).lower().replace("'s","") for x in name_entity]
    # split names into single words ('Harry Potter' -> ['Harry', 'Potter'])
    name_entity = [x.split(' ') for x in name_entity]
    # flatten the name list
    name_entity = flatten(name_entity)
    # remove name words that are less than 3 letters to raise recognition accuracy
    name_entity = [x for x in name_entity if len(x) >= 3]
    # remove name words that are in the set of 4000 common words
    name_entity = [x for x in name_entity if x not in words]
    
    # Extract contextual information
    female_keywords = ['she', 'her', 'hers', 'woman', 'girl', 'lady', 'queen', 'princess', 'ms', 'mrs']
    male_keywords = ['he', 'him', 'his', 'man', 'boy', 'gentleman', 'king', 'prince', 'mr']


    # Find the index of the current sentence in the sentence list
    current_index = sentence_list.index(sentence)

    # Extract surrounding sentences for context
    surrounding_text = " ".join(
        sentence_list[max(0, current_index - 1):min(len(sentence_list), current_index + 2)]
    ).lower()
    
    for name in name_entity:
        female_count = sum(keyword in surrounding_text for keyword in female_keywords)
        male_count = sum(keyword in surrounding_text for keyword in male_keywords)
        
        if female_count > male_count:
            context_info[name] = "female"
        elif male_count > female_count:
            context_info[name] = "male"
        else:
            context_info[name] = "unknown"

    return name_entity, context_info

def iterative_NER(sentence_list, threshold_rate=0.0005):
    '''
    A function to execute the name entity recognition function iteratively. The purpose of this
    function is to recognise all the important names while reducing recognition errors.
    :param sentence_list: the list of sentences from the novel
    :param threshold_rate: the per sentence frequency threshold, if a word's frequency is lower than this
    threshold, it would be removed from the list because there might be recognition errors.
    :return: a non-duplicate list of names in the novel.
    '''
    name_list = []
    context_info= {}
    for i in sentence_list:
        names, sentence_context_info = name_entity_recognition(i, sentence_list)
        if names != []:
            name_list.append(names)
            context_info.update(sentence_context_info)
    name_list = flatten(name_list)
    
    from collections import Counter
    name_list = Counter(name_list)
    name_list = [x for x in name_list if name_list[x] >= threshold_rate * len(sentence_list)]

    return name_list, context_info


def top_names(name_list, novel, top_num=20):
    '''
    A function to return the top names in a novel and their frequencies.
    :param name_list: the non-duplicate list of names of a novel.
    :param novel: the novel text.
    :param top_num: the number of names the function finally output.
    :return: the list of top names and the list of top names' frequency.
    '''

    if not name_list:  # name_list가 비었을 경우
        raise ValueError("The name list is empty. Please check the input to 'top_names'.")

    
    vect = CountVectorizer(vocabulary=name_list, stop_words='english')
    name_frequency = vect.fit_transform([novel.lower()])
    name_frequency = pd.DataFrame(name_frequency.toarray(), columns=vect.get_feature_names_out())
    name_frequency = name_frequency.T
    name_frequency = name_frequency.sort_values(by=0, ascending=False)
    name_frequency = name_frequency[0:top_num]
    names = list(name_frequency.index)
    name_frequency = list(name_frequency[0])

    return name_frequency, names

# 성별 예측기 초기화
detector = gender.Detector()

# 성별 추론 (이름 + 맥락 기반)
def predict_gender(names, context_info):
    genders = {}
    for name in names:
        # 컨텍스트 정보가 있으면 이를 우선 사용
        if name in context_info:
            gender_prediction = context_info[name]
        else:
            first_name = name.split()[0]  # 이름의 첫 번째 부분만 사용
            gender_prediction = detector.get_gender(first_name)
        
        genders[name] = gender_prediction
    return genders


def calculate_align_rate(sentence_list):
    '''
    Function to calculate the align_rate of the whole novel
    :param sentence_list: the list of sentence of the whole novel.
    :return: the align rate of the novel.
    '''
    afinn = Afinn()
    sentiment_score = [afinn.score(x) for x in sentence_list]
    align_rate = np.sum(sentiment_score)/len(np.nonzero(sentiment_score)[0]) * -2

    return align_rate


def calculate_matrix(name_list, sentence_list, align_rate):
    '''
    Function to calculate the co-occurrence matrix and sentiment matrix among all the top characters
    :param name_list: the list of names of the top characters in the novel.
    :param sentence_list: the list of sentences in the novel.
    :param align_rate: the sentiment alignment rate to align the sentiment score between characters due to the writing style of
    the author. Every co-occurrence will lead to an increase or decrease of one unit of align_rate.
    :return: the co-occurrence matrix and sentiment matrix.
    '''

    # calculate a sentiment score for each sentence in the novel
    afinn = Afinn()
    sentiment_score = [afinn.score(x) for x in sentence_list]
    # calculate occurrence matrix and sentiment matrix among the top characters
    name_vect = CountVectorizer(vocabulary=name_list, binary=True)
    occurrence_each_sentence = name_vect.fit_transform(sentence_list).toarray()
    cooccurrence_matrix = np.dot(occurrence_each_sentence.T, occurrence_each_sentence)
    sentiment_matrix = np.dot(occurrence_each_sentence.T, (occurrence_each_sentence.T * sentiment_score).T)
    sentiment_matrix += align_rate * cooccurrence_matrix
    cooccurrence_matrix = np.tril(cooccurrence_matrix)
    sentiment_matrix = np.tril(sentiment_matrix)
    # diagonals of the matrices are set to be 0 (co-occurrence of name itself is meaningless)
    shape = cooccurrence_matrix.shape[0]
    cooccurrence_matrix[[range(shape)], [range(shape)]] = 0
    sentiment_matrix[[range(shape)], [range(shape)]] = 0

    return cooccurrence_matrix, sentiment_matrix

def combine_edgelists(co_occurrence_file, sentiment_file, output_file):
    '''
    Combine co-occurrence and sentiment edgelists into a single file.
    :param co_occurrence_file: Path to the co-occurrence edgelist file.
    :param sentiment_file: Path to the sentiment edgelist file.
    :param output_file: Path to save the combined edgelist file.
    '''
    # Load the two edgelists
    co_occurrence_edges = nx.read_edgelist(co_occurrence_file, data=True)
    sentiment_edges = nx.read_edgelist(sentiment_file, data=True)

    # Create a new graph for the combined edgelist
    combined_graph = nx.Graph()

    # Add co-occurrence edges
    for u, v, data in co_occurrence_edges.edges(data=True):
        combined_graph.add_edge(u, v, co_occurence=data['co_occurence'], sentiment=0)  # Initialize sentiment as 0

    # Add sentiment edges
    for u, v, data in sentiment_edges.edges(data=True):
        if combined_graph.has_edge(u, v):
            combined_graph[u][v]['sentiment'] = data['sentiment']  # Update sentiment if edge exists
        else:
            combined_graph.add_edge(u, v, co_occurence=0, sentiment=data['weight'])  # Initialize co-occurrence as 0

    # Save the combined edgelist
    nx.write_edgelist(combined_graph, output_file, data=True)

def matrix_to_combined_edge_list(cooccurrence_matrix, sentiment_matrix, name_list):
    '''
    Function to create a combined edge list from co-occurrence and sentiment matrices.
    :param cooccurrence_matrix: Co-occurrence matrix.
    :param sentiment_matrix: Sentiment matrix.
    :param name_list: The list of names of the top characters in the novel.
    :return: A combined edge list with 'co_occurrence' and 'sentiment' weights.
    '''
    edge_list = []
    shape = cooccurrence_matrix.shape[0]
    lower_tri_loc = list(zip(*np.where(np.triu(np.ones([shape, shape])) == 0)))

    # Normalize matrices
    normalized_cooccurrence = cooccurrence_matrix / np.max(cooccurrence_matrix)
    normalized_sentiment = sentiment_matrix / np.max(np.abs(sentiment_matrix))

    for i in lower_tri_loc:
        if cooccurrence_matrix[i[0], i[1]] != 0 or sentiment_matrix[i[0], i[1]] != 0:
            edge_list.append((
                name_list[i[0]],
                name_list[i[1]],
                {
                    'co_occurrence': np.log(2000 * normalized_cooccurrence[i] + 1) * 0.7 if cooccurrence_matrix[i[0], i[1]] != 0 else 0,
                    'sentiment': np.log(np.abs(1000 * normalized_sentiment[i]) + 1) * 0.7 if sentiment_matrix[i[0], i[1]] != 0 else 0
                }
            ))

    return edge_list

def plot_combined_graph(name_list, name_frequency, cooccurrence_matrix, sentiment_matrix, genders, plt_name, path=''):
    """
    성별에 따라 노드 색상을 설정하여 그래프를 생성합니다.
    :param name_list: 주요 등장인물 리스트.
    :param name_frequency: 주요 등장인물의 등장 빈도 리스트.
    :param cooccurrence_matrix: 공존 행렬.
    :param sentiment_matrix: 감정 행렬.
    :param genders: {이름: 성별} 형태의 사전.
    :param plt_name: 그래프 파일 이름.
    :param path: 그래프 파일 저장 경로.
    """
    # 성별을 F, M, U로 변환
    gender_map = {
        'male': 'M',
        'female': 'F',
        'unknown': 'U',
        'andy': 'U'  # 모호한 경우 처리
    }
    mapped_genders = {name: gender_map.get(gender, 'U') for name, gender in genders.items()}
    
    # 성별에 따른 색상 매핑
    color_map = {'F': 'red', 'M': 'blue', 'U': 'gray'}
    node_colors = [color_map[mapped_genders.get(name, 'U')] for name in name_list]
    
    # 노드 레이블 설정
    label = {i: i for i in name_list}
    
    # 엣지 리스트 생성
    edge_list = matrix_to_combined_edge_list(cooccurrence_matrix, sentiment_matrix, name_list)
    
    # 등장 빈도 정규화
    normalized_frequency = np.array(name_frequency) / np.max(name_frequency)

    # 그래프 생성
    G = nx.Graph()
    G.add_nodes_from(name_list)
    G.add_edges_from(edge_list)
    
    pos = nx.circular_layout(G)
    edges = G.edges()

    # 엣지 가중치 추출
    co_occurrence_weights = [G[u][v]['co_occurrence'] for u, v in edges]
    sentiment_weights = [G[u][v]['sentiment'] for u, v in edges]

    # 그래프 그리기
    plt.figure(figsize=(20, 20))
    nx.draw(
        G,
        pos,
        node_color=node_colors,  # 노드 색상
        node_size=np.sqrt(normalized_frequency) * 4000,  # 노드 크기
        linewidths=10,
        font_size=35,
        labels=label,
        edge_color=sentiment_weights,  # 감정 가중치로 엣지 색상
        edge_cmap=plt.cm.RdYlBu,
        with_labels=True,
        width=co_occurrence_weights
    )

    # 그래프 저장
    plt.savefig(path + plt_name + '.png')
    nx.write_edgelist(G, path + plt_name + '.edgelist', data=True)

def save_nodelist(genders, output_file):
    """
    성별 정보를 기반으로 nodelist 파일을 생성합니다.
    :param genders: {이름: 성별} 형태의 사전.
    :param output_file: 저장할 파일 경로.
    """
    # 성별을 F, M, U 형식으로 매핑
    gender_map = {
        'male': 'M',
        'female': 'F',
        'unknown': 'U',
        'andy': 'U'  # gender_guesser의 'andy' (ambiguous) 처리
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for name, gender in genders.items():
            mapped_gender = gender_map.get(gender, 'U')  # 성별 매핑, 없으면 'U'
            f.write(f"{name},{mapped_gender}\n")

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 2000000 
    words = common_words('common_datas/common_words.txt')
    novel_name = 'APairOfBlueEyes'
    novel_folder = Path(os.getcwd()) / 'novels'
    novel = read_novel(novel_name, novel_folder)
    # sentence_list = sent_tokenize(novel)
    sentence_list = [sent.text for sent in nlp(novel).sents]
    align_rate = calculate_align_rate(sentence_list)
    # 이름 및 맥락 추출
    preliminary_name_list, context_info = iterative_NER(sentence_list) 
    name_frequency, name_list = top_names(preliminary_name_list, novel, 25)
    
    # 성별 추론
    genders = predict_gender(name_list, context_info)

    # 최상위 등장인물 및 빈도 출력
    print("주요 등장인물 및 등장 빈도:")
    for name, freq in zip(name_list, name_frequency):
        print(f"{name}: {freq}회 등장")
    
    print("\n등장인물의 성별 추정:")
    for name, gender_prediction in genders.items():
        print(f"{name}: {gender_prediction}")
   
    cooccurrence_matrix, sentiment_matrix = calculate_matrix(name_list, sentence_list, align_rate)
    
    # Nodelist 저장
    nodelist_path = f'./graphs/{novel_name}_gender.nodelist'
    save_nodelist(genders, nodelist_path)


    # Create and save the combined graph
    plot_combined_graph(
        name_list,
        name_frequency,
        cooccurrence_matrix,
        sentiment_matrix,
        genders,
        novel_name + ' combined graph',
        path='./graphs/'
    )