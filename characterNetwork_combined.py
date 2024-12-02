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
import gender_guesser.detector as gender
from collections import Counter
import re

# 텍스트 파일 읽기
def read_novel(novel_path):
    with open(novel_path, "r", encoding="utf-8") as f:
        return f.read()

# 성별 예측기 초기화
detector = gender.Detector()

# 텍스트 전처리
def preprocess_text(text):
    # 줄바꿈 제거 및 기본 전처리
    return text.replace("\n", " ").replace("\r", "").strip()

# 인물 이름 및 맥락 정보 추출
def extract_names_and_context(text):
    doc = nlp(text)
    characters = []
    context_info = {}
    
    male_keywords = {"mr.", "sir", "lord", "father", "clergyman", "king", "husband", "son", "brother", "prince", "captain", "duke"}
    female_keywords = {"mrs.", "miss", "lady", "queen", "mother", "wife", "daughter", "sister", "widow", "maid", "princess", "duchess"}
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
             # 이름 정규화 및 소문자 변환
            name = (
                ent.text.replace("’s", "")
                .replace("'s", "")
                .replace("Mr. ", "")
                .replace("Miss ", "")
                .replace("Mrs. ", "")
                .strip()
                .lower()
            )
            characters.append(name)
            
            name = re.sub(r'\.$', '', name)
            # 주변 텍스트에서 역할이나 힌트 추출
            surrounding_text = doc[max(0, ent.start-5):min(len(doc), ent.end+5)].text.lower()
            if any(keyword in surrounding_text for keyword in female_keywords):
                context_info[name] = "female"
            elif any(keyword in surrounding_text for keyword in male_keywords):
                context_info[name] = "male"
                
    print(characters)
    return characters, context_info



def get_top_characters(names, top_n=20):
    """
    Extracts the top N characters based on their frequency in the text.

    :param names: List of all character names in the text.
    :param top_n: Number of top characters to return.
    :return: A tuple of (frequencies, character_names).
    """
    name_counts = Counter(names)
    most_common = name_counts.most_common(top_n)
    
    # Extract frequencies and names into separate lists
    frequencies = [count for _, count in most_common]
    character_names = [name for name, _ in most_common]
    
    return frequencies, character_names



# 성별 추론 (이름 + 맥락 기반)
def predict_gender(names, context_info):
    genders = {}
    for name in names:
        first_name = name.split()[0]  # 이름의 첫 번째 부분만 사용
        gender_prediction = detector.get_gender(first_name)
        
        # 맥락 정보로 보완
        if name in context_info:
            gender_prediction = context_info[name]
        elif gender_prediction == "unknown":  # 주변 맥락으로 성별 보완
            gender_prediction = context_info.get(name, "unknown")
        
        genders[name] = gender_prediction
    return genders


def top_names(name_list, novel, top_num=20):
    '''
    A function to return the top names in a novel and their frequencies.
    :param name_list: the non-duplicate list of names of a novel.
    :param novel: the novel text.
    :param top_num: the number of names the function finally output.
    :return: the list of top names and the list of top names' frequency.
    '''

    vect = CountVectorizer(vocabulary=name_list, stop_words='english')
    name_frequency = vect.fit_transform([novel.lower()])
    name_frequency = pd.DataFrame(name_frequency.toarray(), columns=vect.get_feature_names_out())
    name_frequency = name_frequency.T
    name_frequency = name_frequency.sort_values(by=0, ascending=False)
    name_frequency = name_frequency[0:top_num]
    names = list(name_frequency.index)
    name_frequency = list(name_frequency[0])

    return name_frequency, names


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


if __name__ == "__main__":
    # Spacy 및 Afinn 로드
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 2000000
    afinn = Afinn()

    # 텍스트 읽기
    novel_name = 'ThePhantomOfTheOpera'  # 확장자를 제외한 제목만 입력
    novel_folder = Path(os.getcwd()) / 'novels'

    # 파일 경로 조합 시 확장자 자동 추가
    novel_path = novel_folder / f"{novel_name}.txt"
    
    # 텍스트 읽기 및 전처리
    novel = preprocess_text(read_novel(novel_path))
    
    # 문장 리스트 생성
    sentence_list = [sent.text for sent in nlp(novel).sents]
    total_sentence = len(sentence_list)
    
    # 이름 및 맥락 추출
    names, context_info = extract_names_and_context(novel)
    
    name_frequency, name_list = get_top_characters(names)
    name_list = [name.lower() for name in name_list]
    # 성별 추론
    genders = predict_gender(name_list, context_info)

    # 최상위 등장인물 및 빈도 출력
    print("주요 등장인물 및 등장 빈도:")
    for name, freq in zip(name_list, name_frequency):
        print(f"{name}: {freq}회 등장")
    
    print("\n등장인물의 성별 추정:")
    for name, gender_prediction in genders.items():
        print(f"{name}: {gender_prediction}")

    # 감정 및 공존 행렬 계산
    align_rate = calculate_align_rate(sentence_list)
    cooccurrence_matrix, sentiment_matrix = calculate_matrix(name_list, sentence_list, align_rate)
    
    # Nodelist 저장
    nodelist_path = f'./graphs/{novel_name}_gender.nodelist'
    save_nodelist(genders, nodelist_path)

    # 네트워크 그래프 생성
    plot_combined_graph(
        name_list=name_list,
        name_frequency=name_frequency,
        cooccurrence_matrix=cooccurrence_matrix,
        sentiment_matrix=sentiment_matrix,
        plt_name=novel_name + ' combined graph',
        genders=genders,
        path='./graphs/'
    )
