"""
Genetic analysis functions for the Red Panda Family Tree application
"""

import numpy as np
import networkx as nx
import pandas as pd
from utils import filter_japan_living_individuals


def find_oldest_ancestors(df, individual_name):
    """
    Find the oldest ancestors for a given individual using a directed graph.
    Returns a list of tuples (ancestor_name, weight) where weight is based on generation level.
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges from parents to children
    for _, row in df.iterrows():
        if pd.notna(row['father']):
            G.add_edge(row['father'], row['name'])
        if pd.notna(row['mother']):
            G.add_edge(row['mother'], row['name'])
    
    # Find all ancestors using networkx
    ancestors = nx.ancestors(G, individual_name)
    
    # Find the oldest ancestors (those with no parents in the dataset)
    oldest_ancestors = []
    for ancestor in ancestors:
        if ancestor == '':
            continue
        # Check if this ancestor has any parents in the dataset
        has_parents = False
        for _, row in df.iterrows():
            if row['name'] == ancestor:
                if row['father'] != '' or row['mother'] != '':
                    has_parents = True
                break
        if not has_parents:
            # Calculate the weight based on the shortest path length (generation level)
            path_length = len(nx.shortest_path(G, ancestor, individual_name)) - 1
            weight = 0.5 ** path_length
            oldest_ancestors.append((ancestor, weight))
    
    return oldest_ancestors


def get_gene_vector(df, ind_name, zoo_list, zoo_index):
    """個体の遺伝子ベクトルを計算する"""
    ancestors = find_oldest_ancestors(df, ind_name)
    vec = np.zeros(len(zoo_list))
    for ancestor, weight in ancestors:
        ancestor_data = df[df['name'] == ancestor]
        if not ancestor_data.empty:
            zoo = ancestor_data.iloc[0]['cur_zoo']
            if zoo in zoo_index:
                vec[zoo_index[zoo]] += weight
    return vec


def calculate_genetic_distances(df, target_individual, candidates_df, zoo_list, zoo_index):
    """遺伝子的距離を計算する"""
    target_vec = get_gene_vector(df, target_individual, zoo_list, zoo_index)
    distances = []
    
    for _, candidate_row in candidates_df.iterrows():
        name = candidate_row['name']
        if name == target_individual:
            continue
        
        vec = get_gene_vector(df, name, zoo_list, zoo_index)
        dist = np.linalg.norm(target_vec - vec)
        age = candidate_row['age']
        distances.append((name, dist, age))
    
    return sorted(distances, key=lambda x: -x[1])  # 距離が遠い順にソート


def get_opposite_gender_candidates(japan_df, selected_gender, max_age=9):
    """反対の性別で指定年齢以下の候補者を取得する"""
    opposite_gender = 'メス' if selected_gender == 'オス' else 'オス'
    candidates_df = japan_df[(japan_df['gender'] == opposite_gender) & (japan_df['age'] <= max_age)]
    return candidates_df, opposite_gender


def prepare_genetic_analysis_data(df):
    """Prepare data for genetic analysis"""
    # 日本にいる生存個体をフィルタリング
    japan_df = filter_japan_living_individuals(df)
    
    if japan_df is None:
        return None, None, None
    
    # 遺伝子ベクトル計算のための準備
    zoo_set = set(df['cur_zoo'].dropna().unique())
    zoo_list = sorted(list(zoo_set))
    zoo_index = {z: i for i, z in enumerate(zoo_list)}
    
    return japan_df, zoo_list, zoo_index 