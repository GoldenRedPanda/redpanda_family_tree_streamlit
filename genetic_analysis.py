"""
Genetic analysis functions for the Red Panda Family Tree application
"""

import numpy as np
import networkx as nx
import pandas as pd
from utils import filter_japan_living_individuals, clean_name


def compute_pedigree_inbreeding(raw_df):
    """
    Wright の近交係数（親の共祖先に基づく）を、加法的関係行列の標準アルゴリズムで算出する。
    F_i = A_ii - 1（親共に登録済みであること）。片親欠損や未登場の親は非近交交配とみなす。
    戻り値: pandas.DataFrame columns:
      name, inbreeding_coef, sire, dam, (+ merge された列がある場合あり)
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["name", "inbreeding_coef", "sire", "dam"])

    cdf = raw_df[["name", "father", "mother"]].copy()
    for col in ("name", "father", "mother"):
        cdf[col] = cdf[col].apply(lambda x: clean_name("" if pd.isna(x) else x))
    cdf = cdf[cdf["name"].str.len() > 0]
    if cdf.empty:
        return pd.DataFrame(columns=["name", "inbreeding_coef", "sire", "dam"])

    sire_by_name = dict(zip(cdf["name"], cdf["father"]))
    dam_by_name = dict(zip(cdf["name"], cdf["mother"]))

    all_names = (
        set(cdf["name"].unique())
        | {s for s in cdf["father"] if s}
        | {d for d in cdf["mother"] if d}
    )

    sire = {}
    dam = {}
    for n in sorted(all_names):
        if n in sire_by_name:
            s = sire_by_name[n]
            d = dam_by_name[n]
        else:
            s = ""
            d = ""
        s = "" if _invalid_parent_pair(n, s) else s
        d = "" if _invalid_parent_pair(n, d) else d
        sire[n] = s
        dam[n] = d

    # 親から子への辺でトポロジカル順（親が先行）
    G = nx.DiGraph()
    for child in sorted(all_names):
        G.add_node(child)
        for p in _parents_of(sire.get(child), dam.get(child)):
            G.add_edge(p, child)

    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        order = sorted(all_names)

    idx = {animal: i for i, animal in enumerate(order)}
    n = len(order)
    A = np.zeros((n, n), dtype=float)

    for ki, animal in enumerate(order):
        isire = sire.get(animal) or ""
        idam = dam.get(animal) or ""
        si = idx[isire] if isire else None
        di = idx[idam] if idam else None

        for j in range(ki):
            v = 0.0
            if si is not None:
                v += 0.5 * A[si, j]
            if di is not None:
                v += 0.5 * A[di, j]
            A[ki, j] = v
            A[j, ki] = v

        if si is not None and di is not None:
            A[ki, ki] = 1.0 + 0.5 * A[si, di]
        else:
            A[ki, ki] = 1.0

    out = pd.DataFrame(
        {
            "name": order,
            "inbreeding_coef": np.diag(A) - 1.0,
            "sire": [sire.get(a, "") for a in order],
            "dam": [dam.get(a, "") for a in order],
        }
    )
    rows_with_record = cdf["name"].unique()
    out = out[out["name"].isin(rows_with_record)].copy()
    return out.sort_values("inbreeding_coef", ascending=False)


def _invalid_parent_pair(child, parent):
    if not parent:
        return True
    if parent == child:
        return True
    return False


def _parents_of(sire, dam):
    return [p for p in (sire, dam) if p]


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