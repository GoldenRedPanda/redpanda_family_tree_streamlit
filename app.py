import csv
import re
import streamlit as st
import zlib
import base64
import urllib.parse
import numpy as np
from datetime import datetime
import os
from streamlit_mermaid import st_mermaid
from collections import OrderedDict
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from urllib.parse import urlparse
import networkx as nx

def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def mermaid_to_pako_url(mermaid_code: str) -> str:
    # gzip 圧縮（zlib形式 + headerなし + 最小サイズ）
    compressed = zlib.compress(mermaid_code.encode('utf-8'), level=9)[2:-4]

    # base64url エンコード
    encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')

    # Mermaid Live Editor のURLに付加
    url = f"https://mermaid.live/edit#pako:{encoded}"
    return url

class OrderedSet:
    def __init__(self):
        self.data = OrderedDict()

    def add(self, item):
        self.data[item] = None  # 値は不要

    def __contains__(self, item):
        return item in self.data

    def __iter__(self):
        return iter(self.data.keys())

    def __repr__(self):
        return f"OrderedSet({list(self.data.keys())})"

def clean_name(name):
    if pd.isna(name) or name is None:
        return ''
    if isinstance(name, float):
        return ''
    return re.sub(r'\s*\(.*?\).*', '', str(name)) if name else ''

def escape_mermaid(name):
    return name.replace("-", "\\-").replace(" ", "\\ ").replace("(", "\\(").replace(")", "\\)")
    return name

def parse_birthdate(birthdate):
    match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', birthdate)
    if match:
        year, month, day = map(int, match.groups())
        return datetime(year, month, day)
    return datetime.min

def read_csv(file_path):
    family_data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['name'] = escape_mermaid(clean_name(row['name']))
            row['father'] = escape_mermaid(clean_name(row['father']))
            row['mother'] = escape_mermaid(clean_name(row['mother']))
            row['birthdate'] = row['birthdate'] if 'birthdate' in row else ''
            row['gender'] = row['gender'] if 'gender' in row else 'オス'
            row['image'] = row['image'] if 'image' in row else ''
            family_data.append(row)
    return family_data

def sort_children(children):
    def sort_key(child):
        birthdate_parsed = parse_birthdate(child.get('birthdate', ''))
        gender = child.get('gender', 'オス')  # デフォルトは"オス"
        return (birthdate_parsed, 0 if gender == "オス" else 1)  # オスを左に配置
    
    return sorted(children, key=sort_key)

def generate_mermaid(family_data, root_name=None, parent_depth=2, child_depth=2, show_images=False):
    mermaid_code = "graph TD;\n"
    connections = OrderedSet()
    
    family_dict = {person['name']: person for person in family_data}
    
    def add_person_node(person):
        if person.get('image', ''):
            img = person.get('image', '').split(',')[0]
            if show_images and is_url(img):
                return f"{person['name']}[{person['name']}<img src=\"{img}\" />]"
            else:
                return f"{person['name']}[{person['name']}]"
        return f"{person['name']}"
    
    def add_ancestors(person_name, depth=0):
        if person_name not in family_dict or depth >= parent_depth:
            return
        
        person = family_dict[person_name]
        father = person['father']
        mother = person['mother']
        
        if father or mother:
            parent_node = f"{father}_{mother}" if father and mother else father if father else mother
            connections.add(f"{parent_node}(( ))")  # 中間ノード
            connections.add(f"{parent_node} --> {add_person_node(person)}")
        
        if father:
            connections.add(f"{add_person_node({'name': father})} --> {parent_node}")
            add_ancestors(father, depth + 1)
        if mother:
            connections.add(f"{add_person_node({'name': mother})} --> {parent_node}")
            add_ancestors(mother, depth + 1)
    
    def add_descendants(person_name, depth=0):
        if person_name not in family_dict or depth >= child_depth:
            return
        
        children = [member for member in family_data if member['father'] == person_name or member['mother'] == person_name]
        sorted_children = sort_children(children)
        
        for child in sorted_children:
            child_name = child['name']
            father = child['father']
            mother = child['mother']
            
            if father and mother:
                parent_node = f"{father}_{mother}"
                connections.add(f"{parent_node}(( ))")  # 中間ノード
                connections.add(f"{add_person_node({'name': father})} --> {parent_node}")
                connections.add(f"{add_person_node({'name': mother})} --> {parent_node}")
            else:
                parent_node = f"{father or mother}_child"
                connections.add(f"{parent_node}(( ))")  # 中間ノード
                connections.add(f"{add_person_node({'name': father or mother})} --> {parent_node}")
            
            connections.add(f"{parent_node} --> {add_person_node(child)}")
            add_descendants(child_name, depth + 1)
    
    if root_name:
        add_ancestors(root_name)
        add_descendants(root_name)
    #else:
    #    for person in family_data:
    #        add_ancestors(person['name'])
    #        add_descendants(person['name'])
    
    mermaid_code += "\n".join(connections)
    return mermaid_code

def create_gantt_chart(tasks, zoo_name):
    """
    Create a Gantt chart from a list of tasks.
    Each task should be a dictionary with 'Task', 'Start', 'End', and 'Status' keys.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort tasks by start date
    tasks = sorted(tasks, key=lambda x: x['Start'])
    
    # Create y-axis labels (task names)
    y_labels = [task['Task'] for task in tasks]
    
    # Calculate task durations
    durations = [(task['End'] - task['Start']).days for task in tasks]
    print(durations)
    
    # Calculate start positions (days from the earliest start date)
    min_start = min(task['Start'] for task in tasks)
    start_positions = [(task['Start'] - min_start).days for task in tasks]
    
    # Define colors based on status
    colors = {
        'Dead': 'lightgray',
        'Live': 'lightgreen',
    }
    
    # Create horizontal bars
    bars = ax.barh(y_labels, durations, left=start_positions, 
                  color=[colors.get(task['Status'], 'lightgray') for task in tasks])
    
    # Add task labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(bar.get_x() + width/2, bar.get_y() + bar.get_height()/2,
                f"{tasks[i]['Status']}", ha='center', va='center')
    
    # Set x-axis to show dates
    date_range = (min_start, max(task['End'] for task in tasks))
    date_delta = (date_range[1] - date_range[0]).days
    
    # Generate date ticks every year
    current_date = min_start
    date_ticks = []
    while current_date <= date_range[1]:
        date_ticks.append(current_date)
        # Add 1 year (365 days)
        current_date = current_date + timedelta(days=365)
    
    ax.set_xticks([(d - min_start).days for d in date_ticks])
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in date_ticks], rotation=45)
    
    # Add grid and labels
    ax.grid(True, axis='x')
    ax.set_xlabel('Date')
    ax.set_ylabel('Name')
    ax.set_title(f'Gantt Chart of {zoo_name}')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig

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

def plot_genetic_distribution(df, ancestors):
    """
    Plot the genetic distribution of the oldest ancestors with weights.
    """
    # Count the weighted number of ancestors from each zoo
    zoo_counts = {}
    for ancestor, weight in ancestors:
        ancestor_data = df[df['name'] == ancestor]
        if not ancestor_data.empty:
            birth_zoo = ancestor_data.iloc[0]['cur_zoo']
            zoo_counts[birth_zoo] = zoo_counts.get(birth_zoo, 0) + weight
    
    # Create a pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    if zoo_counts:
        labels = list(zoo_counts.keys())
        sizes = list(zoo_counts.values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Weighted Genetic Distribution of Oldest Ancestors')
    else:
        ax.text(0.5, 0.5, 'No ancestor data available', 
                horizontalalignment='center', verticalalignment='center')
        ax.set_title('Genetic Distribution')
    
    return fig

st.title("Family Tree Generator")

default_csv_path = "redpanda.csv"
use_default = st.checkbox("Use default CSV file (family_data.csv in the same folder)", value=True)
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
data = None
if use_default and os.path.exists(default_csv_path):
    data = read_csv(default_csv_path)
elif uploaded_file is not None:
    data = read_csv(uploaded_file)

tr, ppy, gantt, genetic = st.tabs(["Family Tree", "Population Pyramid", "Gantt Chart", "Genetic Distribution"])
with tr:
    show_images = st.checkbox("Show Images", value=False)
    parent_depth = st.number_input("Parent Generation Depth", min_value=1, value=2)
    child_depth = st.number_input("Child Generation Depth", min_value=1, value=2)
    
    if data:
        name_options = [person['name'] for person in data]
        root_name = st.selectbox("Root Name", [""] + name_options)
        mermaid_code = generate_mermaid(data, root_name if root_name else None, parent_depth, child_depth, show_images)
        st.text_area("Generated Mermaid Code", mermaid_code, height=300)
        url = mermaid_to_pako_url(mermaid_code)
        st.markdown(f"[Mermaid Live Editor で開く]({url})", unsafe_allow_html=True)
        st_mermaid(mermaid_code)
with ppy:
    # CSVファイルの読み込み
    if use_default and os.path.exists(default_csv_path):
        df = pd.read_csv(default_csv_path)
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    sel_date = st.date_input("日付を選んでください", date.today(), min_value=date(2005, 1, 1), max_value=date.today())
    df = df[~df['cur_zoo'].isin(['中国', '台湾', 'カナダ', 'アメリカ', 'チリ', '韓国', 'インドネシア', 'アルゼンチン', 'タイ', 'メキシコ'])]
    live_df = df[df['deaddate'].isnull()].copy()
    dead_df = df[~df['deaddate'].isnull()].copy()
    # 日付の変換とデータのクレンジング
    def convert_date(date_str):
        try:
          if type(date_str) is str:
            return datetime.strptime(date_str, '%Y年%m月%d日').date()
          else:
            return date(1980, 1, 1)
        except ValueError:
          return date(1980, 1, 1)
    
    live_df['birthdate'] = pd.to_datetime(live_df['birthdate'].apply(convert_date))
    live_df['deaddate'] = pd.to_datetime(live_df['deaddate'].apply(convert_date))
    dead_df['birthdate'] = pd.to_datetime(dead_df['birthdate'].apply(convert_date))
    dead_df['deaddate'] = pd.to_datetime(dead_df['deaddate'].apply(convert_date))

    
    # 2003年1月1日以降のデータのみ抽出
    live_df = live_df[live_df['birthdate'] >= pd.Timestamp('2003-01-01')]
    
    # 年齢の計算
    today = pd.Timestamp(sel_date)
    sel_df = live_df[today > live_df['birthdate']]
    sel_df = pd.concat([sel_df, dead_df[(today>dead_df['birthdate'])&(today<dead_df['deaddate'])]])
    sel_df['age'] = (today - sel_df['birthdate']).dt.days // 365
    # 年齢ビン（0歳〜24歳の25区分）
    bins = list(range(0, 25)) + [100]  # 24歳以上も含める
    
    # ヒストグラム用にカウント
    male_counts, _ = np.histogram(sel_df[sel_df['gender'] == 'オス']['age'], bins=bins)
    female_counts, _ = np.histogram(sel_df[sel_df['gender'] == 'メス']['age'], bins=bins)
    
    # 各ビンの中心位置（0〜24歳）
    bin_centers = list(range(0, 25))
    n_panda = len(sel_df)
    # Streamlit タイトル
    st.title(f"人口ピラミッド {n_panda}頭")
    
    # Matplotlib でピラミッドグラフを作成
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(bin_centers, -male_counts, height=0.8, label='Male', color='skyblue')
    ax.barh(bin_centers, female_counts, height=0.8, label='Female', color='lightpink')
    
    ax.set_yticks(bin_centers)
    ax.set_yticklabels([str(age) for age in bin_centers])
    ax.set_xlabel('Population')
    ax.set_title('Population Pyramid')
    ax.grid(True)
    ax.legend()
    
    ax.set_xticks(range(-max(male_counts), max(female_counts) + 1))
    ax.set_xticklabels([abs(x) for x in range(-max(male_counts), max(female_counts) + 1)])
    
    st.pyplot(fig)

with gantt:
    st.title("Project Gantt Chart")
    # CSVファイルの読み込み
    if use_default and os.path.exists(default_csv_path):
        df = pd.read_csv(default_csv_path)
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    live_df = df[df['deaddate'].isnull()].copy()
    dead_df = df[~df['deaddate'].isnull()].copy()
    # 日付の変換とデータのクレンジング
    def convert_date(date_str):
        try:
          if type(date_str) is str:
            return datetime.strptime(date_str, '%Y年%m月%d日').date()
          else:
            return date.today()
        except ValueError:
          return date(1980, 1, 1)
    
    live_df['birthdate'] = pd.to_datetime(live_df['birthdate'].apply(convert_date))
    live_df['deaddate'] = pd.to_datetime(live_df['deaddate'].apply(convert_date))
    dead_df['birthdate'] = pd.to_datetime(dead_df['birthdate'].apply(convert_date))
    dead_df['deaddate'] = pd.to_datetime(dead_df['deaddate'].apply(convert_date))

    
    # 2003年1月1日以降のデータのみ抽出
    live_df = live_df[live_df['birthdate'] >= pd.Timestamp('2003-01-01')]
    
    df = pd.concat([live_df, dead_df])
    zoo_options = list(df['birth_zoo'].unique())
    zoo_name = st.selectbox("Zoo Name", [""] + zoo_options)

    df = df[df['birth_zoo'] == zoo_name]
    # Use the DataFrame columns for the Gantt chart
    if df.shape[0] > 0:
        # Convert DataFrame to tasks format
        tasks = []
        for _, row in df.iterrows():
            if pd.notna(row['birthdate']) and pd.notna(row['deaddate']):
                # Determine status based on end date
                status = "Live" if row['deaddate'].date() == date.today() else "Dead"
                tasks.append({
                    'Task': row['name'],
                    'Start': row['birthdate'].date(),
                    'End': row['deaddate'].date(),
                    'Status': status
                })
        
        if tasks:
            print(tasks)
            # Create and display the Gantt chart
            gantt_fig = create_gantt_chart(tasks, zoo_name)
            st.pyplot(gantt_fig)
            
            # Add explanation
            st.markdown("""
            ### Gantt Chart Explanation
            
            This Gantt chart shows the timeline of each individual:
            
            - **Tasks**: Individual names (y-axis)
            - **Timeline**: Birth date to death date (x-axis)
            - **Status**: 'Live' for individuals alive today, 'Dead' for individuals deceased
            
            The chart visualizes the lifespan of each individual from their birth date to death date.
            """)
        else:
            st.warning("No valid data available for the Gantt chart. Please ensure the data contains birth and death dates.")
    else:
        st.warning("Please upload a CSV file or use the default file to view the Gantt chart.")

with genetic:
    st.title("Genetic Distribution of Oldest Ancestors")
    
    # CSVファイルの読み込み
    if use_default and os.path.exists(default_csv_path):
        df = pd.read_csv(default_csv_path)
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    df['father'] = df['father'].apply(clean_name)
    df['mother'] = df['mother'].apply(clean_name)
    
    # Use the same DataFrame as in the Gantt chart tab
    if df.shape[0] > 0:
        # Select an individual
        individual_options = list(df['name'].unique())
        selected_individual = st.selectbox("Select Individual", [""] + individual_options)
        
        if selected_individual:
            # Find the oldest ancestors
            oldest_ancestors = find_oldest_ancestors(df, selected_individual)
            
            if oldest_ancestors:
                st.write(f"Oldest ancestors for {selected_individual}:")
                for ancestor, weight in oldest_ancestors:
                    st.write(f"- {ancestor} (Weight: {weight:.4f})")
                
                # Plot the genetic distribution
                genetic_fig = plot_genetic_distribution(df, oldest_ancestors)
                st.pyplot(genetic_fig)
            else:
                st.warning(f"No ancestors found for {selected_individual}")
        else:
            st.info("Please select an individual to view their genetic distribution")
    else:
        st.warning("No data available for genetic distribution analysis")

