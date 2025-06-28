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
import json
from streamlit_calendar import calendar

def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def js_btoa(data):
    return base64.b64encode(data)

def pako_deflate(data):
    compress = zlib.compressobj(9, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY)
    compressed_data = compress.compress(data)
    compressed_data += compress.flush()
    return compressed_data

def mermaid_to_pako_url(graphMarkdown: str):
    jGraph = {"code": graphMarkdown, "mermaid": {"theme": "default"}}
    byteStr = json.dumps(jGraph).encode('utf-8')
    deflated = pako_deflate(byteStr)
    dEncode = js_btoa(deflated)
    link = 'http://mermaid.live/edit#pako:' + dEncode.decode('ascii')
    return link



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
        birthdate = child.get('birthdate', '')
        if isinstance(birthdate, str):
            birthdate_parsed = parse_birthdate(birthdate)
        else:
            # If birthdate is already a Timestamp or datetime object
            try:
                birthdate_parsed = birthdate.to_pydatetime() if hasattr(birthdate, 'to_pydatetime') else birthdate
            except:
                birthdate_parsed = datetime.min
        
        gender = child.get('gender', 'オス')  # デフォルトは"オス"
        return (birthdate_parsed, 0 if gender == "オス" else 1)  # オスを左に配置
    
    return sorted(children, key=sort_key)

def get_year_range(person):
    birth_year = ''
    death_year = ''
    
    # Handle birthdate
    birthdate = person.get('birthdate')
    if birthdate:
        if isinstance(birthdate, str):
            try:
                birth_year = datetime.strptime(birthdate, '%Y年%m月%d日').year
            except:
                pass
        else:
            # If birthdate is already a Timestamp or datetime object
            try:
                birth_year = birthdate.year if hasattr(birthdate, 'year') else birthdate.year
            except:
                pass
    
    # Handle deaddate
    deaddate = person.get('deaddate')
    if deaddate:
        if isinstance(deaddate, str):
            try:
                death_year = datetime.strptime(deaddate, '%Y年%m月%d日').year
            except:
                pass
        else:
            # If deaddate is already a Timestamp or datetime object
            try:
                death_year = deaddate.year if hasattr(deaddate, 'year') else deaddate.year
            except:
                pass
    
    if birth_year and death_year:
        return f"{birth_year}-{death_year}"
    elif birth_year:
        return f"{birth_year}-"
    return ""

def add_ancestors_for_root(family_data, person_name, connections, depth=0, max_depth=2):
    """指定された個体の祖先を家系図に追加"""
    if depth >= max_depth:
        return
    
    person = next((p for p in family_data if p['name'] == person_name), None)
    if not person:
        return
    
    father = person.get('father', '')
    mother = person.get('mother', '')
    
    if father or mother:
        parent_node = f"{father}_{mother}" if father and mother else father if father else mother
        connections.add(f"{parent_node}(( ))")  # 中間ノード
        connections.add(f"{parent_node} --> {person_name}")
    
    if father:
        father_data = next((p for p in family_data if p['name'] == father), {'name': father, 'gender': 'オス'})
        gender = father_data.get('gender', 'オス')
        year_range = get_year_range(father_data)
        cur_zoo = father_data.get('cur_zoo', '')
        display_text = f"{father}<br>{cur_zoo}<br>{year_range}" if year_range else f"{father}<br>{cur_zoo}"
        connections.add(f"{father}[{display_text}]:::gender_{gender}")
        connections.add(f"{father} --> {parent_node}")
        add_ancestors_for_root(family_data, father, connections, depth + 1, max_depth)
    
    if mother:
        mother_data = next((p for p in family_data if p['name'] == mother), {'name': mother, 'gender': 'メス'})
        gender = mother_data.get('gender', 'メス')
        year_range = get_year_range(mother_data)
        cur_zoo = mother_data.get('cur_zoo', '')
        display_text = f"{mother}<br>{cur_zoo}<br>{year_range}" if year_range else f"{mother}<br>{cur_zoo}"
        connections.add(f"{mother}[{display_text}]:::gender_{gender}")
        connections.add(f"{mother} --> {parent_node}")
        add_ancestors_for_root(family_data, mother, connections, depth + 1, max_depth)

def add_descendants_for_root(family_data, person_name, connections, depth=0, max_depth=2):
    """指定された個体の子孫を家系図に追加"""
    if depth >= max_depth:
        return
    
    children = [member for member in family_data if member['father'] == person_name or member['mother'] == person_name]
    sorted_children = sort_children(children)
    
    for child in sorted_children:
        child_name = child['name']
        father = child.get('father', '')
        mother = child.get('mother', '')
        
        if father and mother:
            parent_node = f"{father}_{mother}"
            connections.add(f"{parent_node}(( ))")  # 中間ノード
            father_data = next((p for p in family_data if p['name'] == father), {'name': father, 'gender': 'オス'})
            mother_data = next((p for p in family_data if p['name'] == mother), {'name': mother, 'gender': 'メス'})
            
            # 父親のノードを追加（まだ追加されていない場合）
            if not any(father in conn for conn in connections):
                gender = father_data.get('gender', 'オス')
                year_range = get_year_range(father_data)
                cur_zoo = father_data.get('cur_zoo', '')
                display_text = f"{father}<br>{cur_zoo}<br>{year_range}" if year_range else f"{father}<br>{cur_zoo}"
                connections.add(f"{father}[{display_text}]:::gender_{gender}")
            
            # 母親のノードを追加（まだ追加されていない場合）
            if not any(mother in conn for conn in connections):
                gender = mother_data.get('gender', 'メス')
                year_range = get_year_range(mother_data)
                cur_zoo = mother_data.get('cur_zoo', '')
                display_text = f"{mother}<br>{cur_zoo}<br>{year_range}" if year_range else f"{mother}<br>{cur_zoo}"
                connections.add(f"{mother}[{display_text}]:::gender_{gender}")
            
            connections.add(f"{father} --> {parent_node}")
            connections.add(f"{mother} --> {parent_node}")
        else:
            parent_node = f"{father or mother}_child"
            connections.add(f"{parent_node}(( ))")  # 中間ノード
            parent_data = next((p for p in family_data if p['name'] == (father or mother)), 
                             {'name': father or mother, 'gender': 'オス' if father else 'メス'})
            
            # 親のノードを追加（まだ追加されていない場合）
            if not any((father or mother) in conn for conn in connections):
                gender = parent_data.get('gender', 'オス' if father else 'メス')
                year_range = get_year_range(parent_data)
                cur_zoo = parent_data.get('cur_zoo', '')
                display_text = f"{father or mother}<br>{cur_zoo}<br>{year_range}" if year_range else f"{father or mother}<br>{cur_zoo}"
                connections.add(f"{(father or mother)}[{display_text}]:::gender_{gender}")
            
            connections.add(f"{(father or mother)} --> {parent_node}")
        
        # 子のノードを追加
        gender = child.get('gender', 'オス')
        year_range = get_year_range(child)
        cur_zoo = child.get('cur_zoo', '')
        display_text = f"{child_name}<br>{cur_zoo}<br>{year_range}" if year_range else f"{child_name}<br>{cur_zoo}"
        connections.add(f"{child_name}[{display_text}]:::gender_{gender}")
        connections.add(f"{parent_node} --> {child_name}")
        
        add_descendants_for_root(family_data, child_name, connections, depth + 1, max_depth)

def generate_mermaid(family_data, root_name=None, parent_depth=2, child_depth=2, show_images=False):
    mermaid_code = "graph TD;\n"
    connections = OrderedSet()
    
    family_dict = {person['name']: person for person in family_data}
    
    def get_year_range(person):
        birth_year = ''
        death_year = ''
        if person.get('birthdate'):
            try:
                birth_year = datetime.strptime(person['birthdate'], '%Y年%m月%d日').year
            except:
                pass
        if person.get('deaddate'):
            try:
                death_year = datetime.strptime(person['deaddate'], '%Y年%m月%d日').year
            except:
                pass
        if birth_year and death_year:
            return f"{birth_year}-{death_year}"
        elif birth_year:
            return f"{birth_year}-"
        return ""
    
    def add_person_node(person):
        gender = person.get('gender', 'オス')
        year_range = get_year_range(person)
        cur_zoo = person.get('cur_zoo', '')
        display_text = f"{person['name']}<br>{cur_zoo}<br>{year_range}" if year_range else f"{person['name']}<br>{cur_zoo}"
        
        if person.get('image', ''):
            img = person.get('image', '').split(',')[0]
            if show_images and is_url(img):
                return f"{person['name']}[{display_text}<br><img src=\"{img}\" />]:::gender_{gender}"
            else:
                return f"{person['name']}[{display_text}]:::gender_{gender}"
        return f"{person['name']}[{display_text}]:::gender_{gender}"
    
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
            father_data = next((p for p in family_data if p['name'] == father), {'name': father, 'gender': 'オス'})
            connections.add(f"{add_person_node(father_data)} --> {parent_node}")
            add_ancestors(father, depth + 1)
        if mother:
            mother_data = next((p for p in family_data if p['name'] == mother), {'name': mother, 'gender': 'メス'})
            connections.add(f"{add_person_node(mother_data)} --> {parent_node}")
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
                father_data = next((p for p in family_data if p['name'] == father), {'name': father, 'gender': 'オス'})
                mother_data = next((p for p in family_data if p['name'] == mother), {'name': mother, 'gender': 'メス'})
                connections.add(f"{add_person_node(father_data)} --> {parent_node}")
                connections.add(f"{add_person_node(mother_data)} --> {parent_node}")
            else:
                parent_node = f"{father or mother}_child"
                connections.add(f"{parent_node}(( ))")  # 中間ノード
                parent_data = next((p for p in family_data if p['name'] == (father or mother)), 
                                 {'name': father or mother, 'gender': 'オス' if father else 'メス'})
                connections.add(f"{add_person_node(parent_data)} --> {parent_node}")
            
            connections.add(f"{parent_node} --> {add_person_node(child)}")
            add_descendants(child_name, depth + 1)
    
    if root_name:
        add_ancestors(root_name)
        add_descendants(root_name)
    
    # スタイル定義を追加
    mermaid_code += "\nclassDef gender_オス stroke:blue,stroke-width:2px;\n"
    mermaid_code += "classDef gender_メス stroke:pink,stroke-width:2px;\n"
    
    mermaid_code += "\n".join(connections)
    return mermaid_code

def create_gantt_chart(tasks, zoo_name):
    """
    Create a Gantt chart from a list of tasks.
    Each task should be a dictionary with 'Task', 'Start', 'End', 'Status', and 'Zoo' keys.
    """
    # 個体数に応じて図の高さを動的に調整
    num_tasks = len(tasks)
    base_height = 6  # 基本の高さ
    height_per_task = 0.4  # 1個体あたりの高さ
    min_height = 4  # 最小の高さ
    max_height = 20  # 最大の高さ
    
    # 個体数に基づいて高さを計算
    calculated_height = base_height + (num_tasks * height_per_task)
    # 最小・最大値の範囲内に制限
    fig_height = max(min_height, min(calculated_height, max_height))
    
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Sort tasks by start date
    tasks = sorted(tasks, key=lambda x: x['Start'])
    
    # Create y-axis labels (task names)
    y_labels = [task['Task'] for task in tasks]
    
    # Calculate task durations
    durations = [(task['End'] - task['Start']).days for task in tasks]
    
    # Calculate start positions (days from the earliest start date)
    min_start = min(task['Start'] for task in tasks)
    start_positions = [(task['Start'] - min_start).days for task in tasks]
    
    # Define colors based on zoo
    unique_zoos = sorted(set(task['Zoo'] for task in tasks))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_zoos)))
    zoo_colors = dict(zip(unique_zoos, colors))
    
    # Create horizontal bars
    bars = ax.barh(y_labels, durations, left=start_positions, 
                  color=[zoo_colors[task['Zoo']] for task in tasks])
    
    # Add task labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(bar.get_x() + width/2, bar.get_y() + bar.get_height()/2,
                f"{tasks[i]['Zoo']}", ha='center', va='center')
    
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
    ax.set_title(f'Zoo Transitions for {zoo_name}')
    
    # Add legend for zoos
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color) for color in colors]
    ax.legend(legend_elements, unique_zoos, title='Zoos', loc='center left', bbox_to_anchor=(1.0, 0.5))
    
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

tr, ppy, gantt, genetic, death_age, relationship, birthday = st.tabs(["Family Tree", "Population Pyramid", "Gantt Chart", "Genetic Distribution", "Death Age Histogram", "Relationship Analysis", "Birthday Calendar"])
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
    def convert_date_through(date_str):
        try:
          if type(date_str) is str:
            return datetime.strptime(date_str, '%Y年%m月%d日').date()
          else:
            return None
        except ValueError:
          return None
    
    live_df['birthdate'] = pd.to_datetime(live_df['birthdate'].apply(convert_date))
    live_df['deaddate'] = pd.to_datetime(live_df['deaddate'].apply(convert_date))
    live_df['move_date1'] = pd.to_datetime(live_df['move_date1'].apply(convert_date_through))
    live_df['move_date2'] = pd.to_datetime(live_df['move_date2'].apply(convert_date_through))
    live_df['move_date3'] = pd.to_datetime(live_df['move_date3'].apply(convert_date_through))
    dead_df['birthdate'] = pd.to_datetime(dead_df['birthdate'].apply(convert_date))
    dead_df['deaddate'] = pd.to_datetime(dead_df['deaddate'].apply(convert_date))
    dead_df['move_date1'] = pd.to_datetime(dead_df['move_date1'].apply(convert_date_through))
    dead_df['move_date2'] = pd.to_datetime(dead_df['move_date2'].apply(convert_date_through))
    dead_df['move_date3'] = pd.to_datetime(dead_df['move_date3'].apply(convert_date_through))

    # 2003年1月1日以降のデータのみ抽出
    live_df = live_df[live_df['birthdate'] >= pd.Timestamp('2003-01-01')]
    
    df = pd.concat([live_df, dead_df])
    df['father'] = df['father'].apply(clean_name)
    df['mother'] = df['mother'].apply(clean_name)
    zoo_options = list(set(df['birth_zoo'].unique()) | set(df['move_zoo1'].dropna().unique()) | 
                      set(df['move_zoo2'].dropna().unique()) | set(df['move_zoo3'].dropna().unique()))
    zoo_name = st.selectbox("Zoo Name", [""] + sorted(zoo_options))

    if zoo_name:
        # Convert DataFrame to tasks format with zoo transitions
        tasks = []
        for _, row in df.iterrows():
            if pd.notna(row['birthdate']):
                # Check birth zoo period
                if row['birth_zoo'] == zoo_name:
                    start_date = row['birthdate'].date()
                    end_date = row['move_date1'].date() if pd.notna(row['move_date1']) else row['deaddate'].date()
                    exist_live = (live_df['name'] == row['name']).any()
                    if exist_live:
                        to_zoo  = row['move_zoo1'] if pd.notna(row['move_zoo1']) else 'live'
                    else:
                        to_zoo  = row['move_zoo1'] if pd.notna(row['move_zoo1']) else 'dead'
                    tasks.append({
                        'Task': row['name'],
                        'Start': start_date,
                        'End': end_date,
                        'Zoo': to_zoo
                    })
                
                # Check first move period
                if pd.notna(row['move_date1']) and pd.notna(row['move_zoo1']) and row['move_zoo1'] == zoo_name:
                    start_date = row['move_date1'].date()
                    end_date = row['move_date2'].date() if pd.notna(row['move_date2']) else row['deaddate'].date()
                    exist_live = (live_df['name'] == row['name']).any()
                    if exist_live:
                        to_zoo  = row['move_zoo2'] if pd.notna(row['move_zoo2']) else 'live'
                    else:
                        to_zoo  = row['move_zoo2'] if pd.notna(row['move_zoo2']) else 'dead'
                    tasks.append({
                        'Task': row['name'],
                        'Start': start_date,
                        'End': end_date,
                        'Zoo': to_zoo
                    })
                
                # Check second move period
                if pd.notna(row['move_date2']) and pd.notna(row['move_zoo2']) and row['move_zoo2'] == zoo_name:
                    start_date = row['move_date2'].date()
                    end_date = row['move_date3'].date() if pd.notna(row['move_date3']) else row['deaddate'].date()
                    exist_live = (live_df['name'] == row['name']).any()
                    if exist_live:
                        to_zoo  = row['move_zoo3'] if pd.notna(row['move_zoo3']) else 'live'
                    else:
                        to_zoo  = row['move_zoo3'] if pd.notna(row['move_zoo3']) else 'dead'
                    tasks.append({
                        'Task': row['name'],
                        'Start': start_date,
                        'End': end_date,
                        'Zoo': to_zoo
                    })
                
                # Check third move period
                if pd.notna(row['move_date3']) and pd.notna(row['move_zoo3']) and row['move_zoo3'] == zoo_name:
                    start_date = row['move_date3'].date()
                    end_date = row['deaddate'].date()
                    tasks.append({
                        'Task': row['name'],
                        'Start': start_date,
                        'End': end_date,
                        'Zoo': row['move_zoo3']
                    })
        
        if tasks:
            # Create and display the Gantt chart
            gantt_fig = create_gantt_chart(tasks, zoo_name)
            st.pyplot(gantt_fig)
            
            # Add explanation
            st.markdown("""
            ### Gantt Chart Explanation
            
            This Gantt chart shows the periods when individuals were at the selected zoo:
            
            - **Tasks**: Individual names (y-axis)
            - **Timeline**: Periods when individuals were at the selected zoo (x-axis)
            - **Segments**: Each segment represents a continuous period at the selected zoo
            
            The chart visualizes when each individual was present at the selected zoo, whether they were born there or moved there later in life.
            """)
            
            # 家系図の作成
            st.write("### Family Tree of Individuals Born at Selected Zoo")
            
            # 指定された動物園で生まれた個体を取得
            born_at_zoo = df[df['birth_zoo'] == zoo_name]['name'].unique().tolist()
            
            if born_at_zoo:
                # 家系図に含める個体を収集
                family_members = set()
                
                # 指定された動物園で生まれた個体を追加
                family_members.update(born_at_zoo)
                
                # 各個体の親と子を追加
                for individual in born_at_zoo:
                    # 個体の情報を取得
                    individual_data = df[df['name'] == individual]
                    if not individual_data.empty:
                        row = individual_data.iloc[0]
                        
                        # 親を追加
                        if pd.notna(row['father']) and row['father'] != '':
                            family_members.add(row['father'])
                        if pd.notna(row['mother']) and row['mother'] != '':
                            family_members.add(row['mother'])
                        
                        # 子を追加
                        children = df[(df['father'] == individual) | (df['mother'] == individual)]
                        for _, child in children.iterrows():
                            family_members.add(child['name'])
                
                # 家系図データを作成
                family_data = []
                for member in family_members:
                    member_data = df[df['name'] == member]
                    if not member_data.empty:
                        row = member_data.iloc[0]
                        # live_dfに含まれる個体（生存している個体）の場合はdeaddateを削除
                        deaddate = None if member in live_df['name'].values else row['deaddate']
                        family_data.append({
                            'name': row['name'],
                            'father': row['father'] if pd.notna(row['father']) else '',
                            'mother': row['mother'] if pd.notna(row['mother']) else '',
                            'gender': row['gender'],
                            'birthdate': row['birthdate'],
                            'deaddate': deaddate,
                            'cur_zoo': row['cur_zoo'],
                            'image': row.get('image', '')
                        })
                
                # 家系図を生成
                if family_data:
                    # 指定された動物園で生まれた全ての個体をルートとして家系図を生成
                    # 複数のルート個体を処理するためのカスタムMermaidコード生成
                    mermaid_code = "graph TD;\n"
                    connections = OrderedSet()
                    
                    # 各ルート個体について家系図を生成
                    for root_individual in born_at_zoo:
                        # ルート個体の情報を取得
                        root_data = next((p for p in family_data if p['name'] == root_individual), None)
                        if root_data:
                            # ルート個体のノードを追加（特別なスタイルで）
                            gender = root_data.get('gender', 'オス')
                            year_range = get_year_range(root_data)
                            cur_zoo = root_data.get('cur_zoo', '')
                            display_text = f"{root_data['name']}<br>{cur_zoo}<br>{year_range}" if year_range else f"{root_data['name']}<br>{cur_zoo}"
                            connections.add(f"{root_data['name']}[{display_text}]:::root_{gender}")
                            
                            # 親を追加（2世代）
                            add_ancestors_for_root(family_data, root_individual, connections, depth=0, max_depth=2)
                            # 子を追加（2世代）
                            add_descendants_for_root(family_data, root_individual, connections, depth=0, max_depth=2)
                    
                    # スタイル定義を追加
                    mermaid_code += "\nclassDef gender_オス stroke:blue,stroke-width:2px;\n"
                    mermaid_code += "classDef gender_メス stroke:pink,stroke-width:2px;\n"
                    mermaid_code += "classDef root_オス stroke:blue,stroke-width:4px,fill:#e6f3ff;\n"
                    mermaid_code += "classDef root_メス stroke:pink,stroke-width:4px,fill:#ffe6f3;\n"
                    
                    mermaid_code += "\n".join(connections)
                    
                    # 家系図の説明を追加
                    st.write(f"**Unified family tree showing all individuals born at {zoo_name} and their family members:**")
                    st.write(f"- Individuals born at {zoo_name}: {', '.join(born_at_zoo)}")
                    st.write(f"- Total family members shown: {len(family_members)}")
                    st.write(f"- Root individuals (highlighted): {', '.join(born_at_zoo)}")
                    
                    # 家系図を表示
                    st_mermaid(mermaid_code, key=f"gantt_family_tree_{zoo_name}")
                    
                    # Mermaid Live Editor へのリンク
                    url = mermaid_to_pako_url(mermaid_code)
                    st.markdown(f"[Mermaid Live Editor で開く]({url})", unsafe_allow_html=True)
                else:
                    st.warning("No family data available for the selected individuals.")
            else:
                st.warning(f"No individuals were born at {zoo_name}.")
        else:
            st.warning(f"No individuals found who were at {zoo_name} at any point in their lives.")
    else:
        st.warning("Please select a zoo to view the Gantt chart.")

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

with death_age:
    st.title("Death Age Distribution")
    
    # CSVファイルの読み込み
    if use_default and os.path.exists(default_csv_path):
        df = pd.read_csv(default_csv_path)
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    
    # 日付の変換とデータのクレンジング
    def convert_date(date_str):
        try:
            if type(date_str) is str:
                return datetime.strptime(date_str, '%Y年%m月%d日').date()
            else:
                return None
        except ValueError:
            return None
    
    # 死亡日と誕生日を日付型に変換
    df['birthdate'] = pd.to_datetime(df['birthdate'].apply(convert_date))
    df['deaddate'] = pd.to_datetime(df['deaddate'].apply(convert_date))
    
    # 死亡日が存在するデータのみを抽出
    death_df = df[df['deaddate'].notna()].copy()
    
    if not death_df.empty:
        # 死亡時の年齢を計算
        death_df['death_age'] = (death_df['deaddate'] - death_df['birthdate']).dt.days / 365.25
        
        # ヒストグラムの作成
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 年齢の範囲を設定（0歳から最大年齢まで）
        max_age = int(death_df['death_age'].max()) + 1
        bins = range(0, max_age + 1)
        
        # 性別ごとにヒストグラムを作成
        male_data = death_df[death_df['gender'] == 'オス']['death_age']
        female_data = death_df[death_df['gender'] == 'メス']['death_age']
        
        ax.hist([male_data, female_data], bins=bins, alpha=0.7, label=['Male', 'Female'], color=['skyblue', 'lightpink'])
        
        # グラフの装飾
        ax.set_xlabel('Age at Death (years)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Death Ages')
        ax.legend()
        ax.grid(True)
        
        # 統計情報の表示
        st.write("### Statistics")
        st.write(f"Total number of deaths: {len(death_df)}")
        st.write(f"Average age at death: {death_df['death_age'].mean():.1f} years")
        st.write(f"Median age at death: {death_df['death_age'].median():.1f} years")
        st.write(f"Maximum age at death: {death_df['death_age'].max():.1f} years")
        
        # 性別ごとの統計情報
        st.write("### Statistics by Gender")
        male_stats = death_df[death_df['gender'] == 'オス']['death_age']
        female_stats = death_df[death_df['gender'] == 'メス']['death_age']
        
        st.write("**Male**")
        st.write(f"- Count: {len(male_stats)}")
        st.write(f"- Average age: {male_stats.mean():.1f} years")
        st.write(f"- Median age: {male_stats.median():.1f} years")
        
        st.write("**Female**")
        st.write(f"- Count: {len(female_stats)}")
        st.write(f"- Average age: {female_stats.mean():.1f} years")
        st.write(f"- Median age: {female_stats.median():.1f} years")
        
        # ヒストグラムの表示
        st.pyplot(fig)

        # 平均余命の計算とグラフ作成
        st.write("### Life Expectancy by Current Age")
        
        # 各年齢での平均余命を計算
        life_expectancy = []
        current_ages = range(0, int(death_df['death_age'].max()) + 1)
        
        for age in current_ages:
            # その年齢以上で死亡した個体を抽出
            survived = death_df[death_df['death_age'] >= age]
            if len(survived) > 0:
                # 平均余命 = 平均死亡年齢 - 現在の年齢
                avg_life_expectancy = survived['death_age'].mean() - age
                life_expectancy.append(avg_life_expectancy)
            else:
                life_expectancy.append(0)
        
        # 性別ごとの平均余命を計算
        male_life_expectancy = []
        female_life_expectancy = []
        
        for age in current_ages:
            male_survived = death_df[(death_df['death_age'] >= age) & (death_df['gender'] == 'オス')]
            female_survived = death_df[(death_df['death_age'] >= age) & (death_df['gender'] == 'メス')]
            
            if len(male_survived) > 0:
                male_life_expectancy.append(male_survived['death_age'].mean() - age)
            else:
                male_life_expectancy.append(0)
                
            if len(female_survived) > 0:
                female_life_expectancy.append(female_survived['death_age'].mean() - age)
            else:
                female_life_expectancy.append(0)
        
        # 平均余命のグラフを作成
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        ax2.plot(current_ages, life_expectancy, label='Overall', color='green', linewidth=2)
        ax2.plot(current_ages, male_life_expectancy, label='Male', color='skyblue', linestyle='--')
        ax2.plot(current_ages, female_life_expectancy, label='Female', color='lightpink', linestyle='--')
        
        ax2.set_xlabel('Current Age (years)')
        ax2.set_ylabel('Life Expectancy (years)')
        ax2.set_title('Life Expectancy by Current Age')
        ax2.legend()
        ax2.grid(True)
        
        # グラフの表示
        st.pyplot(fig2)
        
        # 平均余命の説明
        st.write("""
        #### About Life Expectancy
        - The graph shows the average remaining years of life for individuals at each age
        - The overall line (green) shows the average for all individuals
        - The dashed lines show the averages for males (blue) and females (pink)
        - The life expectancy decreases as current age increases
        - This is based on historical death data and may not predict future life expectancy
        """)

        # 死亡月の分布を表示
        st.write("### Death Month Distribution")
        
        # 死亡月を抽出
        death_df['death_month'] = death_df['deaddate'].dt.month
        
        # ヒストグラムの作成
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        # 性別ごとにヒストグラムを作成
        male_month_data = death_df[death_df['gender'] == 'オス']['death_month']
        female_month_data = death_df[death_df['gender'] == 'メス']['death_month']
        
        ax3.hist([male_month_data, female_month_data], bins=range(1, 14), alpha=0.7, 
                label=['Male', 'Female'], color=['skyblue', 'lightpink'])
        
        # グラフの装飾
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Number of Deaths')
        ax3.set_title('Distribution of Deaths by Month')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax3.legend()
        ax3.grid(True)
        
        # グラフの表示
        st.pyplot(fig3)
        
        # 月ごとの統計情報
        st.write("### Death Statistics by Month")
        
        # 全体的な月別死亡数
        monthly_deaths = death_df['death_month'].value_counts().sort_index()
        st.write("**Total deaths by month:**")
        for month, count in monthly_deaths.items():
            month_name = ['January', 'February', 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October', 'November', 'December'][month-1]
            st.write(f"- {month_name}: {count} deaths")
        
        # 性別ごとの月別死亡数
        st.write("**Deaths by month and gender:**")
        for month in range(1, 13):
            month_name = ['January', 'February', 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October', 'November', 'December'][month-1]
            male_count = len(death_df[(death_df['death_month'] == month) & (death_df['gender'] == 'オス')])
            female_count = len(death_df[(death_df['death_month'] == month) & (death_df['gender'] == 'メス')])
            st.write(f"- {month_name}:")
            st.write(f"  - Male: {male_count} deaths")
            st.write(f"  - Female: {female_count} deaths")
    else:
        st.warning("No death records found in the dataset.")

with relationship:
    st.title("Relationship Analysis Between Two Individuals")
    
    if data:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        if df.shape[0] > 0:
            # 個体名のリストを取得
            individual_options = list(df['name'].unique())
            
            # 2つの個体を選択
            col1, col2 = st.columns(2)
            with col1:
                individual1 = st.selectbox("Select First Individual", [""] + individual_options)
            with col2:
                individual2 = st.selectbox("Select Second Individual", [""] + individual_options)
            
            if individual1 and individual2:
                if individual1 == individual2:
                    st.warning("Please select two different individuals.")
                else:
                    # 関係を分析する関数
                    def find_relationship_path(df, start_name, target_name):
                        # キューと訪問済みセットを初期化
                        queue = [(start_name, [start_name])]  # (現在のノード, パス)
                        visited = {start_name}
                        
                        while queue:
                            current_name, path = queue.pop(0)  # キューから最初の要素を取り出す
                            
                            # 目標の個体に到達した場合
                            if current_name == target_name:
                                return path
                            
                            # 現在の個体の情報を取得
                            current = df[df['name'] == current_name].iloc[0]
                            
                            # 1. 親を探索（祖先方向）
                            if pd.notna(current['father']) and current['father'] != '':
                                father = current['father']
                                if father in df['name'].values and father not in visited:
                                    visited.add(father)
                                    queue.append((father, path + [father]))
                            
                            if pd.notna(current['mother']) and current['mother'] != '':
                                mother = current['mother']
                                if mother in df['name'].values and mother not in visited:
                                    visited.add(mother)
                                    queue.append((mother, path + [mother]))
                            
                            # 2. 子を探索（子孫方向）
                            children = df[(df['father'] == current_name) | (df['mother'] == current_name)]
                            for _, child in children.iterrows():
                                if child['name'] not in visited:
                                    visited.add(child['name'])
                                    queue.append((child['name'], path + [child['name']]))
                            
                            # 3. 兄弟姉妹を探索
                            if pd.notna(current['father']) and current['father'] != '':
                                siblings = df[df['father'] == current['father']]
                                for _, sibling in siblings.iterrows():
                                    if sibling['name'] != current_name and sibling['name'] not in visited:
                                        visited.add(sibling['name'])
                                        queue.append((sibling['name'], path + [sibling['name']]))
                            
                            if pd.notna(current['mother']) and current['mother'] != '':
                                siblings = df[df['mother'] == current['mother']]
                                for _, sibling in siblings.iterrows():
                                    if sibling['name'] != current_name and sibling['name'] not in visited:
                                        visited.add(sibling['name'])
                                        queue.append((sibling['name'], path + [sibling['name']]))
                        
                        return None
                    
                    # 関係を探索
                    relationship_path = find_relationship_path(df, individual1, individual2)
                    
                    
                    if relationship_path:
                        st.success(f"Relationship found between {individual1} and {individual2}!")
                        
                        # 関係を図示
                        mermaid_code = "graph TD;\n"
                        
                        # ノードの追加
                        for i, name in enumerate(relationship_path):
                            if i == 0:
                                mermaid_code += f"{name}[{name}]:::start;\n"
                            elif i == len(relationship_path) - 1:
                                mermaid_code += f"{name}[{name}]:::target;\n"
                            else:
                                mermaid_code += f"{name}[{name}];\n"
                        
                        # エッジの追加
                        for i in range(len(relationship_path) - 1):
                            current = df[df['name'] == relationship_path[i]].iloc[0]
                            next_person = relationship_path[i + 1]
                            next_person_data = df[df['name'] == next_person].iloc[0]
                            
                            # 親子関係の判定
                            if current['father'] == next_person:
                                mermaid_code += f"{relationship_path[i]} -->|father| {next_person};\n"
                            elif current['mother'] == next_person:
                                mermaid_code += f"{relationship_path[i]} -->|mother| {next_person};\n"
                            elif next_person_data['father'] == relationship_path[i]:
                                if next_person_data['gender'] == 'オス':
                                    mermaid_code += f"{relationship_path[i]} -->|son| {next_person};\n"
                                else:
                                    mermaid_code += f"{relationship_path[i]} -->|daughter| {next_person};\n"
                            elif next_person_data['mother'] == relationship_path[i]:
                                if next_person_data['gender'] == 'オス':
                                    mermaid_code += f"{relationship_path[i]} -->|son| {next_person};\n"
                                else:
                                    mermaid_code += f"{relationship_path[i]} -->|daughter| {next_person};\n"
                            else:
                                # 兄弟姉妹関係の場合
                                if current['father'] == next_person_data['father'] or current['mother'] == next_person_data['mother']:
                                    mermaid_code += f"{relationship_path[i]} ---|sibling| {next_person};\n"
                        
                        # スタイルの追加
                        mermaid_code += "\nclassDef start fill:#f9f,stroke:#333,stroke-width:2px;\n"
                        mermaid_code += "classDef target fill:#bbf,stroke:#333,stroke-width:2px;\n"
                        
                        # 図の表示
                        st_mermaid(mermaid_code)
                        
                        # 関係の説明
                        st.write("### Relationship Path")
                        for i in range(len(relationship_path) - 1):
                            current = df[df['name'] == relationship_path[i]].iloc[0]
                            next_person = relationship_path[i + 1]
                            next_person_data = df[df['name'] == next_person].iloc[0]
                            
                            if current['father'] == next_person:
                                st.write(f"{relationship_path[i]}'s father is {next_person}")
                            elif current['mother'] == next_person:
                                st.write(f"{relationship_path[i]}'s mother is {next_person}")
                            elif next_person_data['father'] == relationship_path[i]:
                                if current['gender'] == 'オス':
                                    st.write(f"{next_person} is {relationship_path[i]}'s son")
                                else:
                                    st.write(f"{next_person} is {relationship_path[i]}'s daughter")
                            elif next_person_data['mother'] == relationship_path[i]:
                                if current['gender'] == 'オス':
                                    st.write(f"{next_person} is {relationship_path[i]}'s son")
                                else:
                                    st.write(f"{next_person} is {relationship_path[i]}'s daughter")
                            else:
                                # 兄弟姉妹関係の場合
                                if current['father'] == next_person_data['father'] or current['mother'] == next_person_data['mother']:
                                    st.write(f"{relationship_path[i]} and {next_person} are siblings")
                    else:
                        st.warning(f"No relationship found between {individual1} and {individual2}.")
            else:
                st.info("Please select two individuals to analyze their relationship.")
        else:
            st.warning("No data available for relationship analysis.")

with birthday:
    st.title("Birthday Calendar")
    
    # CSVファイルの読み込み
    if use_default and os.path.exists(default_csv_path):
        df = pd.read_csv(default_csv_path)
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    
    # 日付の変換とデータのクレンジング
    def convert_date(date_str):
        try:
            if type(date_str) is str:
                return datetime.strptime(date_str, '%Y年%m月%d日').date()
            else:
                return None
        except ValueError:
            return None
    
    # 誕生日を日付型に変換
    df['birthdate'] = pd.to_datetime(df['birthdate'].apply(convert_date))
    
    # 生存している個体のみを抽出（deaddateがnullまたは空の個体）
    live_df = df[df['deaddate'].isna()].copy()
    
    # 日本以外の動物園に所属している個体を除外
    foreign_zoos = ['中国', '台湾', 'カナダ', 'アメリカ', 'チリ', '韓国', 'インドネシア', 'アルゼンチン', 'タイ', 'メキシコ']
    japan_df = live_df[~live_df['cur_zoo'].isin(foreign_zoos)].copy()
    
    # 現在の年の誕生日イベントを作成
    current_year = datetime.now().year
    events = []
    
    for _, row in japan_df.iterrows():
        if pd.notna(row['birthdate']):
            # 誕生日の月と日を取得
            birth_month = row['birthdate'].month
            birth_day = row['birthdate'].day
            
            # 現在の年の誕生日の日付を作成
            birthday_date = datetime(current_year, birth_month, birth_day).strftime('%Y-%m-%d')
            
            # イベントを作成
            event = {
                'title': f"{row['name']}",
                'start': birthday_date,
                'end': birthday_date,
                'color': '#FF9999'  # ピンク系の色
            }
            events.append(event)
    
    # カレンダーの表示設定
    calendar_options = {
        "headerToolbar": {
            "left": "prev,next today",
            "center": "title",
            "right": "dayGridMonth"
        },
        "initialView": "dayGridMonth",
        "locale": "ja",
        "height": "auto"
    }
    
    # カレンダーを表示
    calendar_events = calendar(events=events, options=calendar_options, key="birthday_calendar")
    #st.write(calendar_events)
    
    # 今月の誕生日一覧を表示
    current_month = datetime.now().month
    current_month_birthdays = japan_df[japan_df['birthdate'].dt.month == current_month].copy()
    # 日付でソート（日のみでソート）
    current_month_birthdays['birth_day'] = current_month_birthdays['birthdate'].dt.day
    current_month_birthdays = current_month_birthdays.sort_values('birth_day')
    
    if not current_month_birthdays.empty:
        st.write(f"### {current_month}月の誕生日一覧")
        for _, row in current_month_birthdays.iterrows():
            st.write(f"- {row['name']}: {row['birthdate'].strftime('%m月%d日')}")
    else:
        st.write(f"### {current_month}月の誕生日はありません")

