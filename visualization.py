"""
Visualization functions for the Red Panda Family Tree application
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from utils import get_year_range, OrderedSet, is_url


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


def plot_distance_distribution(distances):
    """距離分布のヒストグラムを描画する"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist([d[1] for d in distances], bins=20, color='gray', alpha=0.7)
    ax.set_xlabel('Genetic Distance')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Genetic Distances')
    return fig


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
    from data_processing import sort_children
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
    """Generate Mermaid diagram code for family tree"""
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
        from data_processing import sort_children
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