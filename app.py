import csv
import re
import streamlit as st
from datetime import datetime
import os
from streamlit_mermaid import st_mermaid
from collections import OrderedDict

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
    return re.sub(r'\s*\(.*?\).*', '', name) if name else ''

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
            if show_images:
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

st.title("Family Tree Generator")

default_csv_path = "redpanda.csv"
use_default = st.checkbox("Use default CSV file (family_data.csv in the same folder)", value=True)
show_images = st.checkbox("Show Images", value=False)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
#root_name = st.text_input("Root Name", "ケンシン")
parent_depth = st.number_input("Parent Generation Depth", min_value=1, value=2)
child_depth = st.number_input("Child Generation Depth", min_value=1, value=2)

data = None
if use_default and os.path.exists(default_csv_path):
    data = read_csv(default_csv_path)
elif uploaded_file is not None:
    data = read_csv(uploaded_file)

if data:
    name_options = [person['name'] for person in data]
    root_name = st.selectbox("Root Name", [""] + name_options)
    mermaid_code = generate_mermaid(data, root_name if root_name else None, parent_depth, child_depth, show_images)
    st.text_area("Generated Mermaid Code", mermaid_code, height=300)
    st_mermaid(mermaid_code)

