"""
Data processing functions for the Red Panda Family Tree application
"""

import csv
import pandas as pd
from datetime import datetime
from utils import escape_mermaid, clean_name, convert_date, convert_date_through, convert_date_fallback, get_foreign_zoos


def read_csv(file_path):
    """Read CSV file and return family data"""
    family_data = []
    with open(file_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['name'] = escape_mermaid(clean_name(row['name']))
            row['father'] = escape_mermaid(clean_name(row['father']))
            row['mother'] = escape_mermaid(clean_name(row['mother']))
            row['birthdate'] = row['birthdate'] if 'birthdate' in row else ''
            row['deaddate'] = row['deaddate'] if 'deaddate' in row else ''
            row['birth_zoo'] = row['birth_zoo'] if 'birth_zoo' in row else ''
            row['move_zoo1'] = row['move_zoo1'] if 'move_zoo1' in row else ''
            row['move_zoo2'] = row['move_zoo2'] if 'move_zoo2' in row else ''
            row['move_zoo3'] = row['move_zoo3'] if 'move_zoo3' in row else ''
            row['cur_zoo'] = row['cur_zoo'] if 'cur_zoo' in row else ''
            row['gender'] = row['gender'] if 'gender' in row else 'オス'
            row['image'] = row['image'] if 'image' in row else ''
            family_data.append(row)
    return family_data


def sort_children(children):
    """Sort children by birthdate and gender (males first)"""
    def sort_key(child):
        birthdate = child.get('birthdate', '')
        if isinstance(birthdate, str):
            from utils import parse_birthdate
            birthdate_parsed = parse_birthdate(birthdate)
        else:
            try:
                birthdate_parsed = birthdate.to_pydatetime() if hasattr(birthdate, 'to_pydatetime') else birthdate
            except:
                birthdate_parsed = datetime.min
        
        gender = child.get('gender', 'オス')
        return (birthdate_parsed, 0 if gender == "オス" else 1)
    
    return sorted(children, key=sort_key)


def prepare_dataframe_for_analysis(df):
    """Prepare DataFrame for analysis by cleaning names and converting dates"""
    df['father'] = df['father'].apply(clean_name)
    df['mother'] = df['mother'].apply(clean_name)
    
    # Convert dates
    df['birthdate'] = pd.to_datetime(df['birthdate'].apply(convert_date))
    df['deaddate'] = pd.to_datetime(df['deaddate'].apply(convert_date))
    
    return df


def prepare_gantt_dataframe(df):
    """Prepare DataFrame specifically for Gantt chart analysis"""
    df['father'] = df['father'].apply(clean_name)
    df['mother'] = df['mother'].apply(clean_name)
    
    # Convert all date columns
    df['birthdate'] = pd.to_datetime(df['birthdate'].apply(convert_date_fallback))
    df['deaddate'] = pd.to_datetime(df['deaddate'].apply(convert_date_fallback))
    df['move_date1'] = pd.to_datetime(df['move_date1'].apply(convert_date_through))
    df['move_date2'] = pd.to_datetime(df['move_date2'].apply(convert_date_through))
    df['move_date3'] = pd.to_datetime(df['move_date3'].apply(convert_date_through))
    
    return df


def get_individual_options(df):
    """Get list of individual names for selection"""
    return list(df['name'].unique())


def get_zoo_options(df):
    """Get list of zoo names for selection (Japan only)"""
    # Get all zoo names from various columns
    all_zoos = set(df['birth_zoo'].dropna().unique()) | 
               set(df['move_zoo1'].dropna().unique()) | 
               set(df['move_zoo2'].dropna().unique()) | 
               set(df['move_zoo3'].dropna().unique())
    
    # Exclude foreign zoos
    foreign_zoos = set(get_foreign_zoos())
    japan_zoos = all_zoos - foreign_zoos
    
    return list(japan_zoos) 