"""
Utility functions for the Red Panda Family Tree application
"""

import re
import zlib
import base64
import json
import pandas as pd
from datetime import datetime, date
from urllib.parse import urlparse
from collections import OrderedDict


def is_url(string):
    """Check if a string is a valid URL"""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def js_btoa(data):
    """Base64 encode data"""
    return base64.b64encode(data)


def pako_deflate(data):
    """Compress data using zlib"""
    compress = zlib.compressobj(9, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY)
    compressed_data = compress.compress(data)
    compressed_data += compress.flush()
    return compressed_data


def mermaid_to_pako_url(graphMarkdown: str):
    """Convert Mermaid code to pako URL for Mermaid Live Editor"""
    jGraph = {"code": graphMarkdown, "mermaid": {"theme": "default"}}
    byteStr = json.dumps(jGraph).encode('utf-8')
    deflated = pako_deflate(byteStr)
    dEncode = js_btoa(deflated)
    link = 'http://mermaid.live/edit#pako:' + dEncode.decode('ascii')
    return link


class OrderedSet:
    """Ordered set implementation using OrderedDict"""
    
    def __init__(self):
        self.data = OrderedDict()

    def add(self, item):
        self.data[item] = None

    def __contains__(self, item):
        return item in self.data

    def __iter__(self):
        return iter(self.data.keys())

    def __repr__(self):
        return f"OrderedSet({list(self.data.keys())})"


def clean_name(name):
    """Clean and normalize individual names"""
    if pd.isna(name) or name is None:
        return ''
    if isinstance(name, float):
        return ''
    return re.sub(r'\s*\(.*?\).*', '', str(name)) if name else ''


def escape_mermaid(name):
    """Escape special characters for Mermaid diagrams"""
    return name.replace("-", "\\-").replace(" ", "\\ ").replace("(", "\\(").replace(")", "\\)")


def parse_birthdate(birthdate):
    """Parse birthdate string in Japanese format"""
    match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', birthdate)
    if match:
        year, month, day = map(int, match.groups())
        return datetime(year, month, day)
    return datetime.min


def convert_date(date_str):
    """Convert date string to datetime object"""
    try:
        if type(date_str) is str:
            return datetime.strptime(date_str, '%Y年%m月%d日').date()
        else:
            return None
    except ValueError:
        return None

def convert_date_fallback(date_str):
    """Convert date string to datetime object with fallback to today"""
    try:
        if type(date_str) is str:
            return datetime.strptime(date_str, '%Y年%m月%d日').date()
        else:
            return date.today()
    except ValueError:
        return date(1980, 1, 1)

def convert_date_through(date_str):
    """Convert date string with fallback to today"""
    try:
        if type(date_str) is str:
            return datetime.strptime(date_str, '%Y年%m月%d日').date()
        else:
            return None
    except ValueError:
        return None


def get_year_range(person):
    """Get year range string for a person (birth-death)"""
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
            try:
                death_year = deaddate.year if hasattr(deaddate, 'year') else deaddate.year
            except:
                pass
    
    if birth_year and death_year:
        return f"{birth_year}-{death_year}"
    elif birth_year:
        return f"{birth_year}-"
    return ""


def get_foreign_zoos():
    """Get list of foreign zoos to exclude from Japan-based analysis"""
    return ['中国', '台湾', 'カナダ', 'アメリカ', 'チリ', '韓国', 'インドネシア', 'アルゼンチン', 'タイ', 'メキシコ']


def filter_japan_living_individuals(df):
    """Filter DataFrame to only include living individuals in Japan"""
    # 生存個体のみ抽出
    live_df = df[df['deaddate'].isna()].copy()
    if live_df.empty:
        return None
    
    # 日本以外の動物園に所属している個体を除外
    foreign_zoos = get_foreign_zoos()
    japan_df = live_df[~live_df['cur_zoo'].isin(foreign_zoos)].copy()
    
    if japan_df.empty:
        return None
    
    # 年齢計算のための日付変換
    japan_df['birthdate'] = pd.to_datetime(japan_df['birthdate'].apply(convert_date))
    # 現在の年齢を計算
    today = pd.Timestamp.now()
    japan_df['age'] = (today - japan_df['birthdate']).dt.days // 365
    
    return japan_df 