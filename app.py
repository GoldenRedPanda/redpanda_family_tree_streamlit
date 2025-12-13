import streamlit as st
import numpy as np
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import os
from streamlit_mermaid import st_mermaid
from streamlit_calendar import calendar
import folium
from streamlit_folium import st_folium

# Import our custom modules
from utils import (
    mermaid_to_pako_url, clean_name, convert_date, convert_date_through,
    get_year_range, get_foreign_zoos, filter_japan_living_individuals, OrderedSet
)
from data_processing import (
    read_csv, sort_children, prepare_dataframe_for_analysis,
    prepare_gantt_dataframe, get_individual_options, get_zoo_options
)
from genetic_analysis import (
    find_oldest_ancestors, get_gene_vector, calculate_genetic_distances,
    get_opposite_gender_candidates, prepare_genetic_analysis_data
)
from visualization import (
    plot_genetic_distribution, plot_distance_distribution, create_gantt_chart,
    add_ancestors_for_root, add_descendants_for_root, generate_mermaid
)

# Zoo coordinates for map visualization
def get_zoo_coordinates():
    """日本の主要動物園の座標を返す"""
    zoo_coords = {
        '上野動物園': [35.7167, 139.7714],
        '多摩動物公園': [35.6492, 139.2758],
        '井の頭自然文化園': [35.7000, 139.5833],
        '葛西臨海水族園': [35.6447, 139.8633],
        '神奈川県立生命の星・地球博物館': [35.2833, 139.1167],
        '横浜市立金沢動物園': [35.3833, 139.6167],
        '千葉市動物公園': [35.6167, 140.1167],
        '市川市動植物園': [35.7167, 139.9333],
        '東武動物公園': [36.0333, 139.7167],
        '群馬サファリパーク': [36.3167, 139.0167],
        'こども動物自然公園': [35.9500, 139.3833],
        'さいたま市大宮公園小動物園': [35.9167, 139.6333],
        '千葉県立房総のむら': [35.7167, 140.2167],
        '市原ぞうの国': [35.4833, 140.1167],
        '新潟市水族館マリンピア日本海': [37.9333, 139.0333],
        '富山市ファミリーパーク': [36.7000, 137.2167],
        'いしかわ動物園': [36.5500, 136.6500],
        '福井県立恐竜博物館': [35.8167, 136.4833],
        '山梨県立富士湧水の里水族館': [35.4833, 138.8000],
        '茶臼山動物園': [36.6500, 138.1833],
        '日本平動物園': [34.9833, 138.4167],
        '浜松市動物園': [34.7167, 137.7333],
        '東山動植物園': [35.1667, 136.9500],
        'のんほいパーク': [34.7667, 137.3833],
        '三重県立みえこどもの城': [34.7167, 136.5167],
        '滋賀県立琵琶湖博物館': [35.0667, 135.8667],
        '京都市動物園': [35.0167, 135.7833],
        '天王寺動物園': [34.6833, 135.5167],
        '王子動物園': [34.7167, 135.1833],
        '姫路セントラルパーク': [34.8500, 134.7000],
        '和歌山県立自然博物館': [34.2333, 135.1667],
        '奈良県立橿原考古学研究所附属博物館': [34.4833, 135.7833],
        '鳥取市立鳥取砂丘こどもの国': [35.5333, 134.2333],
        '島根県立宍道湖自然館ゴビウス': [35.4667, 133.0167],
        '岡山市半田山植物園': [34.6500, 133.9167],
        '安佐動物公園': [34.4667, 132.4500],
        '山口県立山口博物館': [34.1833, 131.4667],
        '徳島県立博物館': [34.0667, 134.5500],
        '香川県立ミュージアム': [34.3333, 134.0500],
        'とべ動物園': [33.8167, 132.7667],
        'のいち動物公園': [33.5500, 133.5333],
        '福岡市動物園': [33.5833, 130.3833],
        '北九州市立到津の森公園': [33.8833, 130.8833],
        '佐賀県立宇宙科学館': [33.2667, 130.3000],
        '長崎県立対馬歴史民俗資料館': [34.2000, 129.2833],
        '熊本市動植物園': [32.8000, 130.7167],
        '大分県立美術館': [33.2333, 131.6000],
        '宮崎県立美術館': [31.9167, 131.4167],
        '平川動物公園': [31.5833, 130.5500],
        '沖縄県立博物館・美術館': [26.2167, 127.6833],
        '沖縄こどもの国': [26.4333, 127.8000],
        '円山動物園': [43.0500, 141.3167],
        '旭山動物園': [43.7667, 142.4833],
        '釧路市動物園': [43.0000, 144.3833],
        '大森山動物園': [39.7500, 140.7167],
        'かみね動物園': [36.3667, 140.4667],
        '西山動物園': [35.9500, 136.1833],
        '神戸どうぶつ王国': [34.6500, 135.1833],
        '那須どうぶつ王国': [36.9333, 140.0167],
        '那須ワールドモンキーパーク': [36.8500, 140.1000],
        'とくしま動物園': [34.0667, 134.5500],
        'アドベンチャーワールド': [33.6667, 135.3667],
        '野毛山動物園': [35.4500, 139.6167],
        'ズーラシア': [35.5167, 139.5167],
        'ネオパークオキナワ': [26.4333, 127.8000],
        'フェニックス自然動物園': [31.9167, 131.4167],
        '九十九島動植物園': [33.2667, 129.8500],
        '八景島シーパラダイス': [35.3833, 139.6333],
        '伊豆シャボテン動物公園': [34.7833, 138.7833],
        '八木山動物公園': [38.2500, 140.8833],
        '到津の森公園': [33.8833, 130.8833],
        '夢見ヶ崎動物公園': [35.5500, 139.7167],
        '大牟田市動物園': [33.0333, 130.4500],
        '大島公園動物園': [34.7500, 139.3667],
        '富士サファリパーク': [35.3167, 138.8167],
        '姫路市立動物園': [34.8500, 134.7000],
        '東北サファリパーク': [37.4333, 140.4833],
        '徳山動物園': [34.0500, 131.8000],
        '池田動物園': [34.6500, 133.9167],
        '桐生が岡動物園': [36.4000, 139.3333],
        '楽寿園': [35.6333, 139.4667],
        '江戸川区自然動物園': [35.7000, 139.8667],
        '羽村市動物公園': [35.7667, 139.3167],
        '福知山市動物園': [35.3000, 135.1167],
        '秋吉台サファリランド': [34.2333, 131.3000],
        '長崎バイオパーク': [32.8333, 129.8833]
    }
    return zoo_coords

st.title("Family Tree Generator")

default_csv_path = "redpanda.csv"
use_default = st.checkbox("Use default CSV file (redpanda.csv in the same folder)", value=True)
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
data = None
if use_default and os.path.exists(default_csv_path):
    data = read_csv(default_csv_path)
elif uploaded_file is not None:
    data = read_csv(uploaded_file)

tr, ppy, gantt, genetic, death_age, relationship, birthday, map_view, genetic_distance, birth_death_stats, survival_timeline = st.tabs(["Family Tree", "Population Pyramid", "Gantt Chart", "Genetic Distribution", "Death Age Histogram", "Relationship Analysis", "Birthday Calendar", "Map View", "Genetic Distance", "Birth/Death Stats", "Survival Timeline"])
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
    df = df[~df['cur_zoo'].isin(get_foreign_zoos())]
    live_df = df[df['deaddate'].isnull()].copy()
    dead_df = df[~df['deaddate'].isnull()].copy()
    
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
    live_df = df[df['deaddate'].isnull()].copy()_
    dead_df = df[~df['deaddate'].isnull()].copy()
    live_df = prepare_gantt_dataframe(live_df)
    dead_df = prepare_gantt_dataframe(dead_df)
    

    # 2003年1月1日以降のデータのみ抽出
    live_df = live_df[live_df['birthdate'] >= pd.Timestamp('2003-01-01')]
    
    df = pd.concat([live_df, dead_df])
    zoo_options = get_zoo_options(df)
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
    
    df = prepare_dataframe_for_analysis(df)
    
    # Use the same DataFrame as in the Gantt chart tab
    if df.shape[0] > 0:
        # Select an individual
        individual_options = get_individual_options(df)
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
            individual_options = get_individual_options(df)
            
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
    
    # 誕生日を日付型に変換
    df['birthdate'] = pd.to_datetime(df['birthdate'].apply(convert_date))
    
    # 生存している個体のみを抽出（deaddateがnullまたは空の個体）
    live_df = df[df['deaddate'].isna()].copy()
    
    # 日本以外の動物園に所属している個体を除外
    japan_df = live_df[~live_df['cur_zoo'].isin(get_foreign_zoos())].copy()
    
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

with map_view:
    st.title("Map View of Living Individuals")
    
    # CSVファイルの読み込み
    if use_default and os.path.exists(default_csv_path):
        df = pd.read_csv(default_csv_path)
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    
    # 生存している個体のみを抽出（deaddateがnullまたは空の個体）
    live_df = df[df['deaddate'].isna()].copy()
    
    # 日本以外の動物園に所属している個体を除外
    japan_df = live_df[~live_df['cur_zoo'].isin(get_foreign_zoos())].copy()
    
    if not japan_df.empty:
        # 動物園の座標を取得
        zoo_coords = get_zoo_coordinates()
        
        # 日本の中心座標でマップを作成
        japan_center = [36.2048, 138.2529]  # 日本の中心座標
        m = folium.Map(location=japan_center, zoom_start=5, tiles='OpenStreetMap')
        
        # 動物園ごとに個体をグループ化
        zoo_individuals = {}
        for _, row in japan_df.iterrows():
            zoo = row['cur_zoo']
            if zoo not in zoo_individuals:
                zoo_individuals[zoo] = []
            zoo_individuals[zoo].append({
                'name': row['name'],
                'gender': row['gender'],
                'birthdate': row['birthdate']
            })
        
        # 各動物園にマーカーを追加
        for zoo, individuals in zoo_individuals.items():
            if zoo in zoo_coords:
                lat, lon = zoo_coords[zoo]
                
                # 個体の情報をHTMLで作成
                individuals_html = "<div style='max-height: 200px; overflow-y: auto;'>"
                individuals_html += f"<h4>{zoo}</h4>"
                individuals_html += f"<p><strong>個体数: {len(individuals)}</strong></p>"
                individuals_html += "<ul>"
                
                # 性別ごとに個体を分類
                males = [ind for ind in individuals if ind['gender'] == 'オス']
                females = [ind for ind in individuals if ind['gender'] == 'メス']
                
                if males:
                    individuals_html += "<li><strong>オス:</strong>"
                    for male in males:
                        individuals_html += f" {male['name']},"
                    individuals_html = individuals_html.rstrip(',') + "</li>"
                
                if females:
                    individuals_html += "<li><strong>メス:</strong>"
                    for female in females:
                        individuals_html += f" {female['name']},"
                    individuals_html = individuals_html.rstrip(',') + "</li>"
                
                individuals_html += "</ul></div>"
                
                # マーカーの色を個体数に応じて変更
                if len(individuals) >= 5:
                    color = 'red'
                elif len(individuals) >= 3:
                    color = 'orange'
                elif len(individuals) >= 2:
                    color = 'purple'
                else:
                    color = 'green'
                
                # マーカーを追加
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(individuals_html, max_width=300),
                    tooltip=f"{zoo}: {len(individuals)}個体",
                    icon=folium.Icon(color=color, icon='info-sign')
                ).add_to(m)
        
        # マップを表示
        st_folium(m, width=700, height=500)
        
        # 統計情報を表示
        st.write("### 統計情報")
        st.write(f"**総個体数**: {len(japan_df)}")
        st.write(f"**動物園数**: {len(zoo_individuals)}")
        
        # 性別ごとの統計
        male_count = len(japan_df[japan_df['gender'] == 'オス'])
        female_count = len(japan_df[japan_df['gender'] == 'メス'])
        st.write(f"**オス**: {male_count}個体")
        st.write(f"**メス**: {female_count}個体")
        
        # 動物園別の個体数ランキング
        st.write("### 動物園別個体数ランキング")
        zoo_counts = {zoo: len(individuals) for zoo, individuals in zoo_individuals.items()}
        sorted_zoos = sorted(zoo_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, (zoo, count) in enumerate(sorted_zoos, 1):
            st.write(f"{i}. {zoo}: {count}個体")
        
        # マーカーの色の説明
        st.write("### マーカーの色の意味")
        st.write("- 🟢 緑: 1個体")
        st.write("- 🟡 黄: 2個体")
        st.write("- 🟠 オレンジ: 3-4個体")
        st.write("- 🔴 赤: 5個体以上")
        
    else:
        st.warning("日本国内に生存している個体が見つかりませんでした。")

with genetic_distance:
    st.title("Genetic Distance Between Individuals")
    
    # CSVファイルの読み込み
    if use_default and os.path.exists(default_csv_path):
        df = pd.read_csv(default_csv_path)
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = None
    
    if df is not None and df.shape[0] > 0:
        df = prepare_dataframe_for_analysis(df)
        
        # 日本にいる生存個体をフィルタリング
        japan_df = filter_japan_living_individuals(df)
        
        if japan_df is None:
            st.warning("No living individuals in Japan available for genetic distance analysis.")
        else:
            individual_options = get_individual_options(japan_df)
            selected_individual = st.selectbox(
                "Select Individual for Genetic Distance", 
                [""] + individual_options, 
                key="genetic_distance_select"
            )
            
            if selected_individual:
                selected_row = japan_df[japan_df['name'] == selected_individual]
                if selected_row.empty:
                    st.warning("Selected individual not found among living individuals in Japan.")
                else:
                    selected_gender = selected_row.iloc[0]['gender']
                    
                    # 反対の性別で9歳以下の候補者を取得
                    candidates_df, opposite_gender = get_opposite_gender_candidates(japan_df, selected_gender, max_age=9)
                    
                    if candidates_df.empty:
                        st.warning(f"No living {opposite_gender} individuals aged 9 or younger found in Japan.")
                    else:
                        # 遺伝子ベクトル計算のための準備
                        zoo_set = set(df['cur_zoo'].dropna().unique())
                        zoo_list = sorted(list(zoo_set))
                        zoo_index = {z: i for i, z in enumerate(zoo_list)}
                        
                        # 遺伝子的距離を計算
                        distances = calculate_genetic_distances(df, selected_individual, candidates_df, zoo_list, zoo_index)
                        
                        # 結果を表示
                        st.write(f"Top 5 living {opposite_gender} individuals (aged 9 or younger) with the largest genetic distance from {selected_individual}:")
                        for i, (name, dist, age) in enumerate(distances[:5], 1):
                            st.write(f"{i}. {name} (Age: {age}, Genetic Distance: {dist:.4f})")
                        
                        # 距離分布のヒストグラムを表示
                        if distances:
                            st.write("\n#### Genetic Distance Distribution (all candidates)")
                            fig = plot_distance_distribution(distances)
                            st.pyplot(fig)
            else:
                st.info("Please select a living individual in Japan to calculate genetic distances.")
    else:
        st.warning("No data available for genetic distance analysis.")

with birth_death_stats:
    st.title("Yearly Births and Deaths")
    # CSVファイルの読み込み
    if use_default and os.path.exists(default_csv_path):
        df = pd.read_csv(default_csv_path)
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = None
    if df is not None and df.shape[0] > 0:
        # 日本の個体だけに限定
        japan_df = df[~df['cur_zoo'].isin(get_foreign_zoos())].copy()
        # 日付の変換
        japan_df['birthdate'] = pd.to_datetime(japan_df['birthdate'].apply(convert_date))
        japan_df['deaddate'] = pd.to_datetime(japan_df['deaddate'].apply(convert_date))
        # 2000年以降に限定
        japan_df = japan_df[(japan_df['birthdate'].dt.year >= 2000) | (japan_df['deaddate'].dt.year >= 2000)]
        # 年ごとに集計
        birth_years = japan_df['birthdate'].dt.year.dropna().astype(int)
        death_years = japan_df['deaddate'].dt.year.dropna().astype(int)
        birth_counts = birth_years.value_counts().sort_index()
        death_counts = death_years.value_counts().sort_index()
        # 年の範囲を決定
        all_years = [y for y in sorted(set(birth_counts.index).union(set(death_counts.index))) if y >= 2000]
        birth_vals = [birth_counts.get(y, 0) for y in all_years]
        death_vals = [death_counts.get(y, 0) for y in all_years]
        # 棒グラフ
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.4
        ax.bar([y - width/2 for y in all_years], birth_vals, width=width, label='Births', color='skyblue')
        ax.bar([y + width/2 for y in all_years], death_vals, width=width, label='Deaths', color='salmon')
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        ax.set_title('Yearly Births and Deaths (Japan Only, 2000-)')
        ax.legend()
        ax.set_xticks(all_years)
        ax.set_xticklabels([str(y) for y in all_years], rotation=45)
        # 5ごとにグリッドライン
        max_count = max(birth_vals + death_vals) if (birth_vals + death_vals) else 0
        ax.set_yticks(np.arange(0, max_count + 6, 5))
        ax.yaxis.grid(True, which='major', linestyle='--', color='gray', alpha=0.7)
        st.pyplot(fig)
        
        # 指定された年の死亡個体の年齢分布
        st.write("---")
        st.subheader("Age Distribution of Deaths by Year")
        
        # 死亡記録があるデータを取得
        death_df = japan_df[japan_df['deaddate'].notna()].copy()
        
        if not death_df.empty:
            # 死亡年を計算
            death_df['death_year'] = death_df['deaddate'].dt.year.astype(int)
            
            # 利用可能な年のリストを作成（2000年から今年まで）
            current_year = datetime.now().year
            available_years = sorted([y for y in death_df['death_year'].unique() if 2000 <= y <= current_year])
            
            if available_years:
                # セレクトボックスで年を選択
                selected_year = st.selectbox(
                    "Select Year for Death Age Distribution",
                    available_years,
                    index=len(available_years) - 1 if available_years else 0,
                    key="death_age_year_select"
                )
                
                # 選択された年に死亡した個体をフィルタリング
                year_deaths = death_df[death_df['death_year'] == selected_year].copy()
                
                if not year_deaths.empty:
                    # 死亡時の年齢を計算
                    year_deaths['death_age'] = (year_deaths['deaddate'] - year_deaths['birthdate']).dt.days / 365.25
                    
                    # 年齢分布のヒストグラムを作成
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    
                    # 年齢の範囲を設定
                    max_age = int(year_deaths['death_age'].max()) + 1
                    bins = range(0, max_age + 1)
                    
                    # 性別ごとにヒストグラムを作成
                    male_data = year_deaths[year_deaths['gender'] == 'オス']['death_age']
                    female_data = year_deaths[year_deaths['gender'] == 'メス']['death_age']
                    
                    # データがある場合のみプロット
                    plot_data = []
                    plot_labels = []
                    plot_colors = []
                    
                    if not male_data.empty:
                        plot_data.append(male_data)
                        plot_labels.append('Male')
                        plot_colors.append('skyblue')
                    
                    if not female_data.empty:
                        plot_data.append(female_data)
                        plot_labels.append('Female')
                        plot_colors.append('lightpink')
                    
                    if plot_data:
                        ax2.hist(plot_data, bins=bins, alpha=0.7, label=plot_labels, color=plot_colors, edgecolor='black')
                    
                    # グラフの装飾
                    ax2.set_xlabel('Age at Death (years)', fontsize=12)
                    ax2.set_ylabel('Number of Individuals', fontsize=12)
                    ax2.set_title(f'Age Distribution of Deaths in {selected_year}', fontsize=14, fontweight='bold')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    # X軸の目盛りを設定
                    ax2.set_xticks(range(0, max_age + 1, 2))  # 2歳ごと
                    
                    # レイアウトの調整
                    plt.tight_layout()
                    
                    # グラフを表示
                    st.pyplot(fig2)
                    
                    # 統計情報の表示
                    st.write(f"### Statistics for {selected_year}")
                    st.write(f"**Total deaths**: {len(year_deaths)} individuals")
                    st.write(f"**Average age at death**: {year_deaths['death_age'].mean():.1f} years")
                    st.write(f"**Median age at death**: {year_deaths['death_age'].median():.1f} years")
                    st.write(f"**Minimum age at death**: {year_deaths['death_age'].min():.1f} years")
                    st.write(f"**Maximum age at death**: {year_deaths['death_age'].max():.1f} years")
                    
                    # 性別ごとの統計情報
                    if not male_data.empty:
                        st.write("**Male Statistics:**")
                        st.write(f"- Count: {len(male_data)}")
                        st.write(f"- Average age: {male_data.mean():.1f} years")
                        st.write(f"- Median age: {male_data.median():.1f} years")
                    
                    if not female_data.empty:
                        st.write("**Female Statistics:**")
                        st.write(f"- Count: {len(female_data)}")
                        st.write(f"- Average age: {female_data.mean():.1f} years")
                        st.write(f"- Median age: {female_data.median():.1f} years")
                else:
                    st.info(f"No death records found for {selected_year}.")
            else:
                st.info("No death records available for years between 2000 and the current year.")
        else:
            st.info("No death records found in the dataset.")
    else:
        st.warning("No data available for birth/death statistics.")

with survival_timeline:
    st.title("Survival Timeline - Living Individuals in Japan")
    
    # CSVファイルの読み込み
    if use_default and os.path.exists(default_csv_path):
        df = pd.read_csv(default_csv_path)
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = None
    
    if df is not None and df.shape[0] > 0:
        # 日本の個体だけに限定（外国の動物園を除外）
        japan_df = df[~df['cur_zoo'].isin(get_foreign_zoos())].copy()
        
        # 日付の変換
        japan_df['birthdate'] = pd.to_datetime(japan_df['birthdate'].apply(convert_date))
        japan_df['deaddate'] = pd.to_datetime(japan_df['deaddate'].apply(convert_date))
        
        # 2000年1月1日から現在までの月ごとの日付を生成
        start_date = pd.Timestamp('2000-01-01')
        end_date = pd.Timestamp.now()
        
        # 月ごとの日付リストを作成（各月の1日）
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # MS = Month Start
        
        # 各月での生存個体数を計算
        survival_counts = []
        survival_dates = []
        
        for month_date in monthly_dates:
            # その月の1日時点での生存個体数を計算
            # 条件: 誕生日がその月以前 かつ (死亡日がその月以降 または 死亡日がnull)
            living_at_date = japan_df[
                (japan_df['birthdate'] <= month_date) & 
                ((japan_df['deaddate'].isna()) | (japan_df['deaddate'] > month_date))
            ]
            
            survival_counts.append(len(living_at_date))
            survival_dates.append(month_date)
        
        # グラフの作成
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 線グラフでプロット
        ax.plot(survival_dates, survival_counts, linewidth=2, color='green', marker='o', markersize=3)
        
        # グラフの装飾
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of Living Individuals', fontsize=12)
        ax.set_title('Timeline of Living Individuals in Japan (2000-Present)', fontsize=14, fontweight='bold')
        
        # グリッドの追加
        ax.grid(True, alpha=0.3)
        
        # X軸の目盛りを月ごとに設定
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=6))  # 6ヶ月ごと
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        
        # X軸のラベルを回転
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Y軸の範囲を調整
        if survival_counts:
            max_count = max(survival_counts)
            min_count = min(survival_counts)
            margin = (max_count - min_count) * 0.1 if max_count != min_count else 1
            ax.set_ylim(max(0, min_count - margin), max_count + margin)
        
        # レイアウトの調整
        plt.tight_layout()
        
        # グラフを表示
        st.pyplot(fig)
        
        # 統計情報の表示
        st.write("### Statistics")
        
        if survival_counts:
            st.write(f"**Maximum population**: {max(survival_counts)} individuals")
            st.write(f"**Minimum population**: {min(survival_counts)} individuals")
            st.write(f"**Current population**: {survival_counts[-1]} individuals")
            
            # 最大・最小の日付を特定
            max_idx = survival_counts.index(max(survival_counts))
            min_idx = survival_counts.index(min(survival_counts))
            
            st.write(f"**Peak population date**: {survival_dates[max_idx].strftime('%Y-%m')} ({max(survival_counts)} individuals)")
            st.write(f"**Lowest population date**: {survival_dates[min_idx].strftime('%Y-%m')} ({min(survival_counts)} individuals)")
            
            # 最近の傾向（過去12ヶ月）
            if len(survival_counts) >= 12:
                recent_counts = survival_counts[-12:]
                recent_dates = survival_dates[-12:]
                
                # 線形回帰で傾向を計算
                x_numeric = np.arange(len(recent_counts))
                slope, intercept = np.polyfit(x_numeric, recent_counts, 1)
                
                st.write("### Recent Trend (Past 12 Months)")
                if slope > 0:
                    st.write(f"📈 **Increasing trend**: +{slope:.1f} individuals per month")
                elif slope < 0:
                    st.write(f"📉 **Decreasing trend**: {slope:.1f} individuals per month")
                else:
                    st.write("📊 **Stable trend**: No significant change")
                
                # 最近12ヶ月の詳細データを表示
                st.write("**Monthly data (past 12 months):**")
                for i, (date, count) in enumerate(zip(recent_dates, recent_counts)):
                    st.write(f"- {date.strftime('%Y-%m')}: {count} individuals")
        
        # 説明文
        st.write("### About This Chart")
        st.write("""
        This chart shows the number of living red pandas in Japan over time:
        
        - **X-axis**: Monthly dates from January 2000 to present
        - **Y-axis**: Number of living individuals at each point in time
        - **Data**: Only includes individuals in Japanese zoos (excludes foreign zoos)
        - **Method**: Counts individuals who were born before or on the date and either died after the date or are still alive
        """)
        
    else:
        st.warning("No data available for survival timeline analysis.")

