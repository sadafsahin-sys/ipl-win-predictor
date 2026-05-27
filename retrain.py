import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print("🔄 Building Dual-Model Architecture with Expanded Teams (GT & LSG)...")

# Datasets Load Kora
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')

# Standardizing Team Names Across All Eras (2008 - Present)
team_mappings = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
}

for old_name, new_name in team_mappings.items():
    match['team1'] = match['team1'].str.replace(old_name, new_name)
    match['team2'] = match['team2'].str.replace(old_name, new_name)
    match['toss_winner'] = match['toss_winner'].str.replace(old_name, new_name)
    match['winner'] = match['winner'].str.replace(old_name, new_name)
    if 'batting_team' in delivery.columns:
        delivery['batting_team'] = delivery['batting_team'].str.replace(old_name, new_name)
    if 'bowling_team' in delivery.columns:
        delivery['bowling_team'] = delivery['bowling_team'].str.replace(old_name, new_name)

# Updated 10 Active Teams Context Blueprint
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bengaluru',
    'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals', 'Gujarat Titans', 'Lucknow Super Giants'
]

match_clean = match[match['team1'].isin(teams) & match['team2'].isin(teams)]
if 'dl_applied' in match_clean.columns:
    match_clean = match_clean[match_clean['dl_applied'] == 0]

# =========================================================================
# MODEL 1 BUILDING: PRE-MATCH ANALYTICS ENGINE
# =========================================================================
print("📈 Processing Pre-Match Features (Random Forest)...")
match_clean['match_winner_binary'] = match_clean.apply(lambda row: 1 if row['team1'] == row['winner'] else 0, axis=1)

X_pre = match_clean[['team1', 'team2', 'city', 'toss_winner', 'toss_decision']]
y_pre = match_clean['match_winner_binary']

trf_pre = ColumnTransformer([
    ('trf_pre', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), ['team1', 'team2', 'city', 'toss_winner', 'toss_decision'])
], remainder='passthrough')

pipe_pre = Pipeline(steps=[
    ('step1', trf_pre),
    ('step2', RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10))
])
pipe_pre.fit(X_pre, y_pre)

with open('pre_match_model.pkl', 'wb') as f:
    pickle.dump(pipe_pre, f)
print("💾 Save Successful: pre_match_model.pkl ready.")

# =========================================================================
# MODEL 2 BUILDING: IN-MATCH LIVE ENGINE
# =========================================================================
print("⏱️ Processing In-Match Cumulative Accumulators...")

inning_col = 'inning' if 'inning' in delivery.columns else 'innings'

# Total Score Evaluation
total_score_df = delivery.groupby(['match_id', inning_col])['total_runs'].sum().reset_index()
total_score_df = total_score_df[total_score_df[inning_col] == 1]

match_df = match_clean.merge(total_score_df[['match_id','total_runs']], left_on='id', right_on='match_id')
delivery_df = match_df.merge(delivery, on='match_id')
delivery_df = delivery_df[delivery_df[inning_col] == 2]

# Explicitly selecting numeric columns for cumsum() to satisfy Pandas 3.0+
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
delivery_df['runs_left'] = delivery_df['total_runs_x'] + 1 - delivery_df['current_score']
delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0").apply(lambda x: 0 if x == "0" else 1).astype('int')

# Explicitly selecting player_dismissed for cumsum()
delivery_df['wickets_lost'] = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
delivery_df['wickets'] = 10 - delivery_df['wickets_lost']

delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])
delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']
delivery_df['result'] = delivery_df.apply(lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1)

final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']].dropna()
final_df = final_df[final_df['balls_left'] != 0]

X_in = final_df.iloc[:,:-1]
y_in = final_df.iloc[:,-1]

trf_in = ColumnTransformer([
    ('trf_in', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

pipe_in = Pipeline(steps=[
    ('step1', trf_in),
    ('step2', LogisticRegression(solver='liblinear'))
])
pipe_in.fit(X_in, y_in)

with open('in_match_model.pkl', 'wb') as f:
    pickle.dump(pipe_in, f)
print("💾 Save Successful: in_match_model.pkl ready.")
print("🏁 Dual-Engine system updated with 100% data integration symmetry for 10 Teams!")