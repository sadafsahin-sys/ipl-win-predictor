import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="IPL Dual-Prediction & Analytics Engine", layout="wide")

st.title('🏏 IPL Dual-Feature Prediction & Analytics Engine')
st.markdown("---")

st.sidebar.header("🛠️ Control Panel")
app_mode = st.sidebar.radio("Select System Feature:", ["Pre-Match Predictor (Static)", "In-Match Predictor (Live)"])

# 10 Expanded Teams Standard List
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bengaluru',
    'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals', 'Gujarat Titans', 'Lucknow Super Giants'
]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamshala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Dubai', 'Mohali', 'Bengaluru', 'Lucknow']

if os.path.exists('pre_match_model.pkl') and os.path.exists('in_match_model.pkl'):
    pipe_pre = pickle.load(open('pre_match_model.pkl', 'rb'))
    pipe_in = pickle.load(open('in_match_model.pkl', 'rb'))
    dual_mode_ready = True
else:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    dual_mode_ready = False

# =========================================================================
# FEATURE 1: PRE-MATCH PREDICTOR (STATIC MODE)
# =========================================================================
if app_mode == "Pre-Match Predictor (Static)":
    st.subheader("📊 Pre-Match Win Probability (Before First Ball)")
    
    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select Team 1 (Batting First)', sorted(teams), index=0)
    with col2:
        bowling_team = st.selectbox('Select Team 2 (Bowling First)', sorted(teams), index=1)
        
    selected_city = st.selectbox('Select Match Venue/City', sorted(cities))
    
    st.markdown("##### ⚙️ Extra Context Constraints")
    toss_winner = st.selectbox('Who won the Toss?', [batting_team, bowling_team])
    toss_decision = st.selectbox('Toss Decision', ['Bat', 'Field'])

    if st.button('Predict Pre-Match Probability'):
        if batting_team == bowling_team:
            st.error("Error: Team 1 and Team 2 cannot be the same!")
        else:
            if dual_mode_ready:
                input_df = pd.DataFrame({
                    'team1': [batting_team], 'team2': [bowling_team], 'city': [selected_city],
                    'toss_winner': [toss_winner], 'toss_decision': [toss_decision.lower()]
                })
                result = pipe_pre.predict_proba(input_df)
            else:
                input_df = pd.DataFrame({
                    'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                    'runs_left': [180], 'balls_left': [120], 'wickets': [10],
                    'total_runs_x': [180], 'crr': [0.0], 'rrr': [9.0]
                })
                result = pipe.predict_proba(input_df)
                
            loss = result[0][0]
            win = result[0][1]
            
            st.markdown("### 📈 Outcome Probability:")
            st.success(f"{batting_team} Win Probability: {round(win*100)}%")
            st.warning(f"{bowling_team} Win Probability: {round(loss*100)}%")

# =========================================================================
# FEATURE 2: IN-MATCH PREDICTOR (LIVE MODE)
# =========================================================================
elif app_mode == "In-Match Predictor (Live)":
    st.subheader("⏱️ Live In-Match Dynamics & Predictive Charting")

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select Batting Team', sorted(teams), index=0)
    with col2:
        bowling_team = st.selectbox('Select Bowling Team', sorted(teams), index=1)

    selected_city = st.selectbox('Select Match Venue/City', sorted(cities))
    target = st.number_input('Target Score Set', min_value=1, value=150)

    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('Current Score', min_value=0, value=0)
    with col4:
        overs = st.number_input('Overs Completed (0.1 - 19.5)', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    with col5:
        wickets = st.number_input('Wickets Down (0-10)', min_value=0, max_value=10, value=2)

    if st.button('Predict Live Probability & Generate Analytics'):
        if batting_team == bowling_team:
            st.error("Error: Batting and Bowling teams cannot be the same!")
        elif score > target:
            st.success(f"🎉 {batting_team} has already won the match!")
        elif wickets >= 10 or overs >= 20:
            st.error("Match Over!")
        elif overs == 0:
            st.warning("Please input current ongoing over context to start simulations.")
        else:
            runs_left = target - score
            balls_left = 120 - int(overs * 6)
            wickets_left = 10 - wickets
            crr = (score * 6) / (120 - balls_left)
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

            input_df = pd.DataFrame({
                'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets_left],
                'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]
            })

            model_to_use = pipe_in if dual_mode_ready else pipe
            result = model_to_use.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]

            st.markdown("---")
            m1, m2 = st.columns(2)
            with m1:
                st.metric(label=f"🔥 {batting_team} Chasing Chance", value=f"{round(win*100)}%")
            with m2:
                st.metric(label=f"🛡️ {bowling_team} Defending Chance", value=f"{round(loss*100)}%")

            # =========================================================================
            # ADVANCED FEATURE: SIMULATING OVER-BY-OVER PROGRESSION GRAPH
            # =========================================================================
            st.markdown("### 📈 Live Over-by-Over Trend Simulation")
            st.write("Below matrix maps the target threshold degradation if batting team scores at current CRR vs required RRR.")

            current_over_int = int(overs)
            sim_overs = list(range(current_over_int, 21))
            
            batting_prob_trend = []
            sim_score = score
            sim_wickets = wickets

            for o in sim_overs:
                if o == current_over_int:
                    batting_prob_trend.append(win)
                    continue
                
                s_left = max(0, target - sim_score)
                b_left = max(1, 120 - (o * 6))
                w_left = max(0, 10 - sim_wickets)
                c_runrate = (sim_score * 6) / (120 - b_left) if (120 - b_left) > 0 else crr
                r_runrate = (s_left * 6) / b_left
                
                sim_df = pd.DataFrame({
                    'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                    'runs_left': [s_left], 'balls_left': [b_left], 'wickets': [w_left],
                    'total_runs_x': [target], 'crr': [c_runrate], 'rrr': [r_runrate]
                })
                
                sim_res = model_to_use.predict_proba(sim_df)
                batting_prob_trend.append(sim_res[0][1])
                
                sim_score += int(crr) if crr > 4 else 6 
                if o % 4 == 0 and sim_wickets < 9:
                    sim_wickets += 1

            chart_data = pd.DataFrame({
                'Overs Timeline': sim_overs,
                f'{batting_team} Probability': [p * 100 for p in batting_prob_trend]
            }).set_index('Overs Timeline')

            st.line_chart(chart_data, y=f'{batting_team} Probability', use_container_width=True)

            # =========================================================================
            # AUTOMATED CONSULTANT REPORTING BOX
            # =========================================================================
            st.markdown("---")
            st.subheader("📋 Consultant Report Summary (For Clients)")
            
            report_text = f"""*IPL Match Insights & Probability Analytics Report*
--------------------------------------------------
- Match Context: {batting_team} vs {bowling_team}
- Venue / Conditions: {selected_city}
- Live State: {score}/{wickets} in {overs} Overs (Target: {target})
- Current CRR: {crr:.2f} | Required RRR: {rrr:.2f}

*Predictive Analytics Machine Learning Probability:*
👉 {batting_team} Chasing Success Probability: {round(win*100)}%
👉 {bowling_team} Defense Success Probability: {round(loss*100)}%
--------------------------------------------------
*Disclaimer: This is a data-driven statistical analysis model for informational insights only.*
Generated via IPL Advanced Predictive Engine."""
            
            st.code(report_text, language="markdown")