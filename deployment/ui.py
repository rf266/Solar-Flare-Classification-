import streamlit as st
import requests
import pandas as pd
  
st.title("Solar Flare Level Predictor")

st.subheader("Enter information about a sunspot to classify the expected number (mean/lambda) of solar flares of each type in a 24H period: common, moderate or severe")

zurich=st.selectbox("Modified Zurich class", ["B", "C", "D", "E", "F", "H"])
spot_size=st.selectbox("Largest spot size", ["X","R","S","A","H","K"])
spot_dist = st.selectbox("Spot distribution", ["X", "O", "I", "C"])
activity = st.selectbox("Activity - 1: reduced, 2: unchanged", [1,2])
evol = st.selectbox("Evolution - 1: decay, 2: no growth, 3: growth", [1,2,3])
prev24 = st.selectbox("Previous 24H flare activity - 1: Not as big as M1, 2: One M1, 3: more activity than one M1",[1,2,3] )
hist = st.selectbox("Historically complex - 1: Yes, 2: No", [1,2])
complex = st.selectbox("Became complex on this pass (across sun's disk) - 1: Yes, 2: No", [1,2])
area =st.selectbox("Area - 1: small, 2: large", [1,2])
area_largest =st.selectbox("Area of largest spot - 1: less than or equal to 5, 2: more than 5", [1,2])
submit = st.button("Submit")

#attempting to one hot encode inputs

if zurich=="B":
    zurich_b = True
    zurich_c = False
    zurich_d = False
    zurich_e = False
    zurich_f = False
    zurich_h = False
elif zurich=="C":
    zurich_b = False
    zurich_c = True
    zurich_d = False
    zurich_e = False
    zurich_f = False
    zurich_h = False

elif zurich=="D":
    zurich_b = False
    zurich_c = False
    zurich_d = True
    zurich_e = False
    zurich_f = False
    zurich_h = False

elif zurich=="E":
    zurich_b = False
    zurich_c = False
    zurich_d = False
    zurich_e = True
    zurich_f = False
    zurich_h = False

elif zurich=="F":
    zurich_b = False
    zurich_c = False
    zurich_d = False
    zurich_e = False
    zurich_f = True
    zurich_h = False

elif zurich=="H":
    zurich_b = False
    zurich_c = False
    zurich_d = False
    zurich_e = False
    zurich_f = False
    zurich_h = True

if spot_size=="X":
    spot_size_a = False
    spot_size_h = False
    spot_size_k = False
    spot_size_r = False
    spot_size_s = False
    spot_size_x = True
elif spot_size=="A":
    spot_size_a = True
    spot_size_h = False
    spot_size_k = False
    spot_size_r = False
    spot_size_s = False
    spot_size_x = False
elif spot_size=="H":
    spot_size_a = False
    spot_size_h = True
    spot_size_k = False
    spot_size_r = False
    spot_size_s = False
    spot_size_x = False
elif spot_size=="K":
    spot_size_a = False
    spot_size_h = False
    spot_size_k = True
    spot_size_r = False
    spot_size_s = False
    spot_size_x = False
elif spot_size=="R":
    spot_size_a = False
    spot_size_h = False
    spot_size_k = False
    spot_size_r = True
    spot_size_s = False
    spot_size_x = False
elif spot_size=="S":
    spot_size_a = False
    spot_size_h = False
    spot_size_k = False
    spot_size_r = False
    spot_size_s = True
    spot_size_x = False

if spot_dist=="X":
    spot_dist_c = False
    spot_dist_i = False
    spot_dist_o = False
    spot_dist_x = True
elif spot_dist=="C":
    spot_dist_c = True
    spot_dist_i = False
    spot_dist_o = False
    spot_dist_x = False
elif spot_dist=="I":
    spot_dist_c = False
    spot_dist_i = False
    spot_dist_o = True
    spot_dist_x = False
elif spot_dist=="O":
    spot_dist_c = False
    spot_dist_i = False
    spot_dist_o = True
    spot_dist_x = False

inputs= {
        "activity": activity,  
        "evolution": evol,
        "previous 24 hour flare activity": prev24,
        "historically-complex":hist,
        "became complex on this pass": complex,
        "area" : area,
        "area of largest spot": area_largest,
        "modified Zurich class_B":zurich_b,
        "modified Zurich class_C":zurich_c,
        "modified Zurich class_D": zurich_d,
        "modified Zurich class_E":zurich_e,	
        "modified Zurich class_F": zurich_f,
        "modified Zurich class_H" : zurich_h,
        "largest spot size_A": spot_size_a,
        "largest spot size_H": spot_size_h,
        "largest spot size_K": spot_size_k,
        "largest spot size_R": spot_size_r,
        "largest spot size_S": spot_size_s,
        "largest spot size_X": spot_size_x,
        "spot distribution_C": spot_dist_c,
        "spot distribution_I": spot_dist_i,
        "spot distribution_O": spot_dist_o,
        "spot distribution_X": spot_dist_x
        }

if submit:
    req = requests.post("https://end-to-end-solar-flare-regression-model.onrender.com/", json=inputs)
    preds = req.json()
    commonf = preds["common flares"]['0']
    moderate = preds["moderate flares"]['0']
    severe = preds["severe flares"]['0']

    st.subheader("Predictions - Expected (Mean) number of solar flares in a 24 hour period")

    st.write(f"Common Flares: {commonf:.5f}") 
    st.write(f"Moderate Flares: {moderate:.5f}")
    st.write(f"Severe Flares: {severe:.5f}")

    st.write("Variable information: UCI Machine Learning Repository - https://archive.ics.uci.edu/dataset/89/solar+flare")



