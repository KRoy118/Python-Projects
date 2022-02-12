from sre_constants import SUCCESS
import streamlit as st
import pandas as pd
import numpy as np
import joblib

pipe_lr=joblib.load(open("models/sentiment_analysis.pkl","rb" ))

def predict_emo(docx):
    results=pipe_lr.predict([docx])
    return results[0]

def get_pred_proba(docx):
    results=pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


def main():
    st.title("Sentiment Analysis App")
    menu=["Home","Display"]
    choice=st.sidebar.selectbox("Dashboard",menu)
    if choice=="Home":
        st.subheader("Emotion/sentiment classification from text")
        with st.form(key='emo_clf_form'):
            raw_text=st.text_area("Type here")
            submit_text=st.form_submit_button(label="Submit")

    if submit_text:
        col1,col2=st.beta_columns(2)
        prediction=predict_emo(raw_text)
        probability=get_pred_proba(raw_text)

        with col1:
            st.success("Original text")
            st.write(raw_text)
            st.success("Prediction")
            emoji_icon=emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction,emoji_icon))
            st.write(prediction)
        with col2:
            st.success("Prediction probability")
            st.success(probability)
            
    else:
        st.subheader("Display")

if __name__=='__main__':
    main()