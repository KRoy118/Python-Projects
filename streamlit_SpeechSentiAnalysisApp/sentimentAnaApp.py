#App for the Emotion/Sentiment in text classifier  using Streamlit

from sre_constants import SUCCESS
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import joblib

pipe_lr=joblib.load(open("model/sentiment_analysis.pkl","rb" ))

def predict_emo(docx):
    results=pipe_lr.predict([docx])
    return results[0]

def get_pred_proba(docx):
    results=pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


def main():
    st.title("Sentiment Analysis App")
    menu=["Home","About"]
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
                    st.write("Confidence:{}".format(np.max(probability)))
                with col2:
                     st.success("Prediction probability")
                     st.success(probability)
                     proba_df= pd.DataFrame(probability,columns=pipe_lr.classes_)
                     st.write(proba_df.T)
                     proba_df_clean=proba_df.T.reset_index()
                     proba_df_clean.columns=["emotions","probability"]
                     fig=alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability')
                     st.altair_chart(fig,use_container_width=True)

    else:
        st.subheader("About")
        st.write("The basic types of sentiments analysed in this application are:")
        st.write("anger:😠,disgust:🤮, fear:😨😱, happy:🤗, joy:😂, neutral:😐, sadness:😔, shame:😳, surprise:😮")


if __name__=='__main__':
    main()