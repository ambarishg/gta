import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

key = st.secrets["OPENAI_KEY"]
filename = "data/ne.txt"
import os
import openai
openai.api_key = key

def create_prompt(context,query):
    header = "Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text and requires some latest information to be updated, print 'Sorry Not Sufficient context to answer query' \n"
    return header + context + "\n\n" + query + "\n"

def generate_answer(prompt):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop = [' END']
    )
    return (response.choices[0].text).strip()

def clean_text(text):
    '''Make text lowercase,remove punctuation
    .'''
    text = str(text).lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    return text

file1 = open(filename, 'r',encoding="utf8")
Lines = file1.readlines()
file1.close()

st.header("EXELON Innovation Question and Answering System")

user_input = st.text_area("Your Question",
"What is SLIQ?")

result = st.button("Make recommendations")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache(allow_output_mutation=True)
def get_encodings(Lines, model):
    pole_data = model.encode(Lines)
    return pole_data

if result:
    pole_data = get_encodings(Lines, model)
    q_new = user_input
    q_new = [model.encode(q_new)]
    result = cosine_similarity(q_new,pole_data)
    result_df = pd.DataFrame(result[0], columns = ['sim'])
    df = pd.DataFrame(Lines,columns = ["text"])
    q = pd.concat([df,result_df],axis = 1)
    q = q.sort_values(by="sim",ascending = False)

    q_n = q[:20]
    q_n = q_n[["text"]]
    context= "\n\n".join(q_n["text"])
    
    prompt = create_prompt(context,user_input)
    reply = generate_answer(prompt)
    st.write(reply)
    

