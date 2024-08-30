from flask import Flask, render_template, request, redirect, session
import os
import joblib
import pandas as pd
import re
import numpy as np
import nltk 
nltk.download("stopwords")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from bs4 import BeautifulSoup

# HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radious:0.25rem; padding:1rem">{}</div>"""
app = Flask(__name__)

app.secret_key = os.urandom(24)
#Model saved with Keras model.save()
# MODEL_PATH = 'model/passmodel.pkl'
# TOKENIZER_PATH = 'model/tfidfvectorizer.pkl'
MODEL_PATH = 'model/dt.pkl'
TOKENIZER_PATH = 'model/countvectorizer.pkl'
# DATA_PATH = 'data/top3drugs.csv'
DATA_PATH = 'data/top3drugs.csv'

# loading vectorizer
vectorizer = joblib.load(TOKENIZER_PATH)
# loading model
model = joblib.load(MODEL_PATH)
top_drugs_df = pd.read_csv(DATA_PATH)
top_drugs_dict = dict(zip(top_drugs_df['condition'], top_drugs_df['top_drugs'].apply(eval)))

#getting stopwords   
# stop = stopwords.words('english')
stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# @app.route('/')
# def index():
#     # if 'user_id' in session:
#         return  render_template('home.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect('/')


@app.route('/')
def index():
    return render_template('index.html', text='')


def predict_text(lst_txt):
  df_test = pd.DataFrame(lst_txt, columns = ['test_sent'])
  x_test = vectorizer.transform(lst_txt)
  prediction = model.predict(x_test)
  df_test['prediction'] = prediction
  return df_test

def classify_and_predict(text):
  df_testsent = predict_text(text)
  top_drugs = top_drugs_dict.get(df_testsent['prediction'][0], [])
  print("text:", text[0],"\n\nConditions:", df_testsent['prediction'][0])
  print("\nTop 3 Suggested Drugs: ")
  print(top_drugs[0])
  print(top_drugs[1])
  print(top_drugs[2])
  return [top_drugs[0],top_drugs[1],top_drugs[2], df_testsent['prediction'][0]]

   

@app.route('/predict', methods=["POST"])
def predict():
    print("here!  1")
    if request.method == 'POST':
        print("here!  2")

        raw_text = request.form['rawtext']

        if raw_text != "":
            print("here!  2")


            # clean_text = cleanText(raw_text)
            clean_text = []
            clean_text.append(raw_text)
            print("here!  3")
            result = classify_and_predict(clean_text)
            # clean_lst = [clean_text]

            # tfidf_vect = vectorizer.transform(clean_lst)
            # prediction = model.predict(tfidf_vect)
            # predicted_cond = prediction[0]
            # df = pd.read_csv(DATA_PATH)
            
            # top_drugs = top_drugs_extractor(predicted_cond, df)
            # print("here!  3")
            # print("raw_text: ", raw_text)
            # print("predicted_cond: ", predicted_cond)
            # print("top_drugs: ", top_drugs)

            return render_template('index.html', rawtext= raw_text, predicted_cond=result[-1],top_drugs=result[:-1])
        
        else:
            raw_text = "There is no text to select"


def cleanText(raw_review):
  # 1 delete HTML
  review_text = BeautifulSoup(raw_review, "html.parser").get_text()
  # 2 Make a space
  letters_only = re.sub("[^a-zA-Z]", " ", review_text)
  # 2 Make a space
  words = letters_only.lower().split()
  # 4 Stopwords
  meaningful_words = [w for w in words if not (w in stop)]
  # 6 Lemmitization
  lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
  # 7 space join words
  return(' '.join(lemmitize_words))



def top_drugs_extractor(condition, df):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False,False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst



if __name__ == "__main__":
    app.run(debug=True, host="localhost")