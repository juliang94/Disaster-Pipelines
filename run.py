import json
import plotly

import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    ## most common genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    ## most common categories
    category_data = df.iloc[:, 4:]
    num_categories = category_data.shape[0]
    
    category_pct = pd.DataFrame(category_data.sum()/num_categories * 100, columns= ['pct_total'])
    category_pct.sort_values(by = 'pct_total', inplace= True)
    category_pct.reset_index(inplace = True)
    category_pct = category_pct.rename(columns = {'index': 'category'})
    category_pct['category'] = category_pct['category'].str.replace('_', ' ')
    
    category_names = category_pct['category']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                    'tickangle': 45
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_pct['pct_total'],
                    y=category_names,
                    orientation = 'h'
                )
            ],

            'layout': {
                'height': 900,
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Category",
                    'tickfont': {'size':7}
                },
                'xaxis': {
                    'title': "Percentage"
                }
            }
        }
    ]
    
    
    
    ## plot horizontal graph

    
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()