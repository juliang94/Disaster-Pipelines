# Disaster Response Pipelines

## Summary
This project entails cleaning a dataset of messages sent during disaster events. After cleaning the data, the next step is to build a natural language processing pipeline that preprocesses the text data. The final output of this pipeline will be used along with a vector of message categories to build a predictive algorithm pipeline. This model will be used in a web app where the user can write a message that will be classified according to the optimal model.

## Files
1 - Data ETL script for data modeling - loads and cleans the data before saving into a database.
2 - NLP Machine Learning Pipeline - loads the data from the saves database and uses an NLP pipeline to process the text data before using it to predict categories with an AdaBoost classifier.
3 - Run a web app with predicted model - The optimal AdaBoost classifier will be used in a web app where the user writes a message to classify.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
