# *Amazon Prime Viewers’ Sentiment Analysis*  

## *1. Project Overview*  
This project aims to analyze Amazon Prime Video ratings and reviews to understand customer sentiment, identify trends in viewer preferences, and determine factors influencing high or low ratings. The insights will help improve recommendations, optimize content offerings, and enhance user satisfaction.

## *2. Business Understanding*  
### *Problem Statement*  
With millions of users, Amazon Prime Video needs data-driven insights to enhance its recommendation system and improve customer experience. This project will leverage sentiment analysis and machine learning models to extract meaningful insights from user reviews.

### *Industry & Target Audience*  
- Streaming service providers (Amazon Prime, Netflix, Disney+, etc.)  
- Content creators and production companies  
- Data analysts & business intelligence professionals  
- Marketing teams in the entertainment industry  

### *Real-World Impact*  
- Improve Amazon Prime’s recommendation algorithms  
- Help content creators align with viewer preferences  
- Provide advertisers with audience sentiment data  
- Assist streaming services in optimizing content portfolios  

---

## *3. Dataset Information*  
### *Data Source*  
The dataset is sourced from [McAuley Lab’s Amazon Reviews Dataset (2023)](https://amazon-reviews-2023.github.io/). It includes:  
- *User Reviews*: Ratings, text, helpfulness votes  
- *Item Metadata*: Descriptions, price, category, images  
- *Links*: User-item relations  

### *Dataset Details*  
- Extracted *233,000* Amazon Prime Video reviews  
- Merged User Reviews and Item Metadata using parent_asin  
- Filtered out non-Prime Video reviews  

### *Key Features*  
- *Review Text* – User-written review content  
- *Star Rating* – Numeric rating (1-5)  
- *Timestamp* – Review date/time  
- *Product Metadata* – Movie/TV title, genre, release year  
- *User Metadata* – Verified purchase status, review count  

### *Preprocessing Steps*  
- Handling missing values & duplicates  
- Removing stopwords, special characters, punctuation  
- Tokenization, lemmatization, vectorization (TF-IDF, embeddings)  
- Addressing class imbalance in sentiment labels  

---

## *4. Methodology*  
### *Baseline Models*  
- *Multinomial Naïve Bayes (NB)* – Fast and efficient for text classification  
- *Support Vector Machines (SVM)* – Handles high-dimensional text data  

### *Advanced Models*  
- *LSTM & BiLSTM (RNNs)* – Captures sequential text dependencies  
- *BERT Transformers* – State-of-the-art NLP model for contextual understanding  

### *Target Variable*  
- *Sentiment Classification*:  
  - *Positive* (4-5 stars)  
  - *Neutral* (3 stars)  
  - *Negative* (1-2 stars)  

### *Evaluation Metrics*  
- *Accuracy* – Overall model correctness  
- *Precision, Recall, F1-Score* – Performance across sentiment classes  
- *Confusion Matrix* – Misclassification visualization  

---

## *5. Results & Insights*  
- Sentiment trends over time  
- Most frequent keywords in positive vs. negative reviews  
- Correlation between genre and sentiment  
- Effectiveness of traditional ML vs. deep learning models  

---

## *6. Visualization Strategy*  
- *Word Clouds* – Common themes in positive & negative reviews  
- *Bar Charts & Histograms* – Rating distributions, sentiment trends  
- *Time-Series Analysis* – Sentiment shifts over time  
- *Box Plots* – Rating variations across genres  
- *Heatmaps* – Correlations between sentiment, ratings, metadata  

---

## *7. Deployment Plan*  
### *Deliverables*  
- Jupyter Notebook with EDA, modeling, and evaluation  
- PowerPoint summary of key insights  

### *Web App (Stretch Goal)*  
- Built using Flask, Streamlit, or Dash  
- Interactive visualization of sentiment trends  
- Review-based sentiment analysis  

### *API (Stretch Goal)*  
- Flask/Django API for sentiment prediction  
- Accepts user reviews and returns sentiment label  

---

## *8. Tools & Technologies*  
### *Libraries*  
- *Data Processing*: pandas, numpy  
- *Visualization*: matplotlib, seaborn, wordcloud  
- *NLP & Feature Engineering*: NLTK, spaCy, TF-IDF, Word2Vec  
- *Machine Learning*: scikit-learn (NB, SVM), XGBoost, Random Forest  
- *Deep Learning*: TensorFlow, Keras, transformers (Hugging Face BERT)  

### *Computational Environment*  
- *Google Colab* (for deep learning models)  
- *Local Machine* (for EDA & preprocessing)  
- *Google Drive* (for dataset storage)  

---

## *9. How to Run the Project*  
### *Installation*  
bash
pip install pandas numpy matplotlib seaborn nltk spacy transformers torch scikit-learn


### *Run Baseline Models*  
python
python svm_model.py


### *Run BERT Model*  
python
python bert_model.py


### *Run Web App (if implemented)*  
bash
streamlit run app.py


---

## *10. Future Enhancements*  
✅ Optimize hyperparameters for ML & DL models  
✅ Implement real-time sentiment analysis API  
✅ Integrate with IMDb or external sources for richer data  
✅ Build a recommendation system based on sentiment trends  

---

## *11. Contributors*  
- *Team Members:* (List names)  
- *Mentor:* (If applicable)  
- *Contact:* (GitHub repo, email, LinkedIn)  