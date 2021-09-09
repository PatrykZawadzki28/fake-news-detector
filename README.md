# Fake news detector 
- Patryk Zawadzki
- Patryk Mucha 
- Dawid Torbus

## Setup
1. python == 3.8
2. wordcloud == 1.6.0
3. tensorflow == 2.1.0
4. seaborn == 0.10.0
5. pandas == 0.25.3
6. numpy == 1.18.2
7. nltk == 3.2.4
8. matplotlib == 3.2.1
9. sklearn == 0.24.2
10. joblib == 1.0.1
11. itertools

## Instructions

### Prepare databases 
1. Go to `https://drive.google.com/drive/folders/1zW7wE-IbmptHAH9ncqH1ZvQ5XQB-vYpW?usp=sharing`
and download the data. In project main directory create folder `/databases` and put the downloaded data inside
   
### Prepare models:
1.  In project main directory create folder `/models`
2. `python LSTM.py` #can take several hours
3. `python LSTM-simple.py` #pretty fast
4. `python NLP.py` #pretty fast

Models are saved inside `/models` folder.

### Start server
1. `python server.py`

### Start extension
1. In your browser go to `chrome://extensions/` and click `Load unpacked` button. Then choose folder `/extension`
2. turn ON the extension

Now you should be able to use the extension on any website beside the `chrome://extensions/` and similar


