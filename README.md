# Movie Review Sentiment Classifier

This project trains a machine-learning sentiment classifier on the IMDb 50K Movie Reviews dataset (50,000 labeled reviews: positive and negative).

## Files
- `train.py`: downloads the IMDb CSV automatically and trains Logistic Regression and Multinomial Naive Bayes models.
- `requirements.txt`: Python packages.
- `README.md`: execution instructions.

## Run in VS Code
1. Extract this ZIP and open the folder in VS Code (File > Open Folder).
2. Open Terminal > New Terminal.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run:
   `python train.py`

The first run downloads the IMDb dataset (about 66 MB), so internet access is required. The program compares models, prints accuracy/precision/recall/F1, saves `movie_sentiment_model.pkl` and `confusion_matrix.png`, then prompts for reviews to classify. Type `exit` to stop.

Dataset source: IMDb 50K Movie Reviews dataset hosted on Zenodo, originally associated with the Stanford Large Movie Review Dataset.
