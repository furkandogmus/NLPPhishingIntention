import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

import time
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib



def train_and_save_models(X_train, y_train, X_test, y_test):

    if os.path.exists('model/catboost_model.pkl') and os.path.exists('model/xgboost_model.pkl'):
        # If model are already trained and saved
        random_search_xgboost = joblib.load('xgboost_model.pkl')
        random_search_catboost_model = joblib.load('catboost_model.pkl')

    else:
        # Train XGBoost model for optimizing the model use Randomized Search method
        params = {
            'n_estimators': [500],
            'min_child_weight': [4, 5],
            'gamma': [i / 10.0 for i in range(3, 6)],
            'subsample': [i / 10.0 for i in range(6, 11)],
            'colsample_bytree': [i / 10.0 for i in range(6, 11)],
            'max_depth': [2, 3, 4, 6, 7],
            'objective': ['reg:squarederror', 'reg:tweedie'],
            'booster': ['gbtree', 'gblinear'],
            'eval_metric': ['rmse'],
            'eta': [i / 10.0 for i in range(3, 6)],
        }
        n_iter_search = 15

        xgb_model = XGBClassifier(nthread=-1)
        random_search_xgboost = RandomizedSearchCV(xgb_model, param_distributions=params,
                                           n_iter=n_iter_search, cv=5,  scoring='neg_mean_squared_error')
        # Calculate the elapsed time.
        start = time.time()
        random_search_xgboost.fit(X_train, y_train)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time.time() - start), n_iter_search))
        print("Best set of hyperparameters: ", random_search_xgboost.best_params_)

        # Save the model
        save_model(random_search_xgboost, "xgboost_model.pkl")



        # Train CatBoost model, for optimizing the model use Randomized Search method.
        params = {'depth': sp_randInt(4, 10),
                  'learning_rate': sp_randFloat(),
                  'iterations': sp_randInt(10, 100)
                  }
        n_iter_search = 15

        catboost_model = CatBoostClassifier()
        random_search_catboost_model = RandomizedSearchCV(catboost_model,
                                                          param_distributions=params,
                                                          n_iter=n_iter_search, cv=5,
                                                          scoring='neg_mean_squared_error')
        # Calculate the time elapsed
        start = time.time()
        random_search_catboost_model.fit(X_train, y_train)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time.time() - start), n_iter_search))
        print("Best set of hyperparameters: ", random_search_catboost_model.best_params_)

        # Save the model
        save_model(random_search_catboost_model, "catboost_model.pkl")

    # Evaluate model
    evaluate_model(random_search_xgboost, X_test, y_test, "XGBoost")
    evaluate_model(random_search_catboost_model, X_test, y_test, "CatBoost")

def save_model(model, filename):
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"{model_name} Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    data_folder = "embeddings/pkl"
    embedding_file_path = 'embeddings/embeddings.pkl'
    with open(embedding_file_path, 'rb') as file:
        embeddings_data = joblib.load(file)

    X_benign = embeddings_data['benign']
    X_phishing = embeddings_data['phishing']

    # Combine features
    X = X_benign + X_phishing
    X = [x.flatten() for x in X]

    # Create labels
    y = [0] * len(X_benign) + [1] * len(X_phishing)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model and save them
    train_and_save_models(X_train, y_train, X_test, y_test)
