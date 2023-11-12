import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('data', engine.connect())
    drop_cols = ['customer_id','offer_id','offer_receive_time','offer_view_time','offer_complete_time']
    df.drop(columns=drop_cols, inplace=True)

    y = df['offer_respond']
    X = df.iloc[:, 1:]
    dummy_cols = ['gender', 'offer_type']
    for dummy_col in dummy_cols:
        X = pd.concat([X, pd.get_dummies(X[dummy_col], prefix=dummy_col)], axis=1)
        X.drop(columns=[dummy_col], inplace=True)
    X.columns = X.columns.astype(str)
    return X, y, X.columns

def train_and_evaluate_models(X, y):
    '''
    Train and evaluate different models, return best model base on f1-score.

    Models evaluated and parameter used in GridSearchCV:
    - Logistics Regression: C
    - Random Forest Classifier: n_estimators, max_depth, min_samples_split
    - XGBoost Classifier: n_estimators, max_depth, learning_rate
    '''
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the data - scaling is often important for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    print("Grid searching Logistic Regression Models")
    lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    lr_model = LogisticRegression()
    lr_grid_search = GridSearchCV(lr_model, lr_param_grid, cv=5, scoring='f1', n_jobs=-1)
    lr_grid_search.fit(X_train_scaled, y_train)
    print("Best Logistic Regression Estimator:")
    print(lr_grid_search.best_estimator_)

    # Random Forest
    print("Grid searching Random Forest Models")
    rf_param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    rf_model = RandomForestClassifier()
    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='f1', n_jobs=-1)
    rf_grid_search.fit(X_train_scaled, y_train)
    print("Best Random Forest Estimator:")
    print(rf_grid_search.best_estimator_)

    # XGBoost
    print("Grid searching XGBoost Models")
    xgb_param_grid = {'n_estimators': [100, 150, 200], 'max_depth': [5, 7, 10], 'learning_rate': [0.1, 0.2, 0.3]}
    xgb_model = XGBClassifier()
    xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='f1', n_jobs=-1)
    xgb_grid_search.fit(X_train_scaled, y_train)
    print("Best XGBoost Estimator:")
    print(xgb_grid_search.best_estimator_)

    # Evaluate models
    models = {
        'Logistic Regression': lr_grid_search.best_estimator_,
        'Random Forest': rf_grid_search.best_estimator_,
        'XGBoost': xgb_grid_search.best_estimator_
    }

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        print(f'\n{name} Results:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'F1 Score: {f1:.2f}')
        print(f'ROC-AUC Score: {roc_auc:.2f}')

    # Return the best model based on f1-score
    best_model_name = max(models, key=lambda k: f1_score(y_test, models[k].predict(X_test_scaled)))
    return models[best_model_name]

def save_model(model, model_filepath):
    '''Save the best model to a specified file.'''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, feature_names = load_data(database_filepath)

        print('Training and evaluating model...')
        model = train_and_evaluate_models(X, y)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/customer_offer_data.db classifier.pkl')

if __name__ == '__main__':
    main()
