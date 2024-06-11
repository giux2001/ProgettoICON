import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import CategoricalNB
import numpy as np
import matplotlib.pyplot as plt

def preprocess_data(only_categorical = False):
    df = pd.read_csv("Working_Dataset.csv")
    le = LabelEncoder()
    df['FACILITY_TYPE'] = le.fit_transform(df['FACILITY_TYPE'])
    df = df[df['NUM_INSP_AREA'] > 1]
    df['RESULTS'] = df['RESULTS'].apply(lambda x: 1 if x == 2 else x)
    if only_categorical:
        X = df.drop(['INSPECTION_ID', 'DBA NAME', 'RESULTS', 'COMMUNITY_AREA', 'NUM_INSP_AREA','PERC_INS_FAILED_AREA','PERC_INS_PASSED_AREA','PERC_INS_PASSED_COND_AREA','PERC_SERIOUS_VIOLATIONS_FAILED_AREA','AREA_BELOW_POVERTY_LEVEL','AREA_PER_CAPITA_INCOME','AREA_UNEMPLOYMENT','AREA_CRIME_INDEX','AREA_HEALTH_INDEX'], axis=1)
        y = df['RESULTS']
    else:
        X = df.drop(['INSPECTION_ID', 'DBA NAME', 'RESULTS', 'COMMUNITY_AREA'], axis=1)
        y = df['RESULTS']

    return X, y

def search_best_hyperparameters(X_train, y_train, model_name):
    if model_name == 'RandomForest':
        model = RandomForestClassifier()
        hyperparameters = {
            'model__criterion': ['gini', 'entropy', 'log_loss'],
            'model__n_estimators': [10, 20, 50],
            'model__max_depth': [None, 5, 10],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
        }
    elif model_name == 'LogisticRegression':
        model = LogisticRegression()
        hyperparameters = {
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.1, 1, 10],
            'model__solver': ['liblinear', 'saga'],
            'model__max_iter': [200,500]  # Aumenta il numero di iterazioni
        }
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier()
        hyperparameters = {
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': [None, 5, 10],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
        }
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier()
        hyperparameters = {
            'model__loss': ['deviance', 'exponential'],
            'model__learning_rate': [0.1, 0.01, 0.001],
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 10],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
        }
    else:
        raise ValueError("Invalid model name")

    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    grid = GridSearchCV(pipe, hyperparameters, cv=5)
    grid.fit(X_train, y_train)

    print("Best hyperparameters found: ", grid.best_params_)
    return grid.best_params_


def apply_kfold_with_best_hyperparameters(X, y, best_params, model_name):
    if model_name == 'RandomForest':
        model = RandomForestClassifier(
            criterion=best_params['model__criterion'],
            n_estimators=best_params['model__n_estimators'],
            max_depth=best_params['model__max_depth'],
            min_samples_split=best_params['model__min_samples_split'],
            min_samples_leaf=best_params['model__min_samples_leaf']
        )
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(
            penalty=best_params['model__penalty'],
            C=best_params['model__C'],
            solver=best_params['model__solver']
        )
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier(
            criterion=best_params['model__criterion'],
            max_depth=best_params['model__max_depth'],
            min_samples_split=best_params['model__min_samples_split'],
            min_samples_leaf=best_params['model__min_samples_leaf']
        )
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(
            loss=best_params['model__loss'],
            learning_rate=best_params['model__learning_rate'],
            n_estimators=best_params['model__n_estimators'],
            max_depth=best_params['model__max_depth'],
            min_samples_split=best_params['model__min_samples_split'],
            min_samples_leaf=best_params['model__min_samples_leaf']
        )
    else:
        raise ValueError("Invalid model name")

    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score)
    }

    results = cross_validate(pipe, X, y, cv=kf, scoring=scoring)

    print(f"Accuracy: {results['test_accuracy'].mean()} ± {results['test_accuracy'].std()}")
    print(f"F1 Score: {results['test_f1'].mean()} ± {results['test_f1'].std()}")
    print(f"Precision: {results['test_precision'].mean()} ± {results['test_precision'].std()}")
    print(f"Recall: {results['test_recall'].mean()} ± {results['test_recall'].std()}")

def plot_learning_curve(X, y, model, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def main():
    X, y = preprocess_data()
    
    # Search best hyperparameters for RandomForest
    best_params_rf = search_best_hyperparameters(X, y, 'RandomForest')
    apply_kfold_with_best_hyperparameters(X, y, best_params_rf, 'RandomForest')
    plot_learning_curve(X, y, RandomForestClassifier(criterion=best_params_rf['model__criterion'], n_estimators=best_params_rf['model__n_estimators'], max_depth=best_params_rf['model__max_depth'], min_samples_split=best_params_rf['model__min_samples_split'], min_samples_leaf=best_params_rf['model__min_samples_leaf']), 'RandomForest')
    
    # Search best hyperparameters for LogisticRegression
    #best_params_lr = search_best_hyperparameters(X, y, 'LogisticRegression')
    #apply_kfold_with_best_hyperparameters(X, y, best_params_lr, 'LogisticRegression')
    #plot_learning_curve(X, y, LogisticRegression(penalty=best_params_lr['model__penalty'], C=best_params_lr['model__C'], solver=best_params_lr['model__solver']), 'LogisticRegression')
    
    
    # Search best hyperparameters for DecisionTree
    #best_params_dt = search_best_hyperparameters(X, y, 'DecisionTree')
    #apply_kfold_with_best_hyperparameters(X, y, best_params_dt, 'DecisionTree')
    #plot_learning_curve(X, y, DecisionTreeClassifier(criterion=best_params_dt['model__criterion'], max_depth=best_params_dt['model__max_depth'], min_samples_split=best_params_dt['model__min_samples_split'], min_samples_leaf=best_params_dt['model__min_samples_leaf']), 'DecisionTree')
    
    # Search best hyperparameters for GradientBoosting
    #best_params_gb = search_best_hyperparameters(X, y, 'GradientBoosting')
    #apply_kfold_with_best_hyperparameters(X, y, best_params_gb, 'GradientBoosting')

def Naive_Bayes():
    model = CategoricalNB()
    X, y = preprocess_data(only_categorical=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Perform cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score)
    }
    results = cross_validate(model, X, y, cv=kf, scoring=scoring)

    print(f"Accuracy: {results['test_accuracy'].mean()} ± {results['test_accuracy'].std()}")
    print(f"F1 Score: {results['test_f1'].mean()} ± {results['test_f1'].std()}")
    print(f"Precision: {results['test_precision'].mean()} ± {results['test_precision'].std()}")
    print(f"Recall: {results['test_recall'].mean()} ± {results['test_recall'].std()}")

main()
Naive_Bayes()