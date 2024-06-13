import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
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
from imblearn.over_sampling import SMOTE

def balance_dataset(data):
    #Bilancia il dataset con SMOTE
    smote = SMOTE(sampling_strategy='minority')
    X = data.drop('RESULTS', axis=1)
    y = data['RESULTS']
    X_resampled, y_resampled = smote.fit_resample(X, y)
    data_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    return data_resampled

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
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
        }
    elif model_name == 'LogisticRegression':
        model = LogisticRegression()
        hyperparameters = {
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.1, 1, 10],
            'model__solver': ['liblinear', 'saga'],
            'model__max_iter': [200,500,1000] 
        }
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier()
        hyperparameters = {
            'model__criterion': ['gini', 'entropy'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
        }
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier()
        hyperparameters = {
            'model__loss': ['log_loss', 'exponential'],
            'model__learning_rate': [0.1, 0.01, 0.001],
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

def training_randomforest_on_maxdepth(X, y, best_params):

    max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for max_depth in max_depth_values:
        model = RandomForestClassifier(
            criterion=best_params['model__criterion'],
            min_samples_split=best_params['model__min_samples_split'],
            min_samples_leaf=best_params['model__min_samples_leaf'],
            max_depth=max_depth
        )
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score)
        }
        results = cross_validate(pipe, X, y, cv=kf, scoring=scoring)
        accuracy_scores.append(results['test_accuracy'].mean())
        f1_scores.append(results['test_f1'].mean())
        precision_scores.append(results['test_precision'].mean())
        recall_scores.append(results['test_recall'].mean())
        print(f"Max Depth: {max_depth}")
        print(f"Accuracy: {results['test_accuracy'].mean()}")
        print(f"F1 Score: {results['test_f1'].mean()}")
        print(f"Precision: {results['test_precision'].mean()}")
        print(f"Recall: {results['test_recall'].mean()}")

    plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Random Forest Depth')

    accuracy_scores = np.array(accuracy_scores)
    f1_scores = np.array(f1_scores)
    precision_scores = np.array(precision_scores)
    recall_scores = np.array(recall_scores)

    print(f"Accuracy: {accuracy_scores.mean()}")
    print(f"F1 Score: {f1_scores.mean()}")
    print(f"Precision: {precision_scores.mean()}")
    print(f"Recall: {recall_scores.mean()}")

    #salvataggio su file
    with open('RandomForestTrainingDepth.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy_scores.mean()}\n")
        f.write(f"F1 Score: {f1_scores.mean()}\n")
        f.write(f"Precision: {precision_scores.mean()}\n")
        f.write(f"Recall: {recall_scores.mean()}\n")

def training_randomforest_on_n_estimators(X, y, best_params):
    
        n_estimators_values = [10, 20, 50, 100, 200, 500, 1000]
    
        accuracy_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
    
        for n_estimators in n_estimators_values:
            model = RandomForestClassifier(
                criterion=best_params['model__criterion'],
                min_samples_split=best_params['model__min_samples_split'],
                min_samples_leaf=best_params['model__min_samples_leaf'],
                n_estimators=n_estimators
            )
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score)
            }
            results = cross_validate(pipe, X, y, cv=kf, scoring=scoring)
            accuracy_scores.append(results['test_accuracy'].mean())
            f1_scores.append(results['test_f1'].mean())
            precision_scores.append(results['test_precision'].mean())
            recall_scores.append(results['test_recall'].mean())

        plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Random Forest Estimators')

        accuracy_scores = np.array(accuracy_scores)
        f1_scores = np.array(f1_scores)
        precision_scores = np.array(precision_scores)
        recall_scores = np.array(recall_scores)
    
        print(f"Accuracy: {accuracy_scores.mean()}")
        print(f"F1 Score: {f1_scores.mean()}")
        print(f"Precision: {precision_scores.mean()}")
        print(f"Recall: {recall_scores.mean()}")
    
        #salvataggio su file
        with open('RandomForestTrainingEstimators.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy_scores.mean()}\n")
            f.write(f"F1 Score: {f1_scores.mean()}\n")
            f.write(f"Precision: {precision_scores.mean()}\n")
            f.write(f"Recall: {recall_scores.mean()}\n")

def training_DecisionTree(X, y, best_params):
    
        max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
        accuracy_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
    
        for max_depth in max_depth_values:
            model = DecisionTreeClassifier(
                criterion=best_params['model__criterion'],
                min_samples_split=best_params['model__min_samples_split'],
                min_samples_leaf=best_params['model__min_samples_leaf'],
                max_depth=max_depth
            )
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score)
            }
            results = cross_validate(pipe, X, y, cv=kf, scoring=scoring)
            accuracy_scores.append(results['test_accuracy'].mean())
            f1_scores.append(results['test_f1'].mean())
            precision_scores.append(results['test_precision'].mean())
            recall_scores.append(results['test_recall'].mean())

        plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Decision Tree')
    
        accuracy_scores = np.array(accuracy_scores)
        f1_scores = np.array(f1_scores)
        precision_scores = np.array(precision_scores)
        recall_scores = np.array(recall_scores)

    
        print(f"Accuracy: {accuracy_scores.mean()}")
        print(f"F1 Score: {f1_scores.mean()}")
        print(f"Precision: {precision_scores.mean()}")
        print(f"Recall: {recall_scores.mean()}")

        #salvataggio su file
        with open('DecisionTreeTrainingDepth.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy_scores.mean()}\n")
            f.write(f"F1 Score: {f1_scores.mean()}\n")
            f.write(f"Precision: {precision_scores.mean()}\n")
            f.write(f"Recall: {recall_scores.mean()}\n")

def training_LogisticRegression(X, y, best_params):
    model = LogisticRegression(
        penalty=best_params['model__penalty'],
        C=best_params['model__C'],
        solver=best_params['model__solver'],
        max_iter=best_params['model__max_iter']
    )
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score)
    }
    results = cross_validate(pipe, X, y, cv=kf, scoring=scoring)

    print(f"Accuracy: {results['test_accuracy'].mean()}")
    print(f"F1 Score: {results['test_f1'].mean()}")
    print(f"Precision: {results['test_precision'].mean()}")
    print(f"Recall: {results['test_recall'].mean()}")
    #salvataggio su file
    with open('LogisticRegressionTraining.txt', 'w') as f:
        f.write(f"Accuracy: {results['test_accuracy'].mean()}\n")
        f.write(f"F1 Score: {results['test_f1'].mean()}\n")
        f.write(f"Precision: {results['test_precision'].mean()}\n")
        f.write(f"Recall: {results['test_recall'].mean()}\n")

def training_GradientBoosting_on_maxdepth(X, y, best_params):
        
            max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
            accuracy_scores = []
            f1_scores = []
            precision_scores = []
            recall_scores = []
        
            for max_depth in max_depth_values:
                model = GradientBoostingClassifier(
                    loss=best_params['model__loss'],
                    learning_rate=best_params['model__learning_rate'],
                    max_depth=max_depth,
                    min_samples_split=best_params['model__min_samples_split'],
                    min_samples_leaf=best_params['model__min_samples_leaf']
                )
                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
                kf = KFold(n_splits=10, shuffle=True, random_state=42)
                scoring = {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score)
                }
                results = cross_validate(pipe, X, y, cv=kf, scoring=scoring)
                accuracy_scores.append(results['test_accuracy'].mean())
                f1_scores.append(results['test_f1'].mean())
                precision_scores.append(results['test_precision'].mean())
                recall_scores.append(results['test_recall'].mean())

            plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Gradient Boosting Depth')
        
            accuracy_scores = np.array(accuracy_scores)
            f1_scores = np.array(f1_scores)
            precision_scores = np.array(precision_scores)
            recall_scores = np.array(recall_scores)
        
            print(f"Accuracy: {accuracy_scores.mean()}")
            print(f"F1 Score: {f1_scores.mean()}")
            print(f"Precision: {precision_scores.mean()}")
            print(f"Recall: {recall_scores.mean()}")
        
            #salvataggio su file
            with open('GradientBoostingTrainingDepth.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy_scores.mean()}\n")
                f.write(f"F1 Score: {f1_scores.mean()}\n")
                f.write(f"Precision: {precision_scores.mean()}\n")
                f.write(f"Recall: {recall_scores.mean()}\n")

def training_GradientBoosting_on_n_estimators(X, y, best_params):
        
            n_estimators_values = [10, 20, 50, 100, 200, 500, 1000]
        
            accuracy_scores = []
            f1_scores = []
            precision_scores = []
            recall_scores = []
        
            for n_estimators in n_estimators_values:
                model = GradientBoostingClassifier(
                    loss=best_params['model__loss'],
                    learning_rate=best_params['model__learning_rate'],
                    n_estimators=n_estimators,
                    min_samples_split=best_params['model__min_samples_split'],
                    min_samples_leaf=best_params['model__min_samples_leaf']
                )
                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
                kf = KFold(n_splits=10, shuffle=True, random_state=42)
                scoring = {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score)
                }
                results = cross_validate(pipe, X, y, cv=kf, scoring=scoring)
                accuracy_scores.append(results['test_accuracy'].mean())
                f1_scores.append(results['test_f1'].mean())
                precision_scores.append(results['test_precision'].mean())
                recall_scores.append(results['test_recall'].mean())

            plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Gradient Boosting Estimators')
        
            accuracy_scores = np.array(accuracy_scores)
            f1_scores = np.array(f1_scores)
            precision_scores = np.array(precision_scores)
            recall_scores = np.array(recall_scores)
        
            print(f"Accuracy: {accuracy_scores.mean()}")
            print(f"F1 Score: {f1_scores.mean()}")
            print(f"Precision: {precision_scores.mean()}")
            print(f"Recall: {recall_scores.mean()}")
        
            #salvataggio su file
            with open('GradientBoostingTrainingEstimators.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy_scores.mean()}\n")
                f.write(f"F1 Score: {f1_scores.mean()}\n")
                f.write(f"Precision: {precision_scores.mean()}\n")
                f.write(f"Recall: {recall_scores.mean()}\n")

def plot_scores(x_values, accuracy_scores, f1_scores, precision_scores, recall_scores, x_label, title):
    plt.figure(figsize=(12, 8))
    
    plt.plot(x_values, accuracy_scores, marker='o', label='Accuracy')
    plt.plot(x_values, f1_scores, marker='o', label='F1 Score')
    plt.plot(x_values, precision_scores, marker='o', label='Precision')
    plt.plot(x_values, recall_scores, marker='o', label='Recall')
    plt.ylim(0.5, 1)
    plt.xlabel(x_label)
    plt.ylabel('Scores')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title}.png')


def main():
    X, y = preprocess_data()
    
    # Search and train the best hyperparameters for each model

    # Random Forest
    best_params_rf = search_best_hyperparameters(X, y, 'RandomForest')
    training_randomforest_on_maxdepth(X, y, best_params_rf)
    training_randomforest_on_n_estimators(X, y, best_params_rf)

    # Logistic Regression
    best_params_lr = search_best_hyperparameters(X, y, 'LogisticRegression')
    training_LogisticRegression(X, y, best_params_lr)
  
    # Decision Tree
    best_params_dt = search_best_hyperparameters(X, y, 'DecisionTree')
    training_DecisionTree(X, y, best_params_dt)
    
    # Gradient Boosting
    best_params_gb = search_best_hyperparameters(X, y, 'GradientBoosting')
    training_GradientBoosting_on_maxdepth(X, y, best_params_gb)
    training_GradientBoosting_on_n_estimators(X, y, best_params_gb)

    # Naive Bayes
    Naive_Bayes()

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

    print(f"Accuracy: {results['test_accuracy'].mean()}")
    print(f"F1 Score: {results['test_f1'].mean()}")
    print(f"Precision: {results['test_precision'].mean()}")
    print(f"Recall: {results['test_recall'].mean()}")

    #salvataggio su file
    with open('NaiveBayesTraining.txt', 'w') as f:
        f.write(f"Accuracy: {results['test_accuracy'].mean()}\n")
        f.write(f"F1 Score: {results['test_f1'].mean()}\n")
        f.write(f"Precision: {results['test_precision'].mean()}\n")
        f.write(f"Recall: {results['test_recall'].mean()}\n")

main()
