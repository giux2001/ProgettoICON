import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import CategoricalNB
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns

from imblearn.over_sampling import SMOTE

def balance_dataset(X, y):
    # Calcola il numero di esempi in ciascuna classe
    count_class_majority = y.value_counts()[1]  # Classe maggioritaria (pass)
    count_class_minority = y.value_counts()[0]  # Classe minoritaria (fail)

    desiderd_minority = 0.16 * (count_class_majority + count_class_minority)
    
    sampling_strategy = desiderd_minority / count_class_minority

    # Bilancia il dataset con SMOTE utilizzando la strategia calcolata
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled

def preprocess_data(balanced = True):
    df = pd.read_csv("Working_Dataset.csv")
    le = LabelEncoder()
    df['FACILITY_TYPE'] = le.fit_transform(df['FACILITY_TYPE'])
    df = df[df['NUM_INSP_AREA'] > 1]
    df['RESULTS'] = df['RESULTS'].apply(lambda x: 1 if x == 2 else x) 
    X = df.drop(['INSPECTION_ID', 'DBA NAME', 'RESULTS', 'COMMUNITY_AREA'], axis=1)
    y = df['RESULTS']

    #splitta in train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if balanced:
        X_train, y_train = balance_dataset(X_train, y_train)

    print("Numero di campioni per ogni valore di RESULTS:")
    print(y_train.value_counts())

    return X_train, y_train, X_test, y_test

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
    
    return grid.best_params_

def training_randomforest_on_maxdepth(X_train, y_train, X_test, y_test, best_params):

    max_depth_values = [i for i in range(1,25)]

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
        results = cross_validate(pipe, X_train, y_train, cv=kf, scoring=scoring)
        accuracy_scores.append(results['test_accuracy'].mean())
        f1_scores.append(results['test_f1'].mean())
        precision_scores.append(results['test_precision'].mean())
        recall_scores.append(results['test_recall'].mean())

    plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Random Forest Depth Balanced')

    #convertire in numpy array per usare argmax
    accuracy_scores = np.array(accuracy_scores)
    best_accuracy = accuracy_scores.argmax()

    model = RandomForestClassifier(
            criterion=best_params['model__criterion'],
            min_samples_split=best_params['model__min_samples_split'],
            min_samples_leaf=best_params['model__min_samples_leaf'],
            max_depth=max_depth_values[best_accuracy])
    
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    plot_confusion_matrix(y_test, y_pred, 'RandomForestMaxDepth')

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    #salvataggio su file
    with open('RandomForestBestDepthBalanced.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")

def training_randomforest_on_n_estimators(X_train, y_train, X_test, y_test, best_params):
    
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
            results = cross_validate(pipe, X_train, y_train, cv=kf, scoring=scoring)
            accuracy_scores.append(results['test_accuracy'].mean())
            f1_scores.append(results['test_f1'].mean())
            precision_scores.append(results['test_precision'].mean())
            recall_scores.append(results['test_recall'].mean())

        plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Random Forest Estimators Balanced')

        accuracy_scores = np.array(accuracy_scores)
        best_accuracy = accuracy_scores.argmax()

        model = RandomForestClassifier(
                criterion=best_params['model__criterion'],
                min_samples_split=best_params['model__min_samples_split'],
                min_samples_leaf=best_params['model__min_samples_leaf'],
                n_estimators=n_estimators_values[best_accuracy]
            )
        
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        plot_confusion_matrix(y_test, y_pred, 'RandomForestEstimators')

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        #salvataggio su file
        with open('RandomForestBestEstimatorsBalanced.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")

def training_DecisionTree(X_train, y_train, X_test, y_test, best_params):
    
        max_depth_values = [i for i in range(1,25)]
    
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
            results = cross_validate(pipe, X_train, y_train, cv=kf, scoring=scoring)

            accuracy_scores.append(results['test_accuracy'].mean())
            f1_scores.append(results['test_f1'].mean())
            precision_scores.append(results['test_precision'].mean())
            recall_scores.append(results['test_recall'].mean())

        plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Decision Tree Balanced')
        

        accuracy_scores = np.array(accuracy_scores)
        best_accuracy = accuracy_scores.argmax()

        model = DecisionTreeClassifier(
                criterion=best_params['model__criterion'],
                min_samples_split=best_params['model__min_samples_split'],
                min_samples_leaf=best_params['model__min_samples_leaf'],
                max_depth=max_depth_values[best_accuracy]
            )
        
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        plot_confusion_matrix(y_test, y_pred, 'DecisionTree')


        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        #salvataggio su file
        with open('DecisionTreeBestDepthBalanced.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
        

def training_LogisticRegression(X_train, y_train, X_test, y_test, best_params):
    model = LogisticRegression(
        penalty=best_params['model__penalty'],
        C=best_params['model__C'],
        solver=best_params['model__solver'],
        max_iter=best_params['model__max_iter']
    )

    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    #matrice di confusione
    plot_confusion_matrix(y_test, y_pred, 'LogisticRegression')

    #salvataggio su file
    with open('LogisticRegressionBalanced.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")

def training_GradientBoosting_on_maxdepth(X_train, y_train, X_test, y_test, best_params):
        
            max_depth_values = [i for i in range(1,25)]
        
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
                results = cross_validate(pipe, X_train, y_train, cv=kf, scoring=scoring)
                accuracy_scores.append(results['test_accuracy'].mean())
                f1_scores.append(results['test_f1'].mean())
                precision_scores.append(results['test_precision'].mean())
                recall_scores.append(results['test_recall'].mean())

            plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Gradient Boosting Depth Balanced')

            accuracy_scores = np.array(accuracy_scores)
            best_acc = accuracy_scores.argmax()

            model = GradientBoostingClassifier(
                    loss=best_params['model__loss'],
                    learning_rate=best_params['model__learning_rate'],
                    max_depth=max_depth_values[best_acc],
                    min_samples_split=best_params['model__min_samples_split'],
                    min_samples_leaf=best_params['model__min_samples_leaf']
                )
            
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)

            plot_confusion_matrix(y_test, y_pred, 'GradientBoostingMaxDepth')

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            #salvataggio su file
            with open('GradientBoostingBestDepthBalanced.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
            



def training_GradientBoosting_on_n_estimators(X_train, y_train, X_test, y_test, best_params):
        
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
                results = cross_validate(pipe, X_train, y_train, cv=kf, scoring=scoring)
                accuracy_scores.append(results['test_accuracy'].mean())
                f1_scores.append(results['test_f1'].mean())
                precision_scores.append(results['test_precision'].mean())
                recall_scores.append(results['test_recall'].mean())

            plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Gradient Boosting Estimators Balanced')
        
            accuracy_scores = np.array(accuracy_scores)
            best_acc = accuracy_scores.argmax()

            model = GradientBoostingClassifier(
                    loss=best_params['model__loss'],
                    learning_rate=best_params['model__learning_rate'],
                    n_estimators=n_estimators_values[best_acc],
                    min_samples_split=best_params['model__min_samples_split'],
                    min_samples_leaf=best_params['model__min_samples_leaf']
                )
            
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)

            plot_confusion_matrix(y_test, y_pred, 'GradientBoostingEstimators')

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            #salvataggio su file
            with open('GradientBoostingBestEstimatorsBalanced.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")

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

def plot_confusion_matrix(y_test, y_pred, model_name):
      #matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    class_names = ['Failed', 'Passed']
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix {model_name}')
    #salva la matrice di confusione in un file
    plt.savefig(f'ConfusionMatrix{model_name}.png')


def main():
    X_train, y_train, X_test, y_test = preprocess_data(balanced=True)
    
    # Search and train the best hyperparameters for each model

    #Random Forest
    print("Random Forest")
    best_params_rf = search_best_hyperparameters(X_train, y_train, 'RandomForest')
    training_randomforest_on_maxdepth(X_train, y_train, X_test, y_test, best_params_rf)
    training_randomforest_on_n_estimators(X_train, y_train, X_test, y_test, best_params_rf)

    # Logistic Regression
    print("Logistic Regression")
    best_params_lr = search_best_hyperparameters(X_train, y_train, 'LogisticRegression')
    training_LogisticRegression(X_train, y_train, X_test, y_test, best_params_lr)
  
    # Decision Tree
    print("Decision Tree")
    best_params_dt = search_best_hyperparameters(X_train, y_train, 'DecisionTree')
    training_DecisionTree(X_train, y_train, X_test, y_test, best_params_dt)
    
    # Gradient Boosting
    print("Gradient Boosting")
    best_params_gb = search_best_hyperparameters(X_train, y_train, 'GradientBoosting')
    training_GradientBoosting_on_maxdepth(X_train, y_train, X_test, y_test, best_params_gb)
    training_GradientBoosting_on_n_estimators(X_train, y_train, X_test, y_test, best_params_gb)

    # Naive Bayes
    #Naive_Bayes()

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
