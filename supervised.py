import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate, learning_curve
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


def balance_dataset(X, y, three_class = True):
    if three_class:
        sampling_strategy = {0: 3000, 1:4506, 2:3000}
    else:
        sampling_strategy = {0: 3000, 1:4506}
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

def preprocess_data(balanced = True, only_categorical = False, three_class = True):
    df = pd.read_csv("Working_Dataset.csv")
    le = LabelEncoder()
    df['FACILITY_TYPE'] = le.fit_transform(df['FACILITY_TYPE'])
    df = df[df['NUM_INSP_AREA'] > 1]

    if not three_class:
        df = df[df['RESULTS'] != 2]

    if only_categorical:
        df = df[['RESULTS', 'IS_HIGH_CRIME_AREA', 'IS_LOW_HEALTH_AREA', 'IS_HIGH_BELOW_POVERTY_LEVEL', 'IS_LOW_PER_CAPITA_INCOME', 'IS_HIGH_UNEMPLOYMENT_RATE','NO_VIOLATIONS','VIOLATIONS_ON_MANAGEMENT_AND_SUPERVISION','VIOLATIONS_ON_HYGIENE_AND_FOOD_SECURITY','VIOLATIONS_ON_TEMPERATURE_AND_SPECIAL_PROCEDURES','VIOLATIONS_ON_FOOD_SAFETY_AND_QUALITY','VIOLATIONS_ON_INSTRUMENT_STORAGE_AND_MAINTENANCE','VIOLATIONS_ON_FACILITIES_AND_REGULATIONS','HAS_INSP_SERIOUS_VIOL']]
        X = df.drop(['RESULTS'], axis=1)
        y = df['RESULTS']
    else:
        X = df.drop(['INSPECTION_ID', 'DBA NAME', 'RESULTS', 'COMMUNITY_AREA'], axis=1)
        y = df['RESULTS']
    
    #splitta in train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if balanced:
        if three_class:
            X_train, y_train = balance_dataset(X_train, y_train, three_class=True)
        else:
            X_train, y_train = balance_dataset(X_train, y_train, three_class=False)

    #stampa il numero di result pass e fail
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
            'model__max_iter': [1000,5000,10000] 
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
            'model__loss': ['log_loss'],
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

def training_randomforest_on_maxdepth(X_train, y_train, X_test, y_test, best_params, balanced=True, three_class=True):

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
        if three_class:
            scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score, average='macro', zero_division=0),
            'precision': make_scorer(precision_score, average='macro', zero_division=0),
            'recall': make_scorer(recall_score, average='macro', zero_division=0)
            }
        else:
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

    if three_class:
        if balanced:
            plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Random Forest Depth Balanced Three Class',three_class=True)
            plot_learning_curve(X_train, y_train, pipe, 'Random Forest Depth Learning Curve Balanced Three Class',three_class=True)
        else:
            plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Random Forest Depth Not Balanced Three Class',three_class=True)
            plot_learning_curve(X_train, y_train, pipe, 'Random Forest Depth Learning Curve Not Balanced Three Class',three_class=True)
    else:
        if balanced:
            plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Random Forest Depth Balanced',three_class=False)
            plot_learning_curve(X_train, y_train, pipe, 'Random Forest Depth Learning Curve Balanced',three_class=False)
        else:
            plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Random Forest Depth Not Balanced',three_class=False)
            plot_learning_curve(X_train, y_train, pipe, 'Random Forest Depth Learning Curve Not Balanced',three_class=False)
    
 
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

    if three_class:
        if balanced:
            plot_confusion_matrix(y_test, y_pred, 'RandomForestMaxDepthBalancedThreeClass',three_class=True)
        else:
            plot_confusion_matrix(y_test, y_pred, 'RandomForestMaxDepthNotBalancedThreeClass',three_class=True)
    else:
        if balanced:
            plot_confusion_matrix(y_test, y_pred, 'RandomForestMaxDepthBalanced',three_class=False)
        else:
            plot_confusion_matrix(y_test, y_pred, 'RandomForestMaxDepthNotBalanced',three_class=False)

    if three_class:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    else:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

    if three_class:
        if balanced:
            with open('ThreeClassModels/RandomForestBestDepthBalancedThreeClass.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
        else:
            with open('ThreeClassModels/RandomForestBestDepthNotBalancedThreeClass.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
    else:
        if balanced:
            with open('BinaryModels/RandomForestBestDepthBalanced.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
        else:
            with open('BinaryModels/RandomForestBestDepthNotBalanced.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")

def training_randomforest_on_n_estimators(X_train, y_train, X_test, y_test, best_params, balanced=True, three_class=True):
    
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
            if three_class:
                scoring = {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score, average='macro', zero_division=0),
                'precision': make_scorer(precision_score, average='macro', zero_division=0),
                'recall': make_scorer(recall_score, average='macro', zero_division=0)
                }
            else:
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

        if three_class:
            if balanced:
                plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Random Forest Estimators Balanced Three Class',three_class=True)
                plot_learning_curve(X_train, y_train, pipe, 'Random Forest Estimators Learning Curve Balanced Three Class',three_class=True)
            else:
                plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Random Forest Estimators Not Balanced Three Class',three_class=True)
                plot_learning_curve(X_train, y_train, pipe, 'Random Forest Estimators Learning Curve Not Balanced Three Class',three_class=True)
        else:
            if balanced:
                plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Random Forest Estimators Balanced',three_class=False)
                plot_learning_curve(X_train, y_train, pipe, 'Random Forest Estimators Learning Curve Balanced',three_class=False)
            else:
                plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Random Forest Estimators Not Balanced',three_class=False)
                plot_learning_curve(X_train, y_train, pipe, 'Random Forest Estimators Learning Curve Not Balanced',three_class=False)

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

        if three_class:
            if balanced:
                plot_confusion_matrix(y_test, y_pred, 'RandomForestEstimatorsBalancedThreeClass',three_class=True)
            else:
                plot_confusion_matrix(y_test, y_pred, 'RandomForestEstimatorsNotBalancedThreeClass',three_class=True)
        else:
            if balanced:
                plot_confusion_matrix(y_test, y_pred, 'RandomForestEstimatorsBalanced',three_class=False)
            else:
                plot_confusion_matrix(y_test, y_pred, 'RandomForestEstimatorsNotBalanced',three_class=False)

        if three_class:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        else:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
        
        if three_class:
            if balanced:
                with open('ThreeClassModels/RandomForestBestEstimatorsBalancedThreeClass.txt', 'w') as f:
                    f.write(f"Accuracy: {accuracy}\n")
                    f.write(f"F1 Score: {f1}\n")
                    f.write(f"Precision: {precision}\n")
                    f.write(f"Recall: {recall}\n")
            else:
                with open('ThreeClassModels/RandomForestBestEstimatorsNotBalancedThreeClass.txt', 'w') as f:
                    f.write(f"Accuracy: {accuracy}\n")
                    f.write(f"F1 Score: {f1}\n")
                    f.write(f"Precision: {precision}\n")
                    f.write(f"Recall: {recall}\n")
        else:
            if balanced:
                with open('BinaryModels/RandomForestBestEstimatorsBalanced.txt', 'w') as f:
                    f.write(f"Accuracy: {accuracy}\n")
                    f.write(f"F1 Score: {f1}\n")
                    f.write(f"Precision: {precision}\n")
                    f.write(f"Recall: {recall}\n")
            else:
                with open('BinaryModels/RandomForestBestEstimatorsNotBalanced.txt', 'w') as f:
                    f.write(f"Accuracy: {accuracy}\n")
                    f.write(f"F1 Score: {f1}\n")
                    f.write(f"Precision: {precision}\n")
                    f.write(f"Recall: {recall}\n")

def training_DecisionTree(X_train, y_train, X_test, y_test, best_params, balanced=True, three_class=True):
    
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
            if three_class:
                scoring = {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score, average='macro', zero_division=0),
                'precision': make_scorer(precision_score, average='macro', zero_division=0),
                'recall': make_scorer(recall_score, average='macro', zero_division=0)
                }
            else:
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

        if three_class:
            if balanced:
                plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Decision Tree Balanced Three Class',three_class=True)
                plot_learning_curve(X_train, y_train, pipe, 'Decision Tree Learning Curve Balanced Three Class',three_class=True)
            else:
                plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Decision Tree Not Balanced Three Class',three_class=True)
                plot_learning_curve(X_train, y_train, pipe, 'Decision Tree Learning Curve Not Balanced Three Class',three_class=True)
        else:
            if balanced:
                plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Decision Tree Balanced',three_class=False)
                plot_learning_curve(X_train, y_train, pipe, 'Decision Tree Learning Curve Balanced',three_class=False)
            else:
                plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Decision Tree Not Balanced',three_class=False)
                plot_learning_curve(X_train, y_train, pipe, 'Decision Tree Learning Curve Not Balanced',three_class=False)

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

        if three_class:
            if balanced:
                plot_confusion_matrix(y_test, y_pred, 'DecisionTreeBalancedThreeClass',three_class=True)
            else:
                plot_confusion_matrix(y_test, y_pred, 'DecisionTreeNotBalancedThreeClass',three_class=True)
        else:
            if balanced:
                plot_confusion_matrix(y_test, y_pred, 'DecisionTreeBalanced',three_class=False)
            else:
                plot_confusion_matrix(y_test, y_pred, 'DecisionTreeNotBalanced',three_class=False)


        if three_class:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        else:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

        if three_class:
            if balanced:
                with open('ThreeClassModels/DecisionTreeBestDepthBalancedThreeClass.txt', 'w') as f:
                    f.write(f"Accuracy: {accuracy}\n")
                    f.write(f"F1 Score: {f1}\n")
                    f.write(f"Precision: {precision}\n")
                    f.write(f"Recall: {recall}\n")
            else:
                with open('ThreeClassModels/DecisionTreeBestDepthNotBalancedThreeClass.txt', 'w') as f:
                    f.write(f"Accuracy: {accuracy}\n")
                    f.write(f"F1 Score: {f1}\n")
                    f.write(f"Precision: {precision}\n")
                    f.write(f"Recall: {recall}\n")
        else:
            if balanced:
                with open('BinaryModels/DecisionTreeBestDepthBalanced.txt', 'w') as f:
                    f.write(f"Accuracy: {accuracy}\n")
                    f.write(f"F1 Score: {f1}\n")
                    f.write(f"Precision: {precision}\n")
                    f.write(f"Recall: {recall}\n")
            else:
                with open('BinaryModels/DecisionTreeBestDepthNotBalanced.txt', 'w') as f:
                    f.write(f"Accuracy: {accuracy}\n")
                    f.write(f"F1 Score: {f1}\n")
                    f.write(f"Precision: {precision}\n")
                    f.write(f"Recall: {recall}\n")
        

def training_LogisticRegression(X_train, y_train, X_test, y_test, best_params, balanced=True, three_class=True):

    if three_class:
        model = LogisticRegression(
        penalty=best_params['model__penalty'],
        C=best_params['model__C'],
        solver=best_params['model__solver'],
        max_iter=best_params['model__max_iter'],
        multi_class='multinomial'  # Specificare la regressione logistica multinomiale
    )
    else:
        model = LogisticRegression(
        penalty=best_params['model__penalty'],
        C=best_params['model__C'],
        solver=best_params['model__solver'],
        max_iter=best_params['model__max_iter']
    )

    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    if three_class:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    else:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

    # Matrice di confusione
    if three_class:
        if balanced:
            plot_confusion_matrix(y_test, y_pred, 'LogisticRegressionBalancedThreeClass',three_class=True)
        else:
            plot_confusion_matrix(y_test, y_pred, 'LogisticRegressionNotBalancedThreeClass',three_class=True)
    else:
        if balanced:
            plot_confusion_matrix(y_test, y_pred, 'LogisticRegressionBalanced',three_class=False)
        else:
            plot_confusion_matrix(y_test, y_pred, 'LogisticRegressionNotBalanced',three_class=False)

    if three_class:
        if balanced:
            with open('ThreeClassModels/LogisticRegressionBalancedThreeClass.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
        else:
            with open('ThreeClassModels/LogisticRegressionNotBalancedThreeClass.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
    else:
        if balanced:
            with open('BinaryModels/LogisticRegressionBalanced.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
        else:
            with open('BinaryModels/LogisticRegressionNotBalanced.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")

def training_GradientBoosting_on_maxdepth(X_train, y_train, X_test, y_test, best_params, balanced=True, three_class=True):
        
            max_depth_values = [i for i in range(1,10)]
        
            accuracy_scores = []
            f1_scores = []
            precision_scores = []
            recall_scores = []
        
            for max_depth in max_depth_values:
                print(max_depth)
                model = GradientBoostingClassifier(
                    loss=best_params['model__loss'],
                    learning_rate=best_params['model__learning_rate'],
                    max_depth=max_depth,
                    min_samples_split=best_params['model__min_samples_split'],
                    min_samples_leaf=best_params['model__min_samples_leaf']
                )
                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
                kf = KFold(n_splits=10, shuffle=True, random_state=42)
                if three_class:
                    scoring = {
                    'accuracy': make_scorer(accuracy_score),
                    'f1': make_scorer(f1_score, average='macro', zero_division=0),
                    'precision': make_scorer(precision_score, average='macro', zero_division=0),
                    'recall': make_scorer(recall_score, average='macro', zero_division=0)
                    }
                else:
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

            if three_class:
                if balanced:
                    plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Gradient Boosting Depth Balanced Three Class',three_class=True)
                    plot_learning_curve(X_train, y_train, pipe, 'Gradient Boosting Depth Learning Curve Balanced Three Class',three_class=True)
                else:
                    plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Gradient Boosting Depth Not Balanced Three Class',three_class=True)
                    plot_learning_curve(X_train, y_train, pipe, 'Gradient Boosting Depth Learning Curve Not Balanced Three Class',three_class=True)
            else:
                if balanced:
                    plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Gradient Boosting Depth Balanced',three_class=False)
                    plot_learning_curve(X_train, y_train, pipe, 'Gradient Boosting Depth Learning Curve Balanced',three_class=False)
                else:
                    plot_scores(max_depth_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'Max Depth', 'Gradient Boosting Depth Not Balanced',three_class=False)
                    plot_learning_curve(X_train, y_train, pipe, 'Gradient Boosting Depth Learning Curve Not Balanced',three_class=False)

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

            if three_class:
                if balanced:
                    plot_confusion_matrix(y_test, y_pred, 'GradientBoostingMaxDepthBalancedThreeClass',three_class=True)
                else:
                    plot_confusion_matrix(y_test, y_pred, 'GradientBoostingMaxDepthNotBalancedThreeClass',three_class=True)
            else:
                if balanced:
                    plot_confusion_matrix(y_test, y_pred, 'GradientBoostingMaxDepthBalanced',three_class=False)
                else:
                    plot_confusion_matrix(y_test, y_pred, 'GradientBoostingMaxDepthNotBalanced',three_class=False)

            if three_class:
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            else:
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

            #salvataggio su file
            if three_class:
                if balanced:
                    with open('ThreeClassModels/GradientBoostingBestDepthBalancedThreeClass.txt', 'w') as f:
                        f.write(f"Accuracy: {accuracy}\n")
                        f.write(f"F1 Score: {f1}\n")
                        f.write(f"Precision: {precision}\n")
                        f.write(f"Recall: {recall}\n")
                else:
                    with open('ThreeClassModels/GradientBoostingBestDepthNotBalancedThreeClass.txt', 'w') as f:
                        f.write(f"Accuracy: {accuracy}\n")
                        f.write(f"F1 Score: {f1}\n")
                        f.write(f"Precision: {precision}\n")
                        f.write(f"Recall: {recall}\n")
            else:
                if balanced:
                    with open('BinaryModels/GradientBoostingBestDepthBalanced.txt', 'w') as f:
                        f.write(f"Accuracy: {accuracy}\n")
                        f.write(f"F1 Score: {f1}\n")
                        f.write(f"Precision: {precision}\n")
                        f.write(f"Recall: {recall}\n")
                else:
                    with open('BinaryModels/GradientBoostingBestDepthNotBalanced.txt', 'w') as f:
                        f.write(f"Accuracy: {accuracy}\n")
                        f.write(f"F1 Score: {f1}\n")
                        f.write(f"Precision: {precision}\n")
                        f.write(f"Recall: {recall}\n")
            
def training_GradientBoosting_on_n_estimators(X_train, y_train, X_test, y_test, best_params, balanced=True, three_class=True):
        
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
                if three_class:
                    scoring = {
                    'accuracy': make_scorer(accuracy_score),
                    'f1': make_scorer(f1_score, average='macro', zero_division=0),
                    'precision': make_scorer(precision_score, average='macro', zero_division=0),
                    'recall': make_scorer(recall_score, average='macro', zero_division=0)
                    }
                else:
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

            if three_class:
                if balanced:
                    plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Gradient Boosting Estimators Balanced Three Class',three_class=True)
                    plot_learning_curve(X_train, y_train, pipe, 'Gradient Boosting Estimators Learning Curve Balanced Three Class',three_class=True)
                else:
                    plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Gradient Boosting Estimators Not Balanced Three Class',three_class=True)
                    plot_learning_curve(X_train, y_train, pipe, 'Gradient Boosting Estimators Learning Curve Not Balanced Three Class',three_class=True)
            else:
                if balanced:
                    plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Gradient Boosting Estimators Balanced',three_class=False)
                    plot_learning_curve(X_train, y_train, pipe, 'Gradient Boosting Estimators Learning Curve Balanced',three_class=False)
                else:
                    plot_scores(n_estimators_values, accuracy_scores, f1_scores, precision_scores, recall_scores, 'N Estimators', 'Gradient Boosting Estimators Not Balanced',three_class=False)
                    plot_learning_curve(X_train, y_train, pipe, 'Gradient Boosting Estimators Learning Curve Not Balanced',three_class=False)
        
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

            if three_class:
                if balanced:
                    plot_confusion_matrix(y_test, y_pred, 'GradientBoostingEstimatorsBalancedThreeClass')
                else:
                    plot_confusion_matrix(y_test, y_pred, 'GradientBoostingEstimatorsNotBalancedThreeClass')
            else:
                if balanced:
                    plot_confusion_matrix(y_test, y_pred, 'GradientBoostingEstimatorsBalanced')
                else:
                    plot_confusion_matrix(y_test, y_pred, 'GradientBoostingEstimatorsNotBalanced')

            if three_class:
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            else:
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

            if three_class:
                if balanced:
                    with open('ThreeClassModels/GradientBoostingBestEstimatorsBalancedThreeClass.txt', 'w') as f:
                        f.write(f"Accuracy: {accuracy}\n")
                        f.write(f"F1 Score: {f1}\n")
                        f.write(f"Precision: {precision}\n")
                        f.write(f"Recall: {recall}\n")
                else:
                    with open('ThreeClassModels/GradientBoostingBestEstimatorsNotBalancedThreeClass.txt', 'w') as f:
                        f.write(f"Accuracy: {accuracy}\n")
                        f.write(f"F1 Score: {f1}\n")
                        f.write(f"Precision: {precision}\n")
                        f.write(f"Recall: {recall}\n")
            else:
                if balanced:
                    with open('BinaryModels/GradientBoostingBestEstimatorsBalanced.txt', 'w') as f:
                        f.write(f"Accuracy: {accuracy}\n")
                        f.write(f"F1 Score: {f1}\n")
                        f.write(f"Precision: {precision}\n")
                        f.write(f"Recall: {recall}\n")
                else:
                    with open('BinaryModels/GradientBoostingBestEstimatorsNotBalanced.txt', 'w') as f:
                        f.write(f"Accuracy: {accuracy}\n")
                        f.write(f"F1 Score: {f1}\n")
                        f.write(f"Precision: {precision}\n")
                        f.write(f"Recall: {recall}\n")

def plot_scores(x_values, accuracy_scores, f1_scores, precision_scores, recall_scores, x_label, title, three_class=True):
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
    if three_class:
        plt.savefig(f'ThreeClassModels/{title}.png')
    else:
        plt.savefig(f'BinaryModels/{title}.png')

def plot_learning_curve(X_train, y_train, model, title, three_class=True):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.title(title)
    if three_class:
        plt.savefig(f'ThreeClassModels/{title}.png')
    else:
        plt.savefig(f'BinaryModels/{title}.png')

def plot_confusion_matrix(y_test, y_pred, model_name, three_class=True):
    #matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    if three_class:
        class_names = ['Failed', 'Passed', 'Passed with Conditions']
    else:
        class_names = ['Failed', 'Passed']
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix {model_name}')
    #salva la matrice di confusione in un file
    if three_class:
        plt.savefig(f'ThreeClassModels/ConfusionMatrix{model_name}.png')
    else:
        plt.savefig(f'BinaryModels/ConfusionMatrix{model_name}.png')

def Naive_Bayes(X_train, y_train, X_test, y_test, balanced=True, three_class=True):
    model = CategoricalNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if three_class:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    else:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

    if three_class:
        if balanced:
            with open('ThreeClass/NaiveBayesBalancedThreeClass.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
        else:
            with open('ThreeClass/NaiveBayesNotBalancedThreeClass.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
    else:
        if balanced:
            with open('BinaryClass/NaiveBayesBalanced.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
        else:
            with open('BinaryClass/NaiveBayesNotBalanced.txt', 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")

def main():
    
    '''X_train, y_train, X_test, y_test = preprocess_data(balanced=False, three_class=False)

    #Random Forest 
    print("Random Forest non bilanciato binario")
    best_params_rf = search_best_hyperparameters(X_train, y_train, 'RandomForest')
    training_randomforest_on_maxdepth(X_train, y_train, X_test, y_test, best_params_rf, balanced=False, three_class=False)
    training_randomforest_on_n_estimators(X_train, y_train, X_test, y_test, best_params_rf, balanced=False, three_class=False)

    # Logistic Regression
    print("Logistic Regression non bilanciato binario")
    best_params_lr = search_best_hyperparameters(X_train, y_train, 'LogisticRegression')
    training_LogisticRegression(X_train, y_train, X_test, y_test, best_params_lr, balanced=False, three_class=False)
  
    # Decision Tree
    print("Decision Tree non bilanciato binario")
    best_params_dt = search_best_hyperparameters(X_train, y_train, 'DecisionTree')
    training_DecisionTree(X_train, y_train, X_test, y_test, best_params_dt, balanced=False, three_class=False)
    
    # Gradient Boosting
    print("Gradient Boosting non bilanciato binario")
    best_params_gb = search_best_hyperparameters(X_train, y_train, 'GradientBoosting')
    training_GradientBoosting_on_maxdepth(X_train, y_train, X_test, y_test, best_params_gb, balanced=False, three_class=False)
    training_GradientBoosting_on_n_estimators(X_train, y_train, X_test, y_test, best_params_gb, balanced=False, three_class=False)
    
    X_train, y_train, X_test, y_test = preprocess_data(balanced=False, three_class=True)

    #Random Forest
    print("Random Forest non bilanciato tre classi")
    best_params_rf = search_best_hyperparameters(X_train, y_train, 'RandomForest')
    training_randomforest_on_maxdepth(X_train, y_train, X_test, y_test, best_params_rf, balanced=False, three_class=True)
    training_randomforest_on_n_estimators(X_train, y_train, X_test, y_test, best_params_rf, balanced=False, three_class=True)

    # Logistic Regression
    print("Logistic Regression non bilanciato tre classi")
    best_params_lr = search_best_hyperparameters(X_train, y_train, 'LogisticRegression')
    training_LogisticRegression(X_train, y_train, X_test, y_test, best_params_lr, balanced=False, three_class=True)

    # Decision Tree
    print("Decision Tree non bilanciato tre classi")
    best_params_dt = search_best_hyperparameters(X_train, y_train, 'DecisionTree')
    training_DecisionTree(X_train, y_train, X_test, y_test, best_params_dt, balanced=False, three_class=True)

    # Gradient Boosting
    print("Gradient Boosting non bilanciato tre classi")
    best_params_gb = search_best_hyperparameters(X_train, y_train, 'GradientBoosting')
    training_GradientBoosting_on_maxdepth(X_train, y_train, X_test, y_test, best_params_gb, balanced=False, three_class=True)
    training_GradientBoosting_on_n_estimators(X_train, y_train, X_test, y_test, best_params_gb, balanced=False, three_class=True)'''



    X_train, y_train, X_test, y_test = preprocess_data(balanced=True, three_class=False)

    #Random Forest
    print("Random Forest bilanciato binario")
    best_params_rf = search_best_hyperparameters(X_train, y_train, 'RandomForest')
    training_randomforest_on_maxdepth(X_train, y_train, X_test, y_test, best_params_rf, balanced=True, three_class=False)
    training_randomforest_on_n_estimators(X_train, y_train, X_test, y_test, best_params_rf, balanced=True, three_class=False)

    # Logistic Regression
    print("Logistic Regression bilanciato binario")
    best_params_lr = search_best_hyperparameters(X_train, y_train, 'LogisticRegression')
    training_LogisticRegression(X_train, y_train, X_test, y_test, best_params_lr, balanced=True, three_class=False)
  
    # Decision Tree
    print("Decision Tree bilanciato binario")
    best_params_dt = search_best_hyperparameters(X_train, y_train, 'DecisionTree')
    training_DecisionTree(X_train, y_train, X_test, y_test, best_params_dt, balanced=True, three_class=False)
    
    # Gradient Boosting
    print("Gradient Boosting bilanciato binario")
    best_params_gb = search_best_hyperparameters(X_train, y_train, 'GradientBoosting')
    training_GradientBoosting_on_maxdepth(X_train, y_train, X_test, y_test, best_params_gb, balanced=True, three_class=False)
    training_GradientBoosting_on_n_estimators(X_train, y_train, X_test, y_test, best_params_gb, balanced=True, three_class=False)


    X_train, y_train, X_test, y_test = preprocess_data(balanced=True, three_class=True)

    #Random Forest
    print("Random Forest bilanciato tre classi")
    best_params_rf = search_best_hyperparameters(X_train, y_train, 'RandomForest')
    training_randomforest_on_maxdepth(X_train, y_train, X_test, y_test, best_params_rf, balanced=True, three_class=True)
    training_randomforest_on_n_estimators(X_train, y_train, X_test, y_test, best_params_rf, balanced=True, three_class=True)

    # Logistic Regression
    print("Logistic Regression bilanciato tre classi")
    best_params_lr = search_best_hyperparameters(X_train, y_train, 'LogisticRegression')
    training_LogisticRegression(X_train, y_train, X_test, y_test, best_params_lr, balanced=True, three_class=True)

    # Decision Tree
    print("Decision Tree bilanciato tre classi")
    best_params_dt = search_best_hyperparameters(X_train, y_train, 'DecisionTree')
    training_DecisionTree(X_train, y_train, X_test, y_test, best_params_dt, balanced=True, three_class=True)

    # Gradient Boosting
    print("Gradient Boosting bilanciato tre classi")
    best_params_gb = search_best_hyperparameters(X_train, y_train, 'GradientBoosting')
    training_GradientBoosting_on_maxdepth(X_train, y_train, X_test, y_test, best_params_gb, balanced=True, three_class=True)
    training_GradientBoosting_on_n_estimators(X_train, y_train, X_test, y_test, best_params_gb, balanced=True, three_class=True)

    # Naive Bayes
    print("Naive Bayes")
    X_train, y_train, X_test, y_test = preprocess_data(balanced=True, only_categorical=True, three_class=False)
    Naive_Bayes(X_train, y_train, X_test, y_test, balanced=True, three_class=False)

    X_train, y_train, X_test, y_test = preprocess_data(balanced=True, only_categorical=True, three_class=True)
    Naive_Bayes(X_train, y_train, X_test, y_test, balanced=True, three_class=True)

    X_train, y_train, X_test, y_test = preprocess_data(balanced=False, only_categorical=True, three_class=False)
    Naive_Bayes(X_train, y_train, X_test, y_test, balanced=False, three_class=False)

    X_train, y_train, X_test, y_test = preprocess_data(balanced=False, only_categorical=True, three_class=True)
    Naive_Bayes(X_train, y_train, X_test, y_test, balanced=False, three_class=True)


main()



