import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_auc_score, roc_curve, average_precision_score, \
    precision_recall_curve, accuracy_score, f1_score, recall_score, precision_score

import optuna
from optuna.samplers import TPESampler


def plot_AUC(y_true, y_predicted, plot=True):
    """
    Функция для подсчета метрик AUC-ROC и AUC-PR
    и отрисовки соответствующих графиков
    """
    if plot:
        positive = y_true.sum()

        plt.figure(figsize=(16, 8))

        plt.subplot(121)
        precision, recall, thresholds = precision_recall_curve(y_true, y_predicted)
        AUC_PR = auc(recall, precision)
        plt.plot(recall, precision, lw=1.5, label='PR curve')
        plt.axhline(positive / y_true.shape[0], 0, 1, linestyle='--', color='r')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR curve, score = {np.round(AUC_PR, 5)}')
        plt.legend(['Model', 'Random Classifier'])
        plt.grid()

        plt.subplot(122)
        fpr, tpr, thresholds = roc_curve(y_true, y_predicted)
        AUC_ROC = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, label='ROC curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='r')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve, score = {np.round(AUC_ROC, 5)}')
        plt.legend(['Model', 'Random Classifier'])

        plt.grid()
        plt.tight_layout()
        plt.savefig('AUC.png')

    else:
        precision, recall, thresholds = precision_recall_curve(y_true, y_predicted)
        AUC_PR = auc(recall, precision)
        fpr, tpr, thresholds = roc_curve(y_true, y_predicted)
        AUC_ROC = auc(fpr, tpr)

    return AUC_PR, AUC_ROC


def get_metrics_scores(y_true, y_predicted):
    """
    Функция рассчитывает метрики для классификации.
    Считаются только метрики для мультиклассовой и бинарной
    классификации.
    """
    metrics = {'precision': precision_score(y_true, y_predicted, average='macro'),
               'recall': recall_score(y_true, y_predicted, average='macro'),
               'f1': f1_score(y_true, y_predicted, average='macro'),
               'accuracy': accuracy_score(y_true, y_predicted)}
    return metrics


def find_optuna_params(X, y, model_type, params, metric='auc_roc', max_trails=100, max_time=120):
    """
    Функция на кроссвалидации подбирает наилучшие гиперпараметры
    через optuna для одной из моделей
    """

    if metric == 'auc_pr':
        metric_function = average_precision_score
    elif metric == 'f1':
        metric_function = f1_score
    elif metric == 'accuracy':
        metric_function = accuracy_score
    else:
        metric_function = roc_auc_score

    kf = StratifiedKFold(shuffle=True, random_state=42)

    if model_type == 'LogisticRegression (LR)':
        def objective(trial):
            model_params = {
                'C': trial.suggest_uniform('C', params['C'][0], params['C'][1]),
                'tol': trial.suggest_uniform('tol', params['tol'][0], params['tol'][1]),
                'max_iter': trial.suggest_int('max_iter', params['max_iter'][0], params['max_iter'][1])
            }

            scores = []

            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LogisticRegression(**model_params)
                model.fit(X_train, y_train)
                if metric == 'f1' or metric == 'accuracy':
                    preds = model.predict(X_test)
                    if metric == 'f1':
                        score = metric_function(y_test, preds, average='macro')
                    else:
                        score = metric_function(y_test, preds)
                else:
                    preds = model.predict_proba(X_test)[:, 1]
                    score = metric_function(y_test, preds)

                scores.append(score)

            return np.mean(scores)

    else:
        def objective(trial):
            model_params = {
                'boosting_type': trial.suggest_categorical('boosting_type', params['boosting_type']),
                'n_estimators': trial.suggest_int('n_estimators', params['n_estimators'][0],
                                                  params['n_estimators'][1]),
                'learning_rate': trial.suggest_uniform('learning_rate', params['learning_rate'][0],
                                                       params['learning_rate'][1]),
                'num_leaves': trial.suggest_int('num_leaves', params['num_leaves'][0],
                                                params['num_leaves'][1]),
                'max_depth': trial.suggest_int('max_depth', params['max_depth'][0],
                                               params['max_depth'][1]),
                'min_child_weight': trial.suggest_int('min_child_weight', params['min_child_weight'][0],
                                                      params['min_child_weight'][1]),
                'reg_alpha': trial.suggest_uniform('reg_alpha', params['reg_alpha'][0],
                                                   params['reg_alpha'][1]),
                'reg_lambda': trial.suggest_uniform('reg_lambda', params['reg_lambda'][0],
                                                    params['reg_lambda'][1]),
                'n_jobs': trial.suggest_int('n_jobs', 1, 1)
            }

            scores = []

            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = LGBMClassifier(**model_params)
                model.fit(X_train, y_train)
                if metric == 'f1' or metric == 'accuracy':
                    preds = model.predict(X_test)
                    if metric == 'f1':
                        score = metric_function(y_test, preds, average='macro')
                    else:
                        score = metric_function(y_test, preds)
                else:
                    preds = model.predict_proba(X_test)[:, 1]
                    score = metric_function(y_test, preds)

                scores.append(score)

            return np.mean(scores)

    optuna.logging.set_verbosity(0)
    sampler = TPESampler(seed=10)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=max_trails, timeout=max_time, n_jobs=1, show_progress_bar=True)

    return study.best_params, study.best_value
