from flask import Flask, request, jsonify, send_file, abort, Response
from flask_restx import Api
from api_models import ML_models

import os


app = Flask(__name__)
api = Api(app)

my_models = ML_models()
my_models.create_model('LogisticRegression (LR)')


@app.route("/api/model_types", methods=['GET'])
def model_types():
    """
    Выводит доступные типы моделей для обучения
    """
    return my_models.get_available_model_types()


@app.route("/api/create_model", methods=['POST'])
def create_model():
    """
    Создает модель. В качестве параметров надо передать:
    model_type,
    model_name (опционально),
    dataset_name (опционально)
    """
    try:
        request.json['model_type']
    except KeyError:
        abort(Response('''model_type should be passed  
{}'''.format(my_models.get_available_model_types())))
    my_models.create_model(**request.json)
    return 'Success'


@app.route("/api/get_model", methods=['GET'])
def get_all_models():
    """
    Выводит все модели
    """
    return jsonify(my_models.models)


@app.route("/api/get_model/<int:model_id>", methods=['GET'])
def get_model(model_id):
    """
    Выводит модель с указанным id
    """
    return my_models.get_model(model_id)


@app.route("/api/update_model", methods=['PUT'])
def update_model():
    """
    Обновление сведений о модели. Для обновления доступны:
    'model_name' - наименование модели,
    'dataset_name' - наименование датасета
    """
    errors = my_models.update_model(request.json)
    if not errors:
        return 'Success'
    else:
        return f'Success, but {errors}'


@app.route("/api/delete_model/<int:model_id>", methods=['DELETE'])
def delete_model(model_id):
    """
    Удаление модели с указанным id
    """
    my_models.delete(model_id)
    return 'Success'


@app.route("/api/set_model_params/<int:model_id>", methods=['PUT'])
def set_model_params(model_id):
    """
    Установка указанных параметров для указанной модели
    """
    my_models.set_model_params(model_id, request.json)
    return 'Success'


@app.route("/api/available_opt_params", methods=['GET'])
def get_available_opt_params():
    """
    Возвращает доступные для оптимазации гиперпараметры.
    Необходимо передать тип модели.
    """
    try:
        request.json['model_type']
    except TypeError:
        abort(Response('''Dictionary should be passed  
{}'''.format(my_models.get_available_model_types())))
    return jsonify(my_models.get_available_opt_params(**request.json))


@app.route("/api/optimize_model_params/<int:model_id>", methods=['GET', 'PUT'])
def optimize_model_params(model_id):
    """
    Для модели подбираются наилучшие гиперпараметры в соответствии с доступным перебором.
    """
    params, score = my_models.optimize_model_params(model_id, **request.json)
    dic = {'params': params,
           'score': score}
    return dic


@app.route("/api/fit/<int:model_id>", methods=['PUT'])
def fit(model_id):
    """
    Обучение модели
    """
    my_models.fit(model_id, **request.json)
    return 'Success'


@app.route("/api/predict/<int:model_id>", methods=['GET', 'PUT'])
def predict(model_id):
    """
    Предсказания модели
    """
    preds = my_models.predict(model_id, **request.json)
    return preds


@app.route("/api/predict_proba/<int:model_id>", methods=['GET', 'PUT'])
def predict_proba(model_id):
    """
    Предсказанные вероятности по классам
    """
    preds_proba = my_models.predict_proba(model_id, **request.json)
    return preds_proba


@app.route("/api/get_scores/<int:model_id>", methods=['GET', 'PUT'])
def get_scores(model_id):
    """
    Возвращаются посчитанные метрики качества
    """
    scores = my_models.get_scores(model_id, **request.json)
    return scores


@app.route("/api/get_auc_plot/<int:model_id>", methods=['GET', 'PUT'])
def get_auc_plot(model_id):
    """
    Строит для модели графики AUC_ROC и AUC_PR, если задача -
    бинарная классификация
    """
    if os.path.isfile(f'AUC_model_{model_id}.png'):
        return send_file(f'AUC_model_{model_id}.png', mimetype='image/gif')
    else:
        abort(Response('get_scores should be called first (only for binary task)'))


if __name__ == '__main__':
    app.run()
