import itertools
import multiprocessing
import numpy as np
from catboost import CatBoostClassifier, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle
import sys
import telegram_send
import os


def test_keras_parameters(all_params):
    self, dataset, params = all_params
    loss = 0.0
    acc = 0.0
    roc_auc = 0.0
    for iteration in range(self.KERAS_HYPERPARAMETER_ITERATIONS):
        model = self.get_compiled_keras_model_from_parameters(*params)
        X, y, _ = self.get_balanced_dataset(dataset, 9999999999, 9999999999)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=0,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            )
        ]
        model.fit(
            X_train, y_train,
            epochs=self.KERAS_EPOCHS, validation_data=(X_valid, y_valid),
            callbacks=callbacks, verbose=False,
        )
        test_loss, test_acc, test_auc = model.evaluate(X_valid,  y_valid, verbose=0)
        loss += test_loss
        acc += test_acc
        roc_auc += test_auc
    metrics = (
        loss / self.KERAS_HYPERPARAMETER_ITERATIONS,
        acc / self.KERAS_HYPERPARAMETER_ITERATIONS,
        roc_auc / self.KERAS_HYPERPARAMETER_ITERATIONS,
    )
    return params, metrics


class BaseExperiment:
    KERAS_MAX_ONEHOT_VALUES = 50  # if unique(feature) < MAX_ONEHOT_VALUES, one-hot encoding, otherwise integer encoding
    KERAS_EPOCHS = 16
    POSITIVE_STEPS = [9999999999]
    NEGATIVE_STEPS = [9999999999]
    ITERATIONS = 5
    PLOT_FIG_SIZE = (9, 12)
    KERAS_HYPERPARAMETER_WORKERS = 8
    KERAS_HYPERPARAMETER_ITERATIONS = 4
    KERAS_LAYERS_RANGE = [2, 3, 4]
    KERAS_LAYER_SIZE_RANGE = [32, 64, 128, 160]
    KERAS_DROPOUT_RANGE = [0.2, 0.35, 0.5]
    KERAS_ACTIVATIONS = ['relu', 'tanh']
    KERAS_OPTIMIZERS_LIST = ['adam']  # ['adam', 'adadelta', 'rmsprop']  # in most cases adam was better
    USE_CTR = False
    LOG_TO_STDOUT = False
    LOG_TO_TELEGRAM = True

    def get_dataset(self):
        raise NotImplementedError()

    def get_balanced_dataset(self, dataset, positives, negatives):
        X, y, cat_features = dataset
        data = X.copy()
        assert 'label' not in data, '"label" can not be a name of a feature'  # the easiest way
        data['label'] = y
        assert (data[data['label'] == 0].shape[0] + data[data['label'] == 1].shape[0]) == data.shape[0], 'labels should be only 0 or 1'
        data = pd.concat([
            data[data['label'] == 1][:positives],
            data[data['label'] == 0][:negatives],
        ]).sample(frac=1).reset_index(drop=True)
        return data.drop(columns=['label']), data['label'], cat_features

    def run(self):
        keras_params = self.tune_keras_hyperparameters()
        catboost_metrics = self.run_catboost()
        keras_metrics = self.run_keras(keras_params)
        self.plot_metrics(catboost_metrics, keras_metrics)
        self.plot_metrics_diff(catboost_metrics, keras_metrics)
        self.print_summary(catboost_metrics, keras_metrics)
        self.write_metrics(catboost_metrics, keras_metrics)
        self.log_progress('Finished')

    def run_catboost(self):
        dataset = self.get_dataset()
        metrics = {
            'loss': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
            'accuracy': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
            'roc_auc': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
            'mean_prediction': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
        }
        for i, positive in enumerate(tqdm(self.POSITIVE_STEPS, desc='Catboost')):
            self.log_progress(f'{i}/{len(self.POSITIVE_STEPS)}')
            for j, negative in enumerate(tqdm(self.NEGATIVE_STEPS, leave=False)):
                loss = 0.0
                acc = 0.0
                roc_auc = 0.0
                mean_pred = 0.0
                for iteration in range(self.ITERATIONS):
                    X, y, cat_features = self.get_balanced_dataset(dataset,  positive, negative)
                    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)

                    params = {
                        'loss_function':'Logloss',
                        'eval_metric':'AUC',
                        'cat_features': cat_features,
                        'early_stopping_rounds': 200,
                        'verbose': False,
                    }
                    cbc_1 = CatBoostClassifier(**params)
                    cbc_1.fit(
                        X_train, y_train,
                        eval_set=(X_valid, y_valid),
                        use_best_model=True,
                        plot=False,
                    )
                    loss += log_loss(y_valid, cbc_1.predict_proba(X_valid)[:, 1])
                    acc += accuracy_score(y_valid, cbc_1.predict(X_valid))
                    roc_auc += roc_auc_score(y_valid, cbc_1.predict_proba(X_valid)[:, 1])
                    mean_pred += cbc_1.predict(X_valid).mean()

                metrics['loss'][j, i] = loss / self.ITERATIONS
                metrics['accuracy'][j, i] = acc / self.ITERATIONS
                metrics['roc_auc'][j, i] = roc_auc / self.ITERATIONS
                metrics['mean_prediction'][j, i] = mean_pred / self.ITERATIONS
        return metrics

    def run_keras(self, model_hyperparameters):
        dataset = self.transform_dataset_for_keras(self.get_dataset())
        metrics = {
            'loss': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
            'accuracy': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
            'roc_auc': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
            'mean_prediction': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
        }
        for i, positive in enumerate(tqdm(self.POSITIVE_STEPS, desc='Keras')):
            self.log_progress(f'{i}/{len(self.POSITIVE_STEPS)}')
            for j, negative in enumerate(tqdm(self.NEGATIVE_STEPS, leave=False)):
                loss = 0.0
                acc = 0.0
                roc_auc = 0.0
                mean_pred = 0.0
                for iteration in range(self.ITERATIONS):
                    X, y, _ = self.get_balanced_dataset(dataset, positive, negative)
                    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
                    
                    model = self.get_compiled_keras_model_from_parameters(*model_hyperparameters)
                    callbacks = [
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            min_delta=0,
                            patience=0,
                            verbose=0,
                            mode="auto",
                            baseline=None,
                            restore_best_weights=False,
                        )
                    ]
                    model.fit(
                        X_train, y_train,
                        epochs=self.KERAS_EPOCHS, validation_data=(X_valid, y_valid),
                        callbacks=callbacks, verbose=False,
                    )
                    test_loss, test_acc, test_auc = model.evaluate(X_valid,  y_valid, verbose=0)
                    loss += test_loss
                    acc += test_acc
                    roc_auc += test_auc
                    mean_pred += model.predict_classes(X_valid).mean()
                
                metrics['loss'][j, i] = loss / self.ITERATIONS
                metrics['accuracy'][j, i] = acc / self.ITERATIONS
                metrics['roc_auc'][j, i] = roc_auc / self.ITERATIONS
                metrics['mean_prediction'][j, i] = mean_pred / self.ITERATIONS
        return metrics

    def transform_dataset_for_keras(self, dataset):
        X, y, cat_features = dataset
        numeric_features = [x for x in X.columns if x not in cat_features]

        columns_to_onehot_encode = [x for x in cat_features if X[x].nunique() <= self.KERAS_MAX_ONEHOT_VALUES]
        if self.USE_CTR:
            columns_to_ctr_encode = [x for x in cat_features if X[x].nunique() > self.KERAS_MAX_ONEHOT_VALUES]
            columns_to_integer_encode = []
        else:
            columns_to_ctr_encode = []
            columns_to_integer_encode = [x for x in cat_features if X[x].nunique() > self.KERAS_MAX_ONEHOT_VALUES]

        new_ctr_features = []
        ctr_combinations = [(column,) for column in columns_to_ctr_encode] + list(itertools.combinations(columns_to_ctr_encode, 2))
        for column_set in ctr_combinations:
            values = X[column_set[0]].astype('str')
            if len(column_set) > 1:
                for column in column_set[1:]:
                    values += ';' + X[column].astype('str')
            unique_counts = values.value_counts()
            value_to_count = {value: count for (value, count) in zip(unique_counts.axes[0], unique_counts)}
            new_column_name = '__'.join(column_set) + '__CTR'
            new_ctr_features.append(new_column_name)
            X[new_column_name] = 0.0
            total_count = X.shape[0]
            PRIOR = 0.5
            for value, count in value_to_count.items():
                X.loc[values == value, (new_column_name,)] = (count + PRIOR) / (total_count + 1)
        X = X.drop(columns=columns_to_ctr_encode)

        for col in X.columns:
            if col in columns_to_integer_encode:
                X[col] = X[col].astype('category').cat.codes.astype('float32')
            if col in numeric_features or col in columns_to_integer_encode or col in new_ctr_features:
                if X[col].max() == X[col].min():
                    col_range = 1
                else:
                    col_range = X[col].max() - X[col].min()
                X[col] = (X[col] - X[col].min()) / col_range

        X = pd.get_dummies(X, columns=columns_to_onehot_encode)

        X = X.astype('float32')
        y = y.astype('float32')
        return X,  y, cat_features

    def get_keras_model(self):
        raise NotImplementedError()

    def tune_keras_hyperparameters(self):
        dataset = self.transform_dataset_for_keras(self.get_dataset())
        loss_by_params = {}
        accuracy_by_params = {}
        roc_auc_by_params = {}
        parameter_tuples = []
        for layers_num in self.KERAS_LAYERS_RANGE:
            for layer_sizes in itertools.product(self.KERAS_LAYER_SIZE_RANGE, repeat=layers_num):
                # In most cases the larger first two layers were – there better, so we cut small layers for faster tuning
                if layer_sizes[0] <= 64:
                    continue
                if layer_sizes[1] <= 32:
                    continue
                if len(layer_sizes) >= 2 and layer_sizes[-1] >= layer_sizes[-2]:
                    continue  # small optimization: we believe that the network should become more "narrow" at the end
                if len(layer_sizes) >= 3 and layer_sizes[-2] > layer_sizes[-3]:
                    continue  # small optimization: we believe that the network should become more "narrow" at the end
                for activation in self.KERAS_ACTIVATIONS:
                    for dropout in self.KERAS_DROPOUT_RANGE:
                        for optimizer in self.KERAS_OPTIMIZERS_LIST:
                            params_key = (layer_sizes, activation, dropout, optimizer)
                            parameter_tuples.append(params_key)

        with multiprocessing.Pool(processes=self.KERAS_HYPERPARAMETER_WORKERS) as pool:
            all_param_tuples = [(self, dataset, param_tuple) for param_tuple in parameter_tuples]
            # results = pool.imap_unordered(test_keras_parameters, all_param_tuples)
            results = pool.map(test_keras_parameters, all_param_tuples)
            for _i, result in enumerate(tqdm(results, total=len(parameter_tuples), desc='Tuning Keras')):
                if _i % 24 == 0:
                    self.log_progress(f'{_i}/{len(parameter_tuples)}')
                params_key, metrics = result
                loss, acc, auc = metrics
                loss_by_params[params_key] = loss
                accuracy_by_params[params_key] = acc
                roc_auc_by_params[params_key] = auc

        ordered_params = sorted(roc_auc_by_params.items(), key=lambda item: item[1], reverse=True)
        print('BEST NETWORK PARAMETERS:')
        for item in ordered_params[:5]:
            params_key, roc_auc = item
            print(f'{params_key}: auc={roc_auc:.3f}, acc={accuracy_by_params[params_key]:.3f}')
        return ordered_params[0][0]

    def get_compiled_keras_model_from_parameters(self, layer_sizes, activation, dropout, optimizer):
        layers = []
        for layer_size in layer_sizes:
            layers.append(keras.layers.Dense(layer_size, activation=activation))
            if dropout > 0:
                layers.append(keras.layers.Dropout(dropout))
        layers.append(keras.layers.Dense(1, activation='sigmoid'))
        model = keras.Sequential(layers)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC()])
        return model

    def plot_metrics(self, catboost_metrics, keras_metrics):
        if not os.path.exists('experiments'):
            os.mkdir('experiments')
        if not os.path.exists(f'experiments/plots'):
            os.mkdir(f'experiments/plots')
        cmap = np.concatenate([
            catboost_metrics['roc_auc'],
            np.full((1, len(self.POSITIVE_STEPS)), min(catboost_metrics['roc_auc'].min(), keras_metrics['roc_auc'].min())),
            keras_metrics['roc_auc']
        ])
        plt.figure(figsize=self.PLOT_FIG_SIZE)
        plt.title(f'AUC', fontsize=18)
        plt.imshow(cmap, cmap=plt.get_cmap("PiYG", 7))
        plt.xlabel('Positives', fontsize=14)
        plt.ylabel('Negatives', fontsize=14)
        plt.xticks(range(len(self.POSITIVE_STEPS)), self.POSITIVE_STEPS)
        plt.yticks(
            range(len(self.NEGATIVE_STEPS)*2+1),
            [f'{x} (Catboost)' for x in self.NEGATIVE_STEPS] + [''] + [f'{x} (Keras)' for x in self.NEGATIVE_STEPS]
        )
        for i in range(len(self.NEGATIVE_STEPS)):
            for j in range(len(self.POSITIVE_STEPS)):
                plt.text(j-0.45, i-0.15, f'true+={round(self.POSITIVE_STEPS[j] / (self.POSITIVE_STEPS[j] + self.NEGATIVE_STEPS[i]), 2)}')
                # plt.text(j-0.45, i, f"pred={round(catboost_metrics['mean_prediction'][i, j], 2)}")
                plt.text(j-0.45, i+0.15, f"auc={round(catboost_metrics['roc_auc'][i, j], 3)}")
                plt.text(j-0.45, i-0.15 + len(self.NEGATIVE_STEPS) + 1, f"true+={round(self.POSITIVE_STEPS[j] / (self.POSITIVE_STEPS[j] + self.NEGATIVE_STEPS[i]), 2)}")
                # plt.text(j-0.45, i + len(self.NEGATIVE_STEPS) + 1, f"pred={round(keras_metrics['mean_prediction'][i, j], 2)}")
                plt.text(j-0.45, i+0.15 + len(self.NEGATIVE_STEPS) + 1, f"auc={round(keras_metrics['roc_auc'][i, j], 3)}")
        plt.savefig(f'experiments/plots/{self.__class__.__name__}_stacked.png')

    def write_metrics(self, catboost_metrics, keras_metrics):
        if not os.path.exists('experiments'):
            os.mkdir('experiments')
        if not os.path.exists('experiments/metrics'):
            os.mkdir('experiments/metrics')
        filename = f'experiments/metrics/{self.__class__.__name__}.pickle'
        result = {
            'catboost': catboost_metrics,
            'keras': keras_metrics,
        }
        with open(filename, 'wb') as f:
            pickle.dump(result, f)

    def plot_metrics_diff(self, catboost_metrics, keras_metrics):
        if not os.path.exists('experiments'):
            os.mkdir('experiments')
        if not os.path.exists('experiments/plots'):
            os.mkdir('experiments/plots')
        diff = (catboost_metrics['roc_auc'] - keras_metrics['roc_auc']) / keras_metrics['roc_auc']
        min_intensity = 0.5
        clip_threshold = 0.7
        diff_rgb = np.full((diff.shape[0], diff.shape[1], 3), 1.0)
        for i in range(diff_rgb.shape[0]):
            for j in range(diff_rgb.shape[1]):
                clipped = max(min(diff[i, j], clip_threshold), -clip_threshold)
                coef = (clip_threshold - abs(clipped)) / clip_threshold
                blue = 1 * coef + min_intensity * (1 - coef)
                if clipped > 0:
                    green = 1
                    red = 1 * coef + min_intensity * (1 - coef)
                else:
                    green = 1 * coef + min_intensity * (1 - coef)
                    red = 1
                diff_rgb[i, j, 0] = red
                diff_rgb[i, j, 1] = green
                diff_rgb[i, j, 2] = blue
        plt.figure(figsize=(self.PLOT_FIG_SIZE[0], self.PLOT_FIG_SIZE[1]/2))
        plt.title('Catboost over Keras', fontsize=18)
        plt.imshow(diff_rgb)
        plt.xlabel('Positives', fontsize=14)
        plt.ylabel('Negatives', fontsize=14)
        plt.xticks(range(len(self.POSITIVE_STEPS)), self.POSITIVE_STEPS)
        plt.yticks(range(len(self.NEGATIVE_STEPS)), self.NEGATIVE_STEPS)
        for i in range(len(self.NEGATIVE_STEPS)):
            for j in range(len(self.POSITIVE_STEPS)):
                plt.text(j-0.45, i-0.15, f'true+={round(self.POSITIVE_STEPS[j] / (self.POSITIVE_STEPS[j] + self.NEGATIVE_STEPS[i]), 2)}')
                plt.text(j-0.45, i+0.15, f"auc={'+' if diff[i, j] > 0 else ''}{round(diff[i, j]*100, 1)}%")
        plt.savefig(f'experiments/plots/{self.__class__.__name__}_diff.png')

    def print_summary(self, catboost_metrics, keras_metrics):
        if not os.path.exists('experiments'):
            os.mkdir('experiments')
        if not os.path.exists('experiments/summaries'):
            os.mkdir('experiments/summaries')
        with open(f'experiments/summaries/{self.__class__.__name__}.txt', 'w') as f:
            print('---- METRICS SUMMARY ----\n')
            f.write('---- METRICS SUMMARY ----\n\n')
            for key in sorted(catboost_metrics):
                if key == 'mean_prediction':
                    continue  # not a very intereting metric
                catboost_value = catboost_metrics[key][-1, -1]
                keras_value = keras_metrics[key][-1, -1]
                catboost_diff = (catboost_value - keras_value) / max(keras_value, 1e-6) * 100
                catboost_diff_sign = '+' if catboost_diff >= 0.0 else ''
                keras_diff = (keras_value - catboost_value) / max(catboost_value, 1e-6) * 100
                keras_diff_sign = '+' if keras_diff >= 0.0 else ''
                print(f'{key:>10}: catboost={catboost_value:.3f} ({catboost_diff_sign}{catboost_diff:.2f}%), keras={keras_value:.3f} ({keras_diff_sign}{keras_diff:.2f}%)\n')
                f.write(f'{key:>10}: catboost={catboost_value:.3f} ({catboost_diff_sign}{catboost_diff:.2f}%), keras={keras_value:.3f} ({keras_diff_sign}{keras_diff:.2f}%)\n\n')
            print('-------------------------')
            f.write('-------------------------\n')

    def log_progress(self, message):
        caller_class = self.__class__.__name__
        caller_fn = sys._getframe().f_back.f_code.co_name
        full_message = f'{caller_class}.{caller_fn}: {message}'
        if self.LOG_TO_STDOUT:
            print(full_message)
        if self.LOG_TO_TELEGRAM:
            try:
                telegram_send.send(messages=[full_message])
            except Exception as exc:
                print(exc)




