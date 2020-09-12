import numpy as np
from catboost import CatBoostClassifier, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle


class BaseExperiment:
    KERAS_MAX_ONEHOT_VALUES = 50  # if unique(feature) < MAX_ONEHOT_VALUES, one-hot encoding, otherwise integer encoding
    KERAS_EPOCHS = 16
    POSITIVE_STEPS = [9999999999]
    NEGATIVE_STEPS = [9999999999]
    ITERATIONS = 5
    PLOT_FIG_SIZE = (9, 12)

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
        catboost_metrics = self.run_catboost()
        keras_metrics = self.run_keras()
        self.plot_metrics(catboost_metrics, keras_metrics)
        self.plot_metrics_diff(catboost_metrics, keras_metrics)
        self.write_metrics(catboost_metrics, keras_metrics)

    def run_catboost(self):
        dataset = self.get_dataset()
        metrics = {
            'accuracy': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
            'roc_auc': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
            'mean_prediction': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
        }
        for i, positive in enumerate(tqdm(self.POSITIVE_STEPS, desc='Catboost')):
            for j, negative in enumerate(tqdm(self.NEGATIVE_STEPS, leave=False)):
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
                    acc += accuracy_score(y_valid, cbc_1.predict(X_valid))
                    roc_auc += roc_auc_score(y_valid, cbc_1.predict_proba(X_valid)[:, 1])
                    mean_pred += cbc_1.predict(X_valid).mean()

                metrics['accuracy'][j, i] = acc / self.ITERATIONS
                metrics['roc_auc'][j, i] = roc_auc / self.ITERATIONS
                metrics['mean_prediction'][j, i] = mean_pred / self.ITERATIONS
        return metrics

    def run_keras(self):
        dataset = self.transform_dataset_for_keras(self.get_dataset())
        metrics = {
            'accuracy': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
            'roc_auc': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
            'mean_prediction': np.zeros(shape=(len(self.NEGATIVE_STEPS), len(self.POSITIVE_STEPS))),
        }
        for i, positive in enumerate(tqdm(self.POSITIVE_STEPS, desc='Keras')):
            for j, negative in enumerate(tqdm(self.NEGATIVE_STEPS, leave=False)):
                acc = 0.0
                roc_auc = 0.0
                mean_pred = 0.0
                for iteration in range(self.ITERATIONS):
                    X, y, _ = self.get_balanced_dataset(dataset, positive, negative)
                    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
                    
                    model = self.get_keras_model()
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
                        'accuracy',
                        keras.metrics.AUC(),
                    ])
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
                    acc += test_acc
                    roc_auc += test_auc
                    mean_pred += model.predict_classes(X_valid).mean()
                    
                metrics['accuracy'][j, i] = acc / self.ITERATIONS
                metrics['roc_auc'][j, i] = roc_auc / self.ITERATIONS
                metrics['mean_prediction'][j, i] = mean_pred / self.ITERATIONS
        return metrics

    def transform_dataset_for_keras(self, dataset):
        X, y, cat_features = dataset

        columns_to_onehot_encode = [x for x in cat_features if X[x].nunique() <= self.KERAS_MAX_ONEHOT_VALUES]
        columns_to_integer_encode = [x for x in cat_features if X[x].nunique() > self.KERAS_MAX_ONEHOT_VALUES]
        for col in X.columns:
            if col in columns_to_integer_encode:
                X[col] = X[col].astype('category').cat.codes.astype('float32')
            if col not in cat_features or col in columns_to_integer_encode:
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

    def plot_metrics(self, catboost_metrics, keras_metrics):
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
        plt.show()

    def write_metrics(self, catboost_metrics, keras_metrics):
        filename = f'{self.__class__.__name__}_metrics.pickle'
        result = {
            'catboost': catboost_metrics,
            'keras': keras_metrics,
        }
        with open(filename, 'wb') as f:
            pickle.dump(result, f)

    def plot_metrics_diff(self, catboost_metrics, keras_metrics):
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
        # plt.imshow(diff, cmap=plt.get_cmap("PiYG", 7))
        plt.imshow(diff_rgb)
        plt.xlabel('Positives', fontsize=14)
        plt.ylabel('Negatives', fontsize=14)
        plt.xticks(range(len(self.POSITIVE_STEPS)), self.POSITIVE_STEPS)
        plt.yticks(range(len(self.NEGATIVE_STEPS)), self.NEGATIVE_STEPS)
        for i in range(len(self.NEGATIVE_STEPS)):
            for j in range(len(self.POSITIVE_STEPS)):
                plt.text(j-0.45, i-0.15, f'true+={round(self.POSITIVE_STEPS[j] / (self.POSITIVE_STEPS[j] + self.NEGATIVE_STEPS[i]), 2)}')
                plt.text(j-0.45, i+0.15, f"auc={'+' if diff[i, j] > 0 else ''}{round(diff[i, j]*100, 1)}%")
        plt.show()



