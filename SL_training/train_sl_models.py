# train_sl_models.py

import os
import time
from SL_training.sl_models import DataProcessor, RFModel, SVMModel
from SL_training.sl_utils import evaluate_plot_models

class SLTraining:
    def __init__(self, seed, data_path, plot_dir, model_dir):
        self.seed       = seed
        self.data_path  = data_path
        self.plot_path  = plot_dir
        self.model_path = model_dir
        self.models = {
        "Random Forest": RFModel(self.model_path, seed),
        "SVM":           SVMModel(self.model_path, seed)
        }
        self.data_processor = DataProcessor(file_path=self.data_path, seed=self.seed)

    def prepare_data(self):
        return self.data_processor.prepare_data()

    def run_model(self, name, x_train, x_test, y_train):
        model = self.models[name]
        print(f"\nTraining {name}........")
        start_time = time.time()
        model.train(x_train, y_train) 
        end_time = time.time()
        train_time = end_time - start_time
        print(f"{name} trained successfully in {train_time:.5f} seconds.")
        y_pred, conf_score = model.evaluate(x_test, return_conf=True)
        model.save_model()
        return y_pred, conf_score, train_time

    def train_sl_models(self):
        x_train, x_test, y_train, y_test = self.prepare_data()
        rf_pred, rf_conf, rf_time = self.run_model("Random Forest", x_train, x_test, y_train)
        svm_pred, svm_conf, svm_time = self.run_model("SVM", x_train, x_test, y_train)
        plot_path = os.path.join(self.plot_path, "sl_model_comparison.jpeg")
        evaluate_plot_models(y_test = y_test, plot_dir = plot_path,
                             rf_pred = rf_pred, svm_pred = svm_pred,
                             rf_time = rf_time, svm_time = svm_time,
                             rf_conf = rf_conf, svm_conf = svm_conf
                             )
        return {"Random Forest": rf_time, "SVM": svm_time}