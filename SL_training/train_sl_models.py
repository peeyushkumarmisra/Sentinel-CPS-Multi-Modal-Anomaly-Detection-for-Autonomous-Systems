# train_sl_models.py

import time
from SL_training.sl_models import DataProcessor, RFModel, SVMModel
from SL_training.sl_utils import evaluate_plot

class SLTraining:
    MODELS = {
        "Random Forest": RFModel,
        "SVM":           SVMModel
    }

    def __init__(self, data_path, seed=47):
        self.data_path = data_path
        self.seed      = seed
        self.data_processor = DataProcessor(file_path=self.data_path, random_state=self.seed)

    def prepare_data(self):
        return self.data_processor.prepare_data()

    def run_model(self, name, x_train, x_test, y_train):
        ModelClass = self.MODELS[name]
        model = ModelClass(random_state=self.seed)
        
        print(f"\nTraining {name}........")
        start_time = time.time()
        model.train(x_train, y_train) 
        end_time = time.time()
        train_time = end_time - start_time
        print(f"{name} trained successfully in {train_time:.5f} seconds.")
        
        y_pred = model.evaluate(x_test)
        model.save_model()
        return y_pred, train_time

    def train_sl_models(self):
        x_train, x_test, y_train, y_test = self.prepare_data()
        rf_pred, rf_time = self.run_model("Random Forest", x_train, x_test, y_train)
        svm_pred, svm_time = self.run_model("SVM", x_train, x_test, y_train)
        
        evaluate_plot(y_test, rf_pred, svm_pred, rf_time, svm_time)
        return {"Random Forest": rf_time, "SVM": svm_time}

if __name__ == "__main__":
    trainer = SLTraining(data_path='data/SensorStats.csv', seed=47)
    trainer.train_sl_models()