# sl_models.py

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, file_path, seed):
        self.file_path = file_path
        self.seed = seed
        self.scaler = StandardScaler()
        self.features = [
            'voc_emissions', 'acoustic_vibration', 'cpu_utilization', 'mechanical_load',
            'elevation_altitude', 'actuator_torque', 'thermal_load', 'compromise_risk_index'
        ]

    # Creating a new Categorization (Further Classifing Based on Binary Columns)
    def categorization(self, df):
        conditions = [
            (df["asset_class"] == "robotic_arm") & (df["remote_override_flag"] == 1),
            (df["asset_class"] == "robotic_arm") & (df["remote_override_flag"] != 1),
            (df["asset_class"] == "agv_unit") & (df["airborne_status"] == 1),
            (df["asset_class"] == "agv_unit") & (df["airborne_status"] != 1),
            (df["asset_class"] == "cnc_machine"),
            (df["asset_class"] == "plc_controller"),
            (df["asset_class"] == "drone")
        ]
        choices = [
            "robotic_arm_override",
            "robotic_arm_non_override",
            "agv_unit_fly",
            "agv_unit_non_fly",
            "cnc_machine",
            "plc_controller",
            "drone_fly"
        ]
        df["new_class"] = np.select(conditions, choices, default="Unknown")
        return df

    def prepare_data(self):
        df = pd.read_csv(self.file_path)
        df = self.categorization(df)
        df = df[df["new_class"] != "Unknown"] 
        x = df[self.features]
        y = df["new_class"]
        x = x.fillna(0)                             # Replacing all blank with Zero (For Getting Errors)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.seed, stratify=y)
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        joblib.dump(self.scaler, "MODELS/sl_scaler.pkl")
        return x_train_scaled, x_test_scaled, y_train, y_test



class SLModel:
    def __init__(self, model_path, seed):
        self.save_path = model_path
        self.seed = seed
        self.model = self.initialize_model()
        self.training_time = 0.0

    def initialize_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, return_conf = True):
        predictions = self.model.predict(x_test)
        if return_conf:
            probab = self.model.predict_proba(x_test)
            conf = np.round(np.max(probab, axis=1)* 100, 2)
            return predictions, conf
        return predictions

    def save_model(self):
        joblib.dump(self.model, self.save_path)

class RFModel(SLModel):
    def __init__(self, model_path, seed):
        super().__init__(model_path + "/rf_model.pkl", seed)

    def initialize_model(self):
        return RandomForestClassifier(n_estimators=10, random_state=self.seed)

class SVMModel(SLModel):
    def __init__(self, model_path, seed):
        super().__init__(model_path + "/svm_model.pkl", seed)

    def initialize_model(self):
        return SVC(kernel='rbf', gamma=0.001, C=1.0, probability=True, random_state=self.seed)