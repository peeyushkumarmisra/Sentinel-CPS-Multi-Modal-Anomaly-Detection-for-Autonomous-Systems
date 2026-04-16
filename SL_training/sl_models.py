# sl_models.py

import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, file_path: str, random_state: int = 47):
        self.file_path = file_path
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.features = [
            'VOC Emissions', 'Acoustic / Vibration', 'CPU Utilization', 'Mechanical Load',
            'Axis Elevation / Altitude', 'Actuator Torque', 'Thermal Load', 'Compromise Risk Index'
        ]

    def categorization(self, df):
        # Optimized vectorized conditions instead of iterrows loop
        conditions = [
            (df["Asset Class"] == "Robotic Arm") & (df["Remote Override Flag"] == 1),
            (df["Asset Class"] == "Robotic Arm") & (df["Remote Override Flag"] != 1),
            (df["Asset Class"] == "AGV Unit") & (df["Airborne Status"] == 1),
            (df["Asset Class"] == "AGV Unit") & (df["Airborne Status"] != 1),
            (df["Asset Class"] == "CNC Machine"),
            (df["Asset Class"] == "PLC Controller"),
            (df["Asset Class"] == "Drone")
        ]
        choices = [
            "Robotic Arm (Override)",
            "Robotic Arm (Non Override)",
            "AGV Unit (Fly)",
            "AGV Unit (No Fly)",
            "CNC Machine",
            "PLC Controller",
            "Drone"
        ]
        
        df["Detailed_Class"] = np.select(conditions, choices, default="Unknown")
        return df

    def prepare_data(self):
        df = pd.read_csv(self.file_path)
        df = self.categorization(df)
        df = df[df["Detailed_Class"] != "Unknown"] 
        
        x = df[self.features]
        y = df["Detailed_Class"]
        x = x.fillna(0)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.random_state, stratify=y)
        
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        
        return x_train_scaled, x_test_scaled, y_train, y_test

class SLModel:
    def __init__(self, save_path: str, random_state: int = 47):
        self.save_path = save_path
        self.random_state = random_state
        self.model = self._initialize_model()
        self.training_time = 0.0

    def _initialize_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evaluate(self, x_test):
        return self.model.predict(x_test)

    def save_model(self):
        joblib.dump(self.model, self.save_path)

class RFModel(SLModel):
    def __init__(self, random_state: int = 47):
        super().__init__("SL_training/rf_model.pkl", random_state)

    def _initialize_model(self):
        return RandomForestClassifier(n_estimators=100, random_state=self.random_state)

class SVMModel(SLModel):
    def __init__(self, random_state: int = 47):
        super().__init__("SL_training/svm_model.pkl", random_state)

    def _initialize_model(self):
        return SVC(kernel='rbf', random_state=self.random_state)