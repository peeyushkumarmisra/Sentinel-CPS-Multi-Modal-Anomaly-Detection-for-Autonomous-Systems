"""
File: spanwer.py
This script manages the dynamic data generation for the simulation, responsible for pairing 
physical assets with their corresponding sensor data and images at each spawner node.

1. DataSpawner:
    * init: Initializes the spawner with the paths to the tabular CSV data and the image 
            repository. It defines five primary asset classes: Robotic Arm, PLC Controller,
            AGV Unit, Drone, and CNC Machine.
    * generate_node:    Creates a randomized list of 10 asset assignments for the map nodes. It
                        guarantees that each of the five classes appears at least once, filling
                        the remaining slots with random selections.
    * get_payload: Retrieves a specific dataset for a given node index.
        * Identifies the target class assigned to that node.
        * Samples 100 unique rows of tabular sensor data from the CSV.
        * Randomly selects 100 corresponding .png image paths from the class-specific image directory.
        * Returns the class name, the sampled data rows, and the list of image paths.
"""


import os
import random
import pandas as pd

class DataSpawner:
    def __init__(self, csv_path, image_dir):
        self.image_dir = image_dir
        self.classes = ['Robotic Arm', 'PLC Controller', 'AGV Unit', 'Drone', 'CNC Machine']

        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"ERROR: Could not find CSV at {csv_path}")
            self.df = pd.DataFrame()

        # Generating random class for the 10 nodes
        self.node = self.generate_node()
        print("\n[MAP GENERATED] Asset Assignments for this Patrol:")
        for idx, asset in enumerate(self.node_assignments):
            print(f"  Node [{idx:02d}]: {asset}")

    def generate_node(self): # Ensures 1 of each class, then 5 random
        base_assignment = self.classes.copy()
        random_assignment = random.choices(self.classes, k=5)
        full_assignment = base_assignment + random_assignment
        random.shuffle(full_assignment)
        return full_assignment
    
    def get_payload(self, node_idx): # Retrieves exactly 100 tabular rows and 100 image paths
        if not (0 <= node_idx <= 9):
            raise ValueError("node_idx must be between 0 and 9")
        target_class = self.node[node_idx]

        # Extracting rows from csv
        class_df = self.df[self.df['Asset Class'] == target_class]
        sampled_rows = class_df.sample(n=100) # Get 100 random unique rows

        # Extracting images from base_dir
        class_img_dir = os.path.join(self.image_dir, target_class)
        image_paths = []

        if os.path.exists(class_img_dir):
            valid_ext = ('.png')
            all_files = [os.path.join(class_img_dir, f) for f in os.listdir(class_img_dir) if f.lower().endswith(valid_ext)]
            image_paths = random.sample(all_files, 100)
        else:
            print(f"Image directory not found for {target_class} at {class_img_dir}")

        return target_class, sampled_rows, image_paths
