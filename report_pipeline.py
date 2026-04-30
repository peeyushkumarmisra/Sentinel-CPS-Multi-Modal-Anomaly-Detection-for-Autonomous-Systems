"""
File: report_pipeline.py (Optimized for Single-File Output)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from collections import Counter

from Env.spanwer import DataSpawner
from Env.env_gen import MapGenerator
from SL.sl_inference import SLInference
from USL.usl_inference import USLInference
from RL.rl_inference import RLInference

SL_MODELS  = ["rf_model.pkl",    "svm_model.pkl"]
USL_MODELS = [("gmm", "gmm_model.pkl"), ("dec", "dec_model.pt")]
RL_MODELS  = ["q_brain.json",    "sarsa_brain.json"]

def parse_label(label):
    if label.startswith("robotic_arm"):
        return "robotic_arm", label[12:]
    elif label.startswith("agv_unit"):
        return "agv_unit", label[9:]
    else:
        parts = label.split("_")
        return parts[0], "_".join(parts[1:])

def evaluate_levels(gt, air, override, sl_ans, sl_conf, usl_ans, usl_conf):
    gt_base, _          = parse_label(gt)
    sl_base, sl_attrs   = parse_label(sl_ans)
    l1 = (sl_base == gt_base) and (sl_conf > 0.98)
    fly_ok      = ("fly" in sl_attrs) if air else ("non_fly" in sl_attrs or "fly" not in sl_attrs)
    override_ok = ("override" in sl_attrs) if override else ("non_override" in sl_attrs or "override" not in sl_attrs)
    l2 = fly_ok and override_ok
    l3 = (usl_ans == gt) and (usl_conf > 0.95)
    return l1, l2, l3, int(l1) + int(l2) + int(l3)

def aggregate_usl_result(result, class_names):
    majority_cluster = Counter(result.predictions).most_common(1)[0][0]
    majority_mask    = result.predictions == majority_cluster
    return class_names[majority_cluster], float(np.mean(result.confidences[majority_mask]))

def combo_tag(sl_file, usl_name, rl_file):
    sl = sl_file.replace(".pkl", "").replace(".pt", "")
    rl = rl_file.replace(".json", "")
    return f"{sl}__{usl_name}__{rl}"

if __name__ == "__main__":
    Path("RESULTS").mkdir(exist_ok=True)
    print("[1] Initializing Data Spawner and Environments...")
    spawner = DataSpawner(csv_path="DATA/SENSOR_STATS.csv", image_dir="DATA/DATASET")
    map_gen = MapGenerator(seed=47)
    map_gen.generate_map()

    # Consolidated containers
    all_df1_rows = []
    df2_rows = []
    
    total_pipeline_time = 0.0
    combo_count = 0

    # Grid Search: 8 combinations
    for sl_file, (usl_name, usl_file), rl_file in product(SL_MODELS, USL_MODELS, RL_MODELS):
        combo_count += 1
        tag = combo_tag(sl_file, usl_name, rl_file)
        print(f"\n[Combo {combo_count}/8] Running: {tag}")

        # Load models
        sl_engine  = SLInference(model_dir="MODELS",  model_file=sl_file)
        usl_engine = USLInference(model_dir="MODELS", pipeline_name=usl_name, seed=47)
        rl_engine  = RLInference(model_dir="MODELS",  qtable_file=rl_file)

        combo_total_time = 0.0
        level_cleared_sum = 0

        # --- SL & USL Evaluation ---
        for loc_id in range(10):
            target_class, sampled_rows_df, images_paths, gt_air, gt_override = spawner.get_payload(loc_id)

            sl_ans, sl_conf, sl_time = sl_engine.run(sampled_rows_df.to_dict('records'))
            usl_result = usl_engine.run(images_paths)
            usl_ans, usl_conf = aggregate_usl_result(usl_result, spawner.classes)
            usl_time = usl_result.elapsed_sec

            combo_total_time += sl_time + usl_time

            l1, l2, l3, lvl = evaluate_levels(
                target_class, gt_air, gt_override,
                sl_ans, sl_conf, usl_ans, usl_conf
            )
            level_cleared_sum += lvl

            # Append to master DF1 list with model identifiers
            all_df1_rows.append({
                "Location_ID": loc_id,
                "Ground_Truth": target_class, "Airborne_(GT)": gt_air, "Override_(GT)": gt_override,
                "SL_Answer": sl_ans, "SL_Conf": sl_conf,
                "USL_Answer": usl_ans, "USL_Conf": usl_conf,
                "L1_Cleared": l1, "L2_Cleared": l2, "L3_Cleared": l3, "Level_Cleared": lvl,
                "USL_Model": usl_file, "SL_Model": sl_file, "RL_Model": rl_file
            })

        # --- RL Navigation ---
        gif_path = f"RESULTS/rl_animation__{tag}.gif"
        path, reward, rl_time = rl_engine.run(map_gen, animate=True, gif_path=gif_path)

        combo_total_time += rl_time
        total_pipeline_time += combo_total_time

        # Append to DF2 list
        df2_rows.append({
            "SL_Model": sl_file,
            "USL_Model": usl_file,
            "RL_Model": rl_file,
            "Points_Level_Cleared": level_cleared_sum,
            "Total_Rewards_Collected": round(reward, 4),
            "Time_Taken_RL": round(rl_time, 4),
            "Total_Time": round(combo_total_time, 4)
        })

    # --- SAVE CONSOLIDATED DF1 ---
    df1 = pd.DataFrame(all_df1_rows)
    df1_cols = [
        "Location_ID", "Ground_Truth", "Airborne_(GT)", "Override_(GT)", 
        "SL_Answer", "SL_Conf", "USL_Answer", "USL_Conf", 
        "L1_Cleared", "L2_Cleared", "L3_Cleared", "Level_Cleared", 
        "USL_Model", "SL_Model", "RL_Model"
    ]
    df1 = df1[df1_cols]
    df1.to_csv("RESULTS/df1_all_combinations.csv", index=False)

    # --- SAVE CONSOLIDATED DF2 ---
    df2 = pd.DataFrame(df2_rows)
    df2_cols = [
        "SL_Model", "USL_Model", "RL_Model", 
        "Points_Level_Cleared", "Total_Rewards_Collected", 
        "Time_Taken_RL", "Total_Time"
    ]
    df2 = df2[df2_cols]
    df2.to_csv("RESULTS/df2_all_combinations.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Grid Search Complete. Results saved in 'RESULTS/' folder.")
    print(f"Total Combined Rows in DF1: {len(df1)}")
    print(f"Animations Generated: {combo_count}")
    print(f"Grand Total Pipeline Time: {total_pipeline_time:.2f}s")
    print(f"{'='*60}")