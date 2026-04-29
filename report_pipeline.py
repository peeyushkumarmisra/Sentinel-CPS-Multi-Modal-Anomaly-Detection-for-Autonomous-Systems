"""
File: report_pipeline.py
"""

import numpy as np
import pandas as pd
from collections import Counter
from RL_env.spanwer import DataSpawner
from RL_env.env_gen import MapGenerator
from SL_training.sl_inference import SLInference
from USL_training.usl_inference import USLInference
from RL_training.rl_inference import RLInference


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
    fly_ok      = ("fly" in sl_attrs)      if air      else ("non_fly" in sl_attrs or "fly" not in sl_attrs)
    override_ok = ("override" in sl_attrs) if override else ("non_override" in sl_attrs or "override" not in sl_attrs)
    l2 = fly_ok and override_ok
    l3 = (usl_ans == gt) and (usl_conf > 0.95)
    return l1, l2, l3, int(l1) + int(l2) + int(l3)


def aggregate_usl_result(result, class_names):
    majority_cluster = Counter(result.predictions).most_common(1)[0][0]
    majority_mask    = result.predictions == majority_cluster
    return class_names[majority_cluster], float(np.mean(result.confidences[majority_mask]))


if __name__ == "__main__":
    print("[1] Initializing Data Spawner and Environments...")
    spawner = DataSpawner(csv_path="DATA/SENSOR_STATS.csv", image_dir="DATA/DATASET")
    map_gen = MapGenerator(seed=47)
    map_gen.generate_map()

    print("[2] Loading Models...")
    sl_engine  = SLInference(model_dir="MODELS")
    usl_engine = USLInference(model_dir="MODELS", pipeline_name="gmm", seed = 47)
    rl_engine  = RLInference(model_dir="MODELS")

    df1_rows            = []
    total_pipeline_time = 0.0

    print("\n--- Running DF1 Evaluation (10 Locations, 100 Samples Each) ---")
    for loc_id in range(10):
        target_class, sampled_rows_df, images_paths, gt_air, gt_override = spawner.get_payload(loc_id)

        sl_ans,  sl_conf,  sl_time  = sl_engine.run(sampled_rows_df.to_dict('records'))
        usl_result                  = usl_engine.run(images_paths)
        usl_ans, usl_conf           = aggregate_usl_result(usl_result, spawner.classes)
        usl_time                    = usl_result.elapsed_sec

        total_pipeline_time += sl_time + usl_time

        l1, l2, l3, lvl = evaluate_levels(
            target_class, gt_air, gt_override,
            sl_ans, sl_conf, usl_ans, usl_conf
        )
        df1_rows.append({
            "Location_ID":    loc_id,
            "Ground_Truth":   target_class, "Airborne_(GT)": gt_air,  "Override_(GT)": gt_override,
            "SL_Answer":      sl_ans,       "SL_Conf":       sl_conf,  "L1_Cleared": l1, "L2_Cleared": l2,
            "USL_Answer":     usl_ans,      "USL_Conf":      usl_conf, "L3_Cleared": l3, "Level_Cleared": lvl
        })

    df1 = pd.DataFrame(df1_rows)
    print("\n[DF1 RESULTS]")
    print(df1.to_string(index=False))
    df1.to_csv("report_df1.csv", index=False)

    print("\n--- Running RL Navigation & Animation ---")
    path, reward, rl_time = rl_engine.run(map_gen, animate=True)
    total_pipeline_time  += rl_time

    print(f"Agent Path Length: {len(path)} steps")
    print(f"Total RL Reward: {reward}")
    print(f"Total Points Cleared (Out of 30): {df1['Level_Cleared'].sum()}")
    print(f"Total Unified Run Time: {total_pipeline_time:.2f} seconds")