import csv
import os
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default='csvfiles/dolphins_benchmark_attack_online_gpt_target_0.002_500.csv')
    args = parser.parse_args()
    
    json_file = args.csv.split('.')[0].split('/')[1] + '.json'
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames)
        data = [row for row in reader]
        
    with open('playground/dolphins_bench/dolphins_benchmark.json', 'r') as f:
        dolphins_benchmark = json.load(f)
         
    with open(os.path.join("results", json_file), 'w') as f:
        for row in data:
            video_path = row['video_path']
            for item in dolphins_benchmark:
                if item['video_path'] == ("dolphins/" + video_path):
                    unique_id = item["id"]
                    label = item["label"]
                    break
            task_name = row['task_name']
            pred = row['dolphins_inference']
            gt = row['ground_truth']
            f.write(json.dumps({"unique_id": unique_id, "task_name": task_name, "pred": pred, "gt": gt, "label": label}) + "\n")
        