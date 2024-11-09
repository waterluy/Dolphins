import json

records = 'results/bench_attack_coi_eps0.2_iter100_query8/records.json'
final_answer = []
with open(records, mode='r') as file:
    data = json.load(file)
    for  item in data:
        final_answer.append({
            "unique_id": item["unique_id"],
            "task_name": item["task_name"],
            "pred": item["induction_answers"][-1],
        })
with open('playground/dolphins_bench/dolphins_benchmark.json', 'r') as file:
    data = json.load(file)
    for item in final_answer:
        target = list(filter(lambda x: x["id"] == item["unique_id"], data))
        assert len(target) == 1
        target = target[0]
        item["gt"] = target["conversations"][1]["value"]
        item["label"] = target["label"]
with open('results/bench_attack_coi_eps0.2_iter100_query8/dolphin_output.json', 'w') as file:
    for line in final_answer:
        file.write(json.dumps(line) + "\n")
