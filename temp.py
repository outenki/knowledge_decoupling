import evaluate

metric = evaluate.load("squad_v2")

predictions = [{
    "id": "1",
    "prediction_text": "unanswerable",
    "no_answer_probability": 0.9,
}]

references = [{
    "id": "1",
    "answers": {
        "text": [],
        "answer_start": [],
    },
}]

print(metric.compute(
    predictions=predictions,
    references=references,
))
