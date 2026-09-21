def process_docs(dataset):
    def valid(doc):
        prompt = doc["prompt"]
        if prompt.strip().startswith("Question:"):
            return True
        return False
    return dataset.filter(valid)
