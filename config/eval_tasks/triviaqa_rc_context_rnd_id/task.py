def process_docs(dataset):
    def valid(doc):
        prompt = doc["prompt"]
        if not prompt.startswith("Background:"):
            return False
        if prompt.lstrip("Background:").lstrip().startswith("Question:"):
            return False
        return True
    return dataset.filter(valid)
