import datasets


def load_responses(path):
    data = datasets.load_from_disk(path)
    # print(f"Loaded dataset: {data}")
    return data
