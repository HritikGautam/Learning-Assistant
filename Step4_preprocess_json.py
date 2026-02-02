def run():
    # import requests
    import os
    import json
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    import pandas as pd
    import joblib

    # Initialize the model once outside the function
    model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    def create_embedding(text_list):
        # This runs on Streamlit's CPU—no local server needed!
        embeddings = model.embed_documents(text_list)
        return embeddings

    jsons = os.listdir("merged_jsons")  # List all the jsons
    my_dicts = []
    chunk_id = 0

    for json_file in jsons:
        with open(f"merged_jsons/{json_file}") as f:
            content = json.load(f)
        embeddings = create_embedding([c["text"] for c in content["chunks"]])

        for i, chunk in enumerate(content["chunks"]):
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embeddings[i]
            chunk_id += 1
            my_dicts.append(chunk)

    # print(my_dicts)

    df = pd.DataFrame.from_records(my_dicts)
    joblib.dump(df, "Step5_embeddings.joblib")


# print("Embeddingd created!")
if __name__ == "__main__":
    run()
