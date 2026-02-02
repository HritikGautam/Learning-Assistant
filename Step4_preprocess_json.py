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
        return model.embed_documents(text_list)

    jsons = os.listdir("merged_jsons")
    print(f"DEBUG: Processing {len(jsons)} files: {jsons}")

    my_dicts = []
    chunk_id = 0

    for json_file in jsons:
        try:
            with open(os.path.join("merged_jsons", json_file)) as f:
                content = json.load(f)

            # Skip if chunks are missing
            if not content.get("chunks"):
                continue

            embeddings = create_embedding([c["text"] for c in content["chunks"]])

            for i, chunk in enumerate(content["chunks"]):
                chunk["chunk_id"] = chunk_id
                chunk["embedding"] = embeddings[i]
                chunk_id += 1
                my_dicts.append(chunk)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    # FIX: Explicitly define columns so the file is never "invalid"
    columns = ["chunk_id", "embedding", "start", "end", "text"]
    if my_dicts:
        df = pd.DataFrame.from_records(my_dicts)
    else:
        # Create an empty dataframe with the required columns
        df = pd.DataFrame(columns=columns)

    joblib.dump(df, "Step5_embeddings.joblib")
    print("Database saved successfully.")


# print("Embeddingd created!")
if __name__ == "__main__":
    run()
