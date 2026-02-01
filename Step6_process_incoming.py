import requests

# import os
# import json
# import numpy as np

# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

# from read_chunks import create_embedding
import joblib


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post(
        "http://localhost:11434/api/embed", json={"model": "bge-m3", "input": text_list}
    )

    embedding = r.json()["embeddings"]
    return embedding


def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            # "model": "deepseek-r1",
            "model": "mistral:7b-instruct",
            "prompt": prompt,
            "stream": False,
        },
    )

    response = r.json()
    # print(response) This prints the dataframe too before answering the question
    return response


# Creating a embedding function for incoming query

embedding_cache = {}


def get_question_embedding(question):
    if question in embedding_cache:
        return embedding_cache[question]
    emb = create_embedding([question])[0]
    embedding_cache[question] = emb
    return emb


df = joblib.load("Step5_embeddings.joblib")


# incoming_query = input("Ask a Question: ")
# # question_embedding = get_question_embedding([incoming_query])[0]
# question_embedding = get_question_embedding(incoming_query)


# # Find similarities of question_embedding with other embeddings
# # print(np.vstack(df['embedding'].values))
# # print(np.vstack(df['embedding']).shape)
# similarities = cosine_similarity(
#     np.vstack(df["embedding"]), [question_embedding]
# ).flatten()
# # print(similarities)
# top_results = 3
# max_indx = similarities.argsort()[::-1][0:top_results]
# # print(max_indx)
# new_df = df.loc[max_indx]
# # print(new_df[["title", "number", "text"]])


# prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

#     {new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
#     ---------------------------------
#     "{incoming_query}"
#     User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
#     '''
# with open("prompt.txt", "w") as f:
#     f.write(prompt)

# response = inference(prompt)["response"]
# print(response)

# with open("response.txt", "w") as f:
#     f.write(response)


# # for index, item in new_df.iterrows():
# #     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])
