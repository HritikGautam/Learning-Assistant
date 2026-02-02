def run():
    import whisper
    import json
    import os
    import streamlit as st

    @st.cache_resource
    def load_whisper():
        return whisper.load_model("base")

    model = load_whisper()

    audios = os.listdir("audios")
    for audio in audios:
        result = model.transcribe(
            audio=f"audios/{audio}",
            # language="hi",
            task="translate",
            word_timestamps=False,
        )
        chunks = []
        for segment in result["segments"]:
            chunks.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                }
            )

        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata, f)


if __name__ == "__main__":
    run()
