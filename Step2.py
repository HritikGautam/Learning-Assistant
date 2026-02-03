# Mp3 to jsons
import whisper
import json
import os
# from pathlib import Path

# jsons = os.listdir("jsons")
model = whisper.load_model("large-v2")

jsons = "jsons/"
audios = os.listdir("audios")


for audio in audios:
    audio_name = audio.split(".")[0]
    json_file_name = os.path.join(jsons, f"{audio_name}.json")
    if os.path.isfile(json_file_name):
        continue
    else:
        print(f"Creating json for {audio_name}")
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(number, title)
        result = model.transcribe(
            audio=f"audios/{audio}",
            # result = model.transcribe(audio=f"audios/sample.mp3",
            language="hi",
            task="translate",
            word_timestamps=False,
        )
        chunks = []
        for segment in result["segments"]:
            chunks.append(
                {
                    "number": number,
                    "title": title,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                }
            )

        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata, f)
