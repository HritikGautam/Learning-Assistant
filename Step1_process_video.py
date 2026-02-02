def run():
    # Converts the videos to mp3
    import os
    import subprocess

    os.makedirs("audios", exist_ok=True)
    files = os.listdir("videos")
    for file in files:
        try:
            if file.count(".") > 1:
                raise Exception(
                    "File name contains extra dots.Rename it as: filename.extension (e.g., video1.mp4)"
                )
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")
            continue

        file_name = file.split(".")[0]
        output_path = f"audios/{file_name}.mp3"
        if not os.path.exists(output_path):
            # Added -y to overwrite if needed and simplified path handling
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    f"videos/{file}",
                    "-q:a",
                    "0",
                    "-map",
                    "a",
                    output_path,
                ]
            )


if __name__ == "__main__":
    run()
