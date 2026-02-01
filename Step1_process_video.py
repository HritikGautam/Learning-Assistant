def run():
    # Converts the videos to mp3
    import os
    import subprocess

    files = os.listdir("videos")
    for file in files:
        # tutorial_number = file.split(" #")[1].split(".")[0]
        # print( tutorial_number,  file_name)

        try:
            if file.count(".") > 1:
                raise Exception(
                    "File name contains extra dots.Rename it as: filename.extension (e.g., video1.mp4)"
                )
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")
            continue

        file_name = file.split(".")[0]
        subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{file_name}.mp3"])


# print("Video processing finished")
if __name__ == "__main__":
    run()
