import sys
import json
import os


in_file, out_dir, out_path = sys.argv[1:4]

with open(in_file, "r"
          ) as f, open(os.path.join(out_dir, "wav.scp"), "w"
                       ) as f1, open(os.path.join(out_dir, "text"
                                                  ), "w") as f2:
    for line in f:
        line = line.strip()
        data = json.loads(line)
        audio = data.get("audio").get("path")
        audio_path = '/'.join(audio.split('/')[2:])
        audio_path = os.path.join(out_path, audio_path)
        key = audio_path.split('/')[-1].split('.')[0]
        task = "transcribe"
        data_new = {"key": key, "language": "chinese", "task": task,
                    "duration": data["duration"], "sentence": data["sentence"], "sentences": data["sentences"]}
        print(f"{key} {audio_path}", file=f1)
        print(f"{key} {json.dumps(data_new, ensure_ascii=False)}", file=f2)
