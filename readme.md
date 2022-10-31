### Turkish Text-to-Speech

#### Preparing Custom ASR Data
The `nemo_asr` collection expects each dataset to consist of a set of utterances in individual audio files plus a manifest that describes the dataset, with information about one utterance per line `(.json)`.
Each line of the manifest `(data/train_manifest.jsonl and data/val_manifest.jsonl)` should be in the following format:
```
{"audio_filepath": "/path/to/audio.wav", "text": "the transcription of the utterance", "duration": 3.147}
```
The `audio_filepath` field should provide an absolute path to the `.wav` file corresponding to the utterance. The `text` field should contain the full transcript for the utterance, and the `duration` field should reflect the duration of the utterance in seconds.

#### Preprocessing
QuartzNet 15x5 checkpoint was used that trained only on LibriSpeech.
Turkish labels were configured in `configs/quartznet5x5.yaml` in the following format:
```
labels: &labels [" ", "a", "b", "c", "ç", "d", "e", "f", "g", "ğ", "h", "ı", "i", "j", "k", "l", "m",
         "n", "o", "ö", "p", "q", "r", "s", "ş", "t", "u", "ü", "v", "w", "x", "y", "z", "'"]
```
#### Training
Run the following command:
```
python fine_tune.py
```
#### Load and Export onnx model to Inference
```
export_model.ipynb
```
#### Deploy model and make inference
Run the following command:
```
stt_inferencer.py
```

