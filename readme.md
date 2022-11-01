### Turkish Text-to-Speech

#### Preparing Custom ASR Data
The `nemo_asr` collection expects each dataset to consist of a set of utterances in individual audio files plus a manifest that describes the dataset, with information about one utterance per line `(.json)`.
Each line of the manifest `(data/train_manifest.jsonl and data/val_manifest.jsonl)` should be in the following format:
```
{"audio_filepath": "/data/train_wav/audio_1.wav", "duration": 2.836326530612245, "text": "bugün hava durumu nasıl"}
```
The `audio_filepath` field should provide an absolute path to the `.wav` file corresponding to the utterance. The `text` field should contain the full transcript for the utterance, and the `duration` field should reflect the duration of the utterance in seconds.

#### Preprocessing
- [QuartzNet 15x5 model config file](https://catalog.ngc.nvidia.com/orgs/nvidia/models/quartznet_15x5_ls_sp/files) was used that trained only on LibriSpeech.
Turkish labels were configured in `configs/quartznet15x5.yaml` in the following format:
```
labels: &labels [" ", "a", "b", "c", "ç", "d", "e", "f", "g", "ğ", "h", "ı", "i", "j", "k", "l", "m",
         "n", "o", "ö", "p", "q", "r", "s", "ş", "t", "u", "ü", "v", "w", "x", "y", "z", "'"]
```
- Turkish model checkpoint `(pretrained_turkish_model/epoch-99.ckpt)` was used for fine-tuning.

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

