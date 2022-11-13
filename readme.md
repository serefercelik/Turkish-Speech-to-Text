## Turkish Text-to-Speech

### Preparing Custom ASR Data
The `nemo_asr` collection expects each dataset to consist of a set of utterances in individual audio files plus a manifest that describes the dataset, with information about one utterance per line `(.json)`.
Each line of the manifest `(data/train_manifest.jsonl and data/val_manifest.jsonl)` should be in the following format:
```
{"audio_filepath": "/data/train_wav/audio_1.wav", "duration": 2.836326530612245, "text": "bugün hava durumu nasıl"}
```
The `audio_filepath` field should provide an absolute path to the `.wav` file corresponding to the utterance. The `text` field should contain the full transcript for the utterance, and the `duration` field should reflect the duration of the utterance in seconds.

### Download Free Audio Data for ASR
You can download and create `manifest.jsonl` from some of the common publically available speech dataset in **Turkish** and some **other languages** from my reporisitory [speech-datasets-for-ASR](https://github.com/Rumeysakeskin/speech-datasets-for-ASR).

### Speech Data Augmentation
Also, you can use my repository [
speech-data-augmentation](https://github.com/Rumeysakeskin/speech-data-augmentation) to **increase the diversity** of your dataset augmenting the data artificially for ASR models training.

### Preprocessing
- [QuartzNet 15x5 model config file](https://catalog.ngc.nvidia.com/orgs/nvidia/models/quartznet_15x5_ls_sp/files) was used that trained only on LibriSpeech.
Turkish labels were configured in `configs/quartznet15x5.yaml` in the following format:
```
labels: &labels [" ", "a", "b", "c", "ç", "d", "e", "f", "g", "ğ", "h", "ı", "i", "j", "k", "l", "m",
         "n", "o", "ö", "p", "q", "r", "s", "ş", "t", "u", "ü", "v", "w", "x", "y", "z", "'"]
```
- Turkish model checkpoint `(pretrained_turkish_model/epoch-99.ckpt)` was used for fine-tuning.

### Training
Run the following command:
```
python fine_tune.py
```
### Load and Export onnx model to Inference
```
export_model.ipynb
```
### Deploy model and inference
Run the following command:
```
python stt_inferencer.py
```
### Evaluate model with Word Error Rate (WER)
```
evaluate_model.ipynb
```


