# Training from Scratch and Fine-tuning for Automatic Speech Recognition (Speech to Text) on low-resource languages

In this repository, we will see how to perform training from scratch and fine-tuning for automatic speech recognition (ASR) for many languages.

### Table Of Contents
- [QuartzNet](#QuartzNet)
- [Download Free Audio Data for ASR](#Download-Free-Audio-Data-for-ASR)
- [Speech Data Augmentation](#Speech-Data-Augmentation)
- [Create Custom ASR Data Manifest](#Create-Custom-ASR-Data-Manifest)
- [Specify Model YAML Configuration](#Specify-Model-YAML-Configuration)
- [Training from Scratch (Transfer Learning from English to Turkish)](#Training-from-Scratch-(Transfer-Learning-from-English-to-Turkish))
- [Fine-tuning for Pretrained Turkish Model](#Fine-tuning-for-Pretrained-Turkish-Model)
- [Convert the PyTorch model to ONNX](#Convert-the-PyTorch-model-to-ONNX)
- [Inference](#Inference)
- [Evaluation with Word Error Rate (WER)](#Evaluation-with-Word-Error-Rate-(WER))

---
### QuartzNet
QuartzNet-15×5 trained on LibriSpeech, Mozilla Common Voice, WSJ, Fisher, and Switchboard datasets combined. In total, the training data used to pretrain this model consists of ~3,300 hours of transcribed English speech. In this study, pretrained English checkpoint and network configuration file will be used.
Because the model initialized with English checkpoints trains faster (training loss goes down faster) and achieves much better generalization performance.

### Download Free Audio Data for ASR
You can download and create `manifest.jsonl` from some of the common publically available speech dataset for **many languages** from my repository [speech-datasets-for-ASR](https://github.com/Rumeysakeskin/speech-datasets-for-ASR).

After the data is downloaded, pre-process it. Convert MP3 files into WAV files with a 16kHz sampling rate to match the sampling rate of the QuartzNet model training data.

---
### Speech Data Augmentation
Also, you can use my repository [
speech-data-augmentation](https://github.com/Rumeysakeskin/speech-data-augmentation) to **increase the diversity** of your dataset augmenting the data artificially for ASR models training.

---
### Create Custom ASR Data Manifest
We need to do now is to create manifests for our training and evaluation data, which will contain the metadata of our audio files.
The `nemo_asr` collection expects each dataset to consist of a set of utterances in individual audio files plus a manifest that describes the dataset, with information about one utterance per line `(.json)`.
- Each line of the manifest `(data/train_manifest.jsonl and data/val_manifest.jsonl)` should be in the following format:
```python
{"audio_filepath": "/data/train_wav/audio_1.wav", "duration": 2.836326530612245, "text": "bugün hava durumu nasıl"}
```
The `audio_filepath` field should provide an absolute path to the `.wav` file corresponding to the utterance. The `text` field should contain the full transcript for the utterance, and the `duration` field should reflect the duration of the utterance in seconds.

- Remove everything that is not belong to your language letter or space. The Turkish vocabulary consists of 29 letters and the space character.
- Preprocess all transcripts by turning all letters into lowercase.
- Take speech samples shorter that 25 seconds.
---
### Specify Model YAML Configuration

Since QuartzNet is a CTC-model that outputs words character-by-character, the fact that these languages all have different alphabets means that there is no way you can reuse an English ASR network as-is. You can also find more information about this in [ASR CTC Language Finetuning](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb). 

- Configure labels according to your language. 


In this study, Turkish labels were configured in `configs/quartznet15x5.yaml` in the following format:
```python
labels: &labels [" ", "a", "b", "c", "ç", "d", "e", "f", "g", "ğ", "h", "ı", "i", "j", "k", "l", "m",
         "n", "o", "ö", "p", "q", "r", "s", "ş", "t", "u", "ü", "v", "w", "x", "y", "z", "'"]
```
- Turn off default transcript normalization because it was designed for the English language. Set `normalize_transcripts: False` in the config file.

---
### Training from Scratch (Transfer Learning from English to Turkish)

The NeMo QuartzNet implementation consists of two neural modules: encoder and decoder.
The encoder module contains most of the network’s weights. It can be thought of as a module that handles acoustics and produces a hidden representation for spoken language (encoding). The decoder takes that representation and generates letters from the target language’s alphabet. 
You don’t re-use the decoder because the alphabets are different. However, you can still reuse the encoder.
```python
first_asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained("QuartzNet15x5Base-En")
```
Open and run the `transfer_learning_from_English_to_Turkish.ipynb` in the Colab.

---
### Fine-tuning for Pretrained Turkish Model

Previous step, we trained the model for 100 epochs and save the models' checkpoints.
Now, in fine-tuning step we load the last model checkpoint to improve especially for domain-specific applications. 
```python
model_to_load = "./pretrained_turkish_model/epoch-99.ckpt"

first_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']))
first_asr_model = first_asr_model.load_from_checkpoint(model_to_load)
```
Run the following command:
```python
python fine_tune.py
```

 Pretrained QuartzNet15x5 Parameters | Transfer-learning QuartzNet15x5 Parameters |
 ------- | ------- |
<img src="QuartzNet_params_pretrained_English.png" width="350" height="145"> |  <img src="quartznet15x5_transfer_learning_params.png" width="350" height="145"> |

---
### Convert the PyTorch model to ONNX
To convert the resulting model you need to `torch.onnx.export`, run the following jupyter notebook to export onnx model.
```python
export_model.ipynb
```

---
### Inference
ONNX Runtime works with different hardware acceleration libraries through its extensible Execution Providers (EP) framework to optimally execute the ONNX models on the hardware platform. 
To find the best performance and apply performance-tuning for your model and hardware with [ONNX Runtime](https://onnxruntime.ai/docs/performance/tune-performance.html).
```python
python stt_inferencer.py
```

---
### Evaluation with Word Error Rate (WER)
```python
evaluate_model.ipynb
```

