# Training from Scratch and Fine-tuning for Automatic Speech Recognition (Speech to Text) on low-resource languages

In this repository, we will see how to perform training from scratch and fine-tuning for automatic speech recognition (ASR) for many languages (especially low-resource).

- For both training from scratch and fine-tuning, you need to prepare your training data and specify your model architecture in the same way.

### Table Of Contents
- [Download Free Audio Data for ASR](#Download-Free-Audio-Data-for-ASR)
- [Speech Data Augmentation](#Speech-Data-Augmentation)
- [Create Custom ASR Data Manifest](#Create-Custom-ASR-Data-Manifest)
- [Specify Model YAML Configuration](#Specify-Model-YAML-Configuration)
- [Training from Scratch (Transfer Learning from English to Turkish)](#Training-from-Scratch-(Transfer-Learning-from-English-to-Turkish))
- [Fine-tuning for Pretrained Turkish Model](#Fine-tuning-for-Pretrained-Turkish-Model)
- [Export to ONNX Model](#Export-to-ONNX-Model)
- [Inference](#Inference)
- [Evaluation with Word Error Rate (WER)](#Evaluation-with-Word-Error-Rate-(WER))


### Download Free Audio Data for ASR
You can download and create `manifest.jsonl` from some of the common publically available speech dataset for **many languages** from my repository [speech-datasets-for-ASR](https://github.com/Rumeysakeskin/speech-datasets-for-ASR).

### Speech Data Augmentation
Also, you can use my repository [
speech-data-augmentation](https://github.com/Rumeysakeskin/speech-data-augmentation) to **increase the diversity** of your dataset augmenting the data artificially for ASR models training.

### Create Custom ASR Data Manifest
We need to do now is to create manifests for our training and evaluation data, which will contain the metadata of our audio files.
The `nemo_asr` collection expects each dataset to consist of a set of utterances in individual audio files plus a manifest that describes the dataset, with information about one utterance per line `(.json)`.
Each line of the manifest `(data/train_manifest.jsonl and data/val_manifest.jsonl)` should be in the following format:
```
{"audio_filepath": "/data/train_wav/audio_1.wav", "duration": 2.836326530612245, "text": "bugün hava durumu nasıl"}
```
The `audio_filepath` field should provide an absolute path to the `.wav` file corresponding to the utterance. The `text` field should contain the full transcript for the utterance, and the `duration` field should reflect the duration of the utterance in seconds.


### Specify Model YAML Configuration
In this study, [QuartzNet 15x5 model config file](https://catalog.ngc.nvidia.com/orgs/nvidia/models/quartznet_15x5_ls_sp/files) was used that pre-trained only on LibriSpeech.
Since QuartzNet is a CTC-model that outputs words character-by-character, 
the fact that these languages all have different alphabets means that there is no way you can reuse an English ASR network as-is. 
Then, you have to configure labels according to your language. You can find more information about this in [ASR CTC Language Finetuning](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb).

In this code, Turkish labels were configured in `configs/quartznet15x5.yaml` in the following format:
```
labels: &labels [" ", "a", "b", "c", "ç", "d", "e", "f", "g", "ğ", "h", "ı", "i", "j", "k", "l", "m",
         "n", "o", "ö", "p", "q", "r", "s", "ş", "t", "u", "ü", "v", "w", "x", "y", "z", "'"]
```

### Training from Scratch (Transfer Learning from English to Turkish)

The NeMo QuartzNet implementation consists of two neural modules: encoder and decoder.
The encoder module contains most of the network’s weights. It can be thought of as a module that handles acoustics and produces a hidden representation for spoken language (encoding). The decoder takes that representation and generates letters from the target language’s alphabet. You don’t re-use the decoder because the alphabets are different. However, you can still reuse the encoder.

As a control for these experiments, I trained the same QuartzNet model from scratch.

### Fine-tuning for Pretrained Turkish Model

We will use Turkish model checkpoint `(pretrained_turkish_model/epoch-99.ckpt)`.

Run the following command:
```
python fine_tune.py
```


 Pretrained QuartzNet15x5 Parameters | Transfer-learning QuartzNet15x5 Parameters |
 ------- | ------- |
<img src="QuartzNet_params_pretrained_English.png" width="350" height="145"> |  <img src="quartznet15x5_transfer_learning_params.png" width="350" height="145"> |

### Export to ONNX Model
```
export_model.ipynb
```
### Inference
ONNX Runtime works with different hardware acceleration libraries through its extensible Execution Providers (EP) framework to optimally execute the ONNX models on the hardware platform. 
To find the best performance and apply performance-tuning for your model and hardware with [ONNX Runtime](https://onnxruntime.ai/docs/performance/tune-performance.html).
```
python stt_inferencer.py
```
### Evaluation with Word Error Rate (WER)
```
evaluate_model.ipynb
```

 Transfer Learning Model Size | Fine Tuning Model Size | Onnx Model Size |
 ------- | ------- |------- |
53 MB |   |   |


