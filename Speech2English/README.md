
Download the Librispeech pretrained model from here: https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth then put it in the `models` folder.

To perform inference, run:
	
	`python deepspeech.pytorch/transcribe.py model.model_path=Speech2English/deepspeech.pytorch/models/librispeech_pretrained_v2.pth audio_path=/path/to/audio.wav`

The output text will be saved at `output/`.

