# Speech-to-ASL
Translates spoken english audio to an ASL avatar video.

The pipeline is composed of multiple models:
1. Translate spoken to english to text
2. Translate english text to ASL gloss using transformer model
3. Align gloss to ASL poses. Poses were generated using OpenPose.
4. Interpolate ASL pose transitions using fully connected model
5. Generate avatar images for each pose using pix2pix GAN model
6. Compile images as video

Part of Microsoft 2020 summer Hackathon. Please see video: aka.ms/AASL

### 1. Install requirements
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Download trained models
#### English to gloss
```
python -m spacy download en_core_web_sm
```
Download the model and vocabulary and unzip in Speech-to-ASL/English2Gloss/
https://drive.google.com/file/d/15PVrfsPG3IYJh0w4nKLgdp6eMULl7-Ty/view?usp=sharing

#### Pose2Avatar
Download and unzip in Speech-to-ASL/Pose2Avatar/checkpoints/pose2avatar/
https://drive.google.com/file/d/15RztDIdzqFu7toskBqg23JBBWc0LUJrV/view?usp=sharing

### 3. Run end to end
```
run.sh
```


## References:
1. Pytorch deepspeech  https://github.com/SeanNaren/deepspeech.pytorch
2. SpeechRecognition library https://github.com/Uberi/speech_recognition#readme
3. Pix2pix  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
4. Open Pose https://github.com/CMU-Perceptual-Computing-Lab/openpose
5. Everbody Dance Now  https://carolineec.github.io/everybody_dance_now/
6. Attention is all you need https://github.com/jadore801120/attention-is-all-you-need-pytorch
7. Speech to signs https://github.com/imatge-upc/speech2signs-2017-nmt/tree/master/ASLG-PC12