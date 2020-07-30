# speech2text
python Speech2English/convert_speech_to_english.py --audio data/example_spoken_english.wav --text data/english_text.txt
# english2gloss
python English2Gloss/translate.py -data_pkl English2Gloss/eng2gloss_data.pkl -model English2Gloss/trained.chkpt -input data/english_text.txt -output data/asl_gloss_text.txt
# gloss to individual poses

# pose interpolation

# data prep for pose2avatar pix2pix model

# pose2avatar frames
python Pose2Avatar/test.py --dataroot /path/to/dir/containing/pose/images/ --name pose2avatar --model test --netG unet_256 --direction AtoB --dataset_mode single --norm batch
# avatar frames to avatar video