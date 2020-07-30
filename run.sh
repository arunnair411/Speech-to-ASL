# speech2text
python Speech2English/convert_speech_to_english.py --audio data/example_spoken_english.wav --text data/english_text.txt
# english2gloss
python English2Gloss/translate.py -data_pkl English2Gloss/eng2gloss_data.pkl -model English2Gloss/trained.chkpt -input data/english_text.txt -output data/asl_gloss_text.txt
# align glosses to poses

# gloss to individual poses
python Gloss2Avatar/gloss_lookup.py --glosses='car,point-to,drive'
# pose interpolation

# data prep for pose2avatar pix2pix model
python Gloss2Avatar/create_test_data.py --output-dir=/path/to/dir/containing/pose/images/ --file-ids='207,15602,15316' --pose-data=car_she_drives.posesequence
# pose2avatar frames
python Pose2Avatar/test.py --dataroot /path/to/dir/containing/pose/images/ --name pose2avatar --model test --netG unet_256 --direction AtoB --dataset_mode single --norm batch
# avatar frames to avatar video
python Gloss2Avatar/generate_video.py --results-dir /path/to/dir/containing/network/outputs
