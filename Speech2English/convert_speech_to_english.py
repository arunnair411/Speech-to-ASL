import argparse
import speech_recognition as sr


def convert_speech_to_english(audio_fp, text_fp):
    # Initialize recognizer class (for recognizing the speech)
    r = sr.Recognizer()

    # Reading Audio file as source
    # listening the audio file and store in audio_text variable
    with sr.AudioFile(audio_fp) as source:
        audio_text = r.listen(source)
        
    # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
        try:
            # using google speech recognition
            text = r.recognize_google(audio_text)
        except:
            text = ''
    
    with open(text_fp, 'w') as f:
        f.write(text)
    return

if __name__ == "__main__":
    '''
    python3 convert_speech_to_english.py --audio example_spoken_english.wav --text speech2english_output.txt
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--text', type=str, default='speech2english_output.txt')
    args = parser.parse_args()

    convert_speech_to_english(args.audio, args.text)