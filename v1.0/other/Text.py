from gtts import gTTS
import pygame
import io

class TextToSpeech:
    def __init__(self):
        pygame.mixer.init()

    def convert_and_play(self, text):
        if not text:
            print("No text to speak.")
            returngit

        tts = gTTS(text)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(10)

if __name__ == "__main__":
    text_to_speech = TextToSpeech()
    input_text = input("Enter the text to speak: ")
    text_to_speech.convert_and_play(input_text)
