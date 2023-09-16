from gtts import gTTS
import pygame
import os

class TextToSpeech:
    def __init__(self):
        pygame.mixer.init()

    def convert_and_play(self, text):
        # Convert text to speech
        tts = gTTS(text)

        # Save the speech to a temporary file
        tts.save("output.mp3")

        # Load the saved speech file
        pygame.mixer.music.load("output.mp3")

        # Play the speech
        pygame.mixer.music.play()

        # Wait for the speech to finish playing
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(10)

        # Clean up temporary file
        os.remove("output.mp3")

if __name__ == "__main__":
    text_to_speech = TextToSpeech()
    input_text = input("Enter the text to speak: ")
    text_to_speech.convert_and_play(input_text)
