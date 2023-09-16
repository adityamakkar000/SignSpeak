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

        # Convert text to speech
        tts = gTTS(text)

        # Create an in-memory byte buffer to hold the audio data
        audio_buffer = io.BytesIO()

        # Save the speech to the in-memory buffer instead of a file
        tts.write_to_fp(audio_buffer)

        # Seek to the beginning of the buffer
        audio_buffer.seek(0)

        # Load the audio data from the buffer
        pygame.mixer.music.load(audio_buffer)

        # Play the speech
        pygame.mixer.music.play()

        # Wait for the speech to finish playing
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(10)

if __name__ == "__main__":
    text_to_speech = TextToSpeech()
    input_text = input("Enter the text to speak: ")
    text_to_speech.convert_and_play(input_text)
