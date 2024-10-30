import os
from pydub import AudioSegment

# Path to the directory containing your audio files
input_directory = "/mnt/c/Users/Lenovo/PycharmProjects/tts/wav/wavs"
output_directory = "/mnt/c/Users/Lenovo/PycharmProjects/tts/wav/mono_wavs"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate over all files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        # Full path of the current file
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename)

        # Load the audio file
        audio = AudioSegment.from_wav(input_file_path)

        # Convert the audio to mono
        mono_audio = audio.set_channels(1)

        # Export the converted audio to the output directory
        mono_audio.export(output_file_path, format="wav")

        print(f"Converted {filename} to mono and saved to {output_file_path}")

print("Conversion complete.")
