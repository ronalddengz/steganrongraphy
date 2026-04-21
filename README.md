# Image and Audio Steganography
A Python-based implementation of steganography techniques for both image and audio files, made for CSCI 10 at Williams College.

## Installation Instructions
`pip install -r requirements.txt`

## Usage

The program uses three subcommands: `encode`, `decode`, and `spectrogram`. 

### Encode/decode text in an image
```bash
# Encode (inline text)
python3 steganrongraphy.py encode image.png secret.png -t "hidden message"

# Encode (text from file)
python3 steganrongraphy.py encode image.png secret.png -t message.txt

# Decode
python3 steganrongraphy.py decode secret.png decoded.txt
```
Feel free to use one of the ten example .pngs I have graciously included in `images`.

### Encode/decode text in audio
```bash
# Encode
python3 steganrongraphy.py encode audio.wav secret.wav -t "hidden message"

# Decode
python3 steganrongraphy.py decode secret.wav decoded.txt
```

### Embed an image into an audio spectrogram
```bash
# Encode (image → audio)
python3 steganrongraphy.py encode image.png output.wav

# View the spectrogram
python3 steganrongraphy.py spectrogram output.wav spectrogram.png
```
I called this "encoding" and "decoding", but really it is just putting the image into the audio file's spectrogram.

## Dependencies
- Pillow: Image processing library
- NumPy: Numerical computing library
- Librosa: Audio processing library
- SoundFile: Audio file I/O
- Matplotlib: Data visualization
