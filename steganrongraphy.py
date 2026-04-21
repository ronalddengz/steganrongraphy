import argparse
from PIL import Image
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import wave
import struct
import os

def encode_audio(audio_path, message, output_path, delimiter="$END$"):
    message += delimiter
    binary_message = ''.join(text_to_bin(message)) 
    message_length = len(binary_message)
    
    with wave.open(audio_path, 'rb') as audio_file:
        n_channels = audio_file.getnchannels()
        sample_width = audio_file.getsampwidth()
        n_frames = audio_file.getnframes()
        framerate = audio_file.getframerate()
        audio_data = audio_file.readframes(n_frames)
        
    # convert audio data to list of samples
    fmt = '<' + str(len(audio_data)//sample_width) + 'h'
    samples = list(struct.unpack(fmt, audio_data))
    
    if len(samples) < message_length:
        raise ValueError("Audio file too short to encode message")

    # embed message bits with minimal modification
    for i in range(message_length):
        bit = int(binary_message[i])
        if bit == 0:
            samples[i] = samples[i] & ~1
        else:
            samples[i] = samples[i] | 1
    
    modified_audio = struct.pack(fmt, *samples)
    
    # save modified audio
    with wave.open(output_path, 'wb') as audio_file:
        audio_file.setnchannels(n_channels)
        audio_file.setsampwidth(sample_width)
        audio_file.setframerate(framerate)
        audio_file.writeframes(modified_audio)

def decode_audio(audio_path, delimiter="$END$"):
    with wave.open(audio_path, 'rb') as audio_file:
        sample_width = audio_file.getsampwidth()
        n_frames = audio_file.getnframes()
        audio_data = audio_file.readframes(n_frames)
    
    fmt = '<' + str(len(audio_data)//sample_width) + 'h'
    samples = struct.unpack(fmt, audio_data)
    
    # extract LSBs
    bits = ''.join(str(sample & 1) for sample in samples)
    
    # convert bits to text
    message = ''
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            char = chr(int(byte, 2))
            message += char
            if delimiter in message:
                return message[:-len(delimiter)]
    
    raise ValueError("No hidden message found")

def create_spectrogram(audio_array, sr, output_path):
    # generate and save a spectrogram of the audio array using linear frequency scale
    D = librosa.stft(audio_array)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear')
    plt.title('Linear-frequency Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def prepare_image(image_path, target_height=512):
    # convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # resize target height with aspect ratio
    aspect_ratio = img.size[0] / img.size[1]
    target_width = int(target_height * aspect_ratio)
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # convert to numpy array
    img_array = np.array(img)
    img_array = np.flipud(img_array)
    
    # ensure proper shape for STFT (short-time Fourier transform)
    if img_array.shape[1] % 2 != 0:
        img_array = img_array[:, :-1]
    
    return img_array

def image_to_audio(image_array, sr=22050):
    # normalize image to [0, 1] range
    spec = image_array.astype(np.float32) / 255.0
    
    D = spec * np.exp(2j * np.pi * np.random.random(spec.shape))
    
    # generate audio using Griffin-Lim
    y = librosa.griffinlim(D, n_iter=64, hop_length=512)
    
    return y

def encode_image_to_audio(image_path, output_path):
    img_array = prepare_image(image_path)
    sr = 22050 
    audio = image_to_audio(img_array, sr)
    sf.write(output_path, audio, sr)
    create_spectrogram(audio, sr, f"{output_path}_spectrogram.png")
    return audio

def analyze_audio(audio_path, output_path):
    # generates spectrogram
    y, sr = librosa.load(audio_path, sr=None)
    create_spectrogram(y, sr, output_path)

# converts text message into binary string
def text_to_bin(data):
    # Updated to match genData functionality
    datalist = []
    for char in data:
        datalist.append(format(ord(char), '08b'))
    return datalist

def embed_data_in_pixels(pix, data):
    # Matching modPix functionality
    datalist = text_to_bin(data)
    lendata = len(datalist)
    imdata = iter(pix)
    
    # Check if image has enough pixels
    required_pixels = lendata * 3  # 3 pixels per character
    total_pixels = sum(1 for _ in pix)
    if total_pixels < required_pixels:
        raise ValueError(f"Image too small to encode message. Need {required_pixels} pixels, have {total_pixels}")

    try:
        for i in range(lendata):
            # Extract 3 pixels at a time
            pix = [value for value in imdata.__next__()[:3] +
                                    imdata.__next__()[:3] +
                                    imdata.__next__()[:3]]
            
            # Modify pixels based on binary data
            for j in range(8):
                if datalist[i][j] == '0' and pix[j] % 2 != 0:
                    pix[j] -= 1
                elif datalist[i][j] == '1' and pix[j] % 2 == 0:
                    if pix[j] != 0:
                        pix[j] -= 1
                    else:
                        pix[j] += 1

            # Set the end marker
            if (i == lendata - 1):
                if (pix[-1] % 2 == 0):
                    if pix[-1] != 0:
                        pix[-1] -= 1
                    else:
                        pix[-1] += 1
            else:
                if (pix[-1] % 2 != 0):
                    pix[-1] -= 1

            pix = tuple(pix)
            yield pix[0:3]
            yield pix[3:6]
            yield pix[6:9]

    except Exception as e:
        raise ValueError(f"Error during pixel embedding: {str(e)}")

def encode_image(newimg, data):
    # matching encode_enc functionality
    if not data:
        raise ValueError("Data is empty")
        
    w = newimg.size[0]
    (x, y) = (0, 0)
    
    try:
        for pixel in embed_data_in_pixels(newimg.getdata(), data):
            newimg.putpixel((x, y), pixel)
            if (x == w - 1):
                x = 0
                y += 1
            else:
                x += 1
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

def decode_image(input_path):
    # matching decode functionality
    try:
        image = Image.open(input_path, 'r')
        # convert to RGB mode if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        data = ''
        imgdata = iter(image.getdata())
    
        while True:
            try:
                pixels = [value for value in imgdata.__next__()[:3] +
                                        imgdata.__next__()[:3] +
                                        imgdata.__next__()[:3]]
        
                binstr = ''
                for i in pixels[:8]:
                    binstr += '0' if (i % 2 == 0) else '1'
        
                data += chr(int(binstr, 2))
                if (pixels[-1] % 2 != 0):
                    return data
                    
            except StopIteration:
                raise Exception("End marker not found - message might be corrupted")
                
    except Exception as e:
        raise Exception(f"Error decoding image: {str(e)}")

IMAGE_EXTS = {'jpg', 'jpeg', 'png', 'bmp'}
AUDIO_EXTS = {'wav', 'mp3'}

def get_ext(path):
    return path.rsplit('.', 1)[-1].lower()

def main():
    parser = argparse.ArgumentParser(
        description='Steganography Tool — hide text in images/audio, or embed images into audio spectrograms.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Hide text in an image
  %(prog)s encode image.png secret.png -t "hidden message"
  %(prog)s encode image.png secret.png -t message.txt

  # Recover text from an image
  %(prog)s decode secret.png output.txt

  # Hide text in audio
  %(prog)s encode audio.wav secret.wav -t "hidden message"

  # Recover text from audio
  %(prog)s decode secret.wav output.txt

  # Embed an image into an audio spectrogram
  %(prog)s encode image.png output.wav

  # Generate a spectrogram from audio
  %(prog)s spectrogram audio.wav spectrogram.png
""")
    
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- encode ---
    encode_parser = subparsers.add_parser('encode', help='Encode data into a file')
    encode_parser.add_argument('input', help='Path to input file (image or audio)')
    encode_parser.add_argument('output', help='Path to output file')
    encode_parser.add_argument('-t', '--text',
                               help='Text message or path to a .txt file (required for text steganography)')

    # --- decode ---
    decode_parser = subparsers.add_parser('decode', help='Decode hidden data from a file')
    decode_parser.add_argument('input', help='Path to encoded file (image or audio)')
    decode_parser.add_argument('output', help='Path to save decoded text')

    # --- spectrogram ---
    spec_parser = subparsers.add_parser('spectrogram', help='Generate a spectrogram from an audio file')
    spec_parser.add_argument('input', help='Path to audio file')
    spec_parser.add_argument('output', help='Path to save spectrogram image')

    args = parser.parse_args()

    try:
        if args.command == 'spectrogram':
            in_ext = get_ext(args.input)
            out_ext = get_ext(args.output)
            if in_ext not in AUDIO_EXTS:
                raise ValueError(f"Spectrogram input must be an audio file ({', '.join(AUDIO_EXTS)})")
            if out_ext not in IMAGE_EXTS:
                raise ValueError(f"Spectrogram output must be an image file ({', '.join(IMAGE_EXTS)})")
            y, sr = librosa.load(args.input, sr=None)
            create_spectrogram(y, sr, args.output)
            print(f"Spectrogram created and saved to: {args.output}")

        elif args.command == 'decode':
            in_ext = get_ext(args.input)
            if in_ext in AUDIO_EXTS:
                message = decode_audio(args.input)
                with open(args.output, 'w') as f:
                    f.write(message)
                print(f"Message extracted from audio. Saved to: {args.output}")
            elif in_ext in IMAGE_EXTS:
                decoded_text = decode_image(args.input)
                with open(args.output, 'w') as f:
                    f.write(decoded_text)
                print(f"Text decoded from image. Saved to: {args.output}")
            else:
                raise ValueError(f"Unsupported input format '.{in_ext}'. Use an image ({', '.join(IMAGE_EXTS)}) or audio ({', '.join(AUDIO_EXTS)}) file.")

        elif args.command == 'encode':
            in_ext = get_ext(args.input)
            out_ext = get_ext(args.output)

            # image → audio  (spectrogram embedding, no --text needed)
            if in_ext in IMAGE_EXTS and out_ext in AUDIO_EXTS:
                encode_image_to_audio(args.input, args.output)
                print(f"Image encoded into audio file. Saved to: {args.output}")

            # audio → audio  (text in audio)
            elif in_ext in AUDIO_EXTS and out_ext in AUDIO_EXTS:
                if not args.text:
                    raise ValueError("--text / -t is required when encoding text into audio")
                message = args.text
                if os.path.isfile(message):
                    with open(message, 'r') as f:
                        message = f.read()
                encode_audio(args.input, message, args.output)
                print(f"Message hidden in audio. Saved to: {args.output}")

            # image → image  (text in image)
            elif in_ext in IMAGE_EXTS and out_ext in IMAGE_EXTS:
                if not args.text:
                    raise ValueError("--text / -t is required when encoding text into an image")
                message = args.text
                if os.path.isfile(message):
                    with open(message, 'r') as f:
                        message = f.read()
                image = Image.open(args.input, 'r')
                newimg = image.copy()
                encode_image(newimg, message)
                newimg.save(args.output)
                print(f"Text encoded in image. Saved to: {args.output}")

            else:
                raise ValueError(
                    f"Unsupported encode combination: .{in_ext} → .{out_ext}. "
                    "Supported: image→image, audio→audio (text), image→audio (spectrogram)."
                )

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
