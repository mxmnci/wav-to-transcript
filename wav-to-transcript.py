import speech_recognition as sr
from pydub import AudioSegment
import multiprocessing
import argparse
from tqdm import tqdm
import tempfile
import os

def format_timestamp(ms):
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def transcribe_segment(segment_path, start_time, end_time):
    recognizer = sr.Recognizer()
    with sr.AudioFile(segment_path) as source:
        audio_data = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio_data)
        return f"{start_time}-{end_time}: {transcription}"
    except sr.UnknownValueError:
        return f"{start_time}-{end_time}: Speech recognition could not understand the audio"
    except sr.RequestError as e:
        return f"{start_time}-{end_time}: Could not request results from Google Speech Recognition service; {e}"

def process_segment(args):
    (audio, start, end, i, temp_folder) = args
    segment = audio[start:end]
    segment_name = os.path.join(temp_folder, f"segment{i}.wav")
    segment.export(segment_name, format="wav")
    start_time = format_timestamp(start)
    end_time = format_timestamp(end)
    return transcribe_segment(segment_name, start_time, end_time)

def split_and_transcribe(audio_path, segment_duration_ms=60000, num_processes=2):
    print("Loading audio file...")
    audio = AudioSegment.from_file(audio_path)

    segment_count = len(audio) // segment_duration_ms + 1
    print(f"Total segments to process: {segment_count}")

    # Create a temporary directory for audio segments
    with tempfile.TemporaryDirectory() as temp_folder:
        transcriptions = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            segment_args = [
                (audio, i * segment_duration_ms, min((i + 1) * segment_duration_ms, len(audio)), i, temp_folder)
                for i in range(segment_count)
            ]
            for transcription in tqdm(pool.imap(process_segment, segment_args), total=segment_count, desc="Processing Segments"):
                transcriptions.append(transcription)
        
        # Writing transcriptions to a text file
        with open('transcriptions.txt', 'w', encoding='utf-8') as file:
            for transcription in transcriptions:
                file.write(f"{transcription}\n\n")
        print("Transcription complete. Results saved to transcriptions.txt")

def main():
    parser = argparse.ArgumentParser(description='Transcribe an audio file.')
    parser.add_argument('audio_path', type=str, help='Path to the audio file')
    args = parser.parse_args()

    split_and_transcribe(args.audio_path)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
