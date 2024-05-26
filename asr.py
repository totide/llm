import os
import numpy as np
import pyaudio
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_name = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype)
model.to(device)


def record_audio(duration, samplerate, channels=1, chunk=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplerate,
                    input=True,
                    frames_per_buffer=chunk)
    print("开始录音...")
    frames = []
    for _ in range(0, int(samplerate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("录音结束")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return np.frombuffer(b''.join(frames), dtype=np.int16)


def asr_by_file(file):
    # 加载音频文件
    audio_input, sample_rate = torchaudio.load(file)

    # 重采样到16kHz，如果需要的话
    def_sample_rate = 16000
    if sample_rate != def_sample_rate:
        audio_input = torchaudio.transforms.Resample(sample_rate, def_sample_rate)(audio_input)

    if audio_input.shape[0] == 2:
        audio_input = audio_input.mean(dim=0, keepdim=True)
    if audio_input.ndim == 1:
        audio_input = audio_input.unsqueeze(0)

    audio_input = audio_input[0]
    inputs = processor(audio_input, sampling_rate=def_sample_rate, return_tensors="pt")
    predicted_ids = model.generate(**inputs)

    # 将输出ID转换为文本
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(transcription)


def asr_realtime():
    duration = 5  # 录音持续时间（秒）
    sample_rate = 16000  # 采样率

    audio_input = record_audio(duration, sample_rate)
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
    predicted_ids = model.generate(**inputs)

    # 将输出ID转换为文本
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(transcription)


def asr_by_pipeline():
    from datasets import load_dataset

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample = dataset[0]["audio"]
    result = pipe(sample)
    print(result["text"])


if __name__ == '__main__':
    # asr_by_file("static/sample2.flac")
    asr_realtime()
    # asr_by_pipeline()

