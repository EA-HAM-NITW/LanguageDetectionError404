import pyaudio, wave
# from pygame import mixer
from scipy.io.wavfile import write
from src.settings import DURATION, DEFAULT_SAMPLE_RATE, MAX_INPUT_CHANNELS, \
                                    WAVE_OUTPUT_FILE, INPUT_DEVICE, CHUNK_SIZE
# import sounddevice as sd
# import soundfile as sf


class Sound(object):
    def __init__(self):
        # Set default configurations for recording device
        # sd.default.samplerate = DEFAULT_SAMPLE_RATE
        # sd.default.channels = DEFAULT_CHANNELS
        self.format = pyaudio.paInt16
        self.channels = MAX_INPUT_CHANNELS
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.chunk = CHUNK_SIZE
        self.duration = DURATION
        self.path = WAVE_OUTPUT_FILE
        self.device = INPUT_DEVICE
        self.frames = []
        self.audio = pyaudio.PyAudio()

    def record(self):
        # start Recording
        self.audio = pyaudio.PyAudio()
        stream = self.audio.open(
                        format=self.format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk,
                        input_device_index=self.device)
        self.frames = []
        for i in range(0, int(self.sample_rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            self.frames.append(data)
        # stop Recording
        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        self.save()

    def save(self):
        waveFile = wave.open(self.path, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.sample_rate)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()

    # def play(self):
    #     logger.info(f"Playing the recorded sound {self.path}")
    #     mixer.init(self.sample_rate)
    #     recording = mixer.Sound(self.path).play()
    #     while recording.get_busy():
    #         continue

sound = Sound()

    # def device_info(kind='input'):
    #     keys = ['name', 'index', 'maxInputChannels', 'defaultSampleRate']
    #     logger.info(f"{kind} device info:")
    #     info_dict = sd.query_devices(kind=kind)
    #     logger.info(([(key, value) for key, value in info_dict.items() if key in keys]))
    #     print()

    # def record(self):
    #     logger.info(f"Recording started for {DURATION} seconds")
    #     self.recording = sd.rec(int(DURATION * DEFAULT_SAMPLE_RATE), \
    #                                             samplerate=DEFAULT_SAMPLE_RATE,
    #                                             channels=DEFAULT_CHANNELS)
    #     sd.wait()
    #     logger.info("Recording completed")
    #     self.save()

    # def play(self):
    #     logger.info(f"Playing the recorded sound {WAVE_OUTPUT_FILE}")
    #     data, fs = sf.read(WAVE_OUTPUT_FILE, dtype='float32')
    #     sd.play(data, fs, device=OUTPUT_DEVICE)
    #     sd.wait()
        # sd.play(self.recording)
        # sd.wait()

    # def save(self):
    #     data = self.recording
    #     scaled = np.int16(data/np.max(np.abs(data)) * 32767)
    #     write(WAVE_OUTPUT_FILE, DEFAULT_SAMPLE_RATE, scaled)
    #     logger.info(f"Recording saved to {WAVE_OUTPUT_FILE}")