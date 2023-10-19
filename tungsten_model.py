import os
from typing import List, Optional

from scipy.io.wavfile import write as write_wav
from tungstenkit import Audio, BaseIO, Field, Option, define_model

os.environ["TORCH_HOME"] = "checkpoints/torch"

from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from utils.prompt_making import make_prompt


class Input(BaseIO):
    prompt: str = Field(
        description="Text prompt. Supported languages: English, Chinese, Japanese"
    )
    audio_with_target_voice: Audio = Field(
        description="Audio file of Speech of 3~10 seconds long with the target voice. "
        "Supported languages: English, Chinese, Japanese"
    )
    transcript: Optional[str] = Option(
        None, description="Transcript of the input audio file"
    )


class Output(BaseIO):
    generated_speech: Audio


@define_model(
    input=Input,
    output=Output,
    system_packages=["ffmpeg"],
    python_packages=[
        "soundfile",
        "numpy",
        "torch==2.0.1",
        "torchvision",
        "torchaudio",
        "tokenizers",
        "encodec",
        "langid",
        "wget",
        "unidecode",
        "pyopenjtalk-prebuilt",
        "pypinyin",
        "inflect",
        "cn2an",
        "jieba",
        "eng_to_ipa",
        "openai-whisper",
        "matplotlib",
        "gradio",
        "nltk",
        "sudachipy",
        "sudachidict_core",
        "vocos",
    ],
    gpu=True,
    gpu_mem_gb=10,
)
class VALL_E_X:
    def setup(self):
        preload_models()

    def predict(self, inputs: List[Input]) -> List[Output]:
        input = inputs[0]

        prompt_path = os.path.join("./customs", "user.npz")
        src_audio_path = os.path.join("./prompts", "user.wav")
        output_path = os.path.join("./output.wav")

        if os.path.exists(output_path):
            os.remove(output_path)

        try:
            make_prompt(
                name="user",
                audio_prompt_path=input.audio_with_target_voice.path,
                transcript=input.transcript if input.transcript else None,
            )
            audio_array = generate_audio(input.prompt, prompt="user")
            write_wav(output_path, SAMPLE_RATE, audio_array)
        finally:
            if os.path.exists(prompt_path):
                os.remove(prompt_path)
            if os.path.exists(src_audio_path):
                os.remove(src_audio_path)

        return [Output(generated_speech=Audio.from_path(output_path))]
