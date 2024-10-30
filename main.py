import os
import torch
import torchaudio
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "my_speaker"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = f"/mnt/c/Users/Lenovo/PycharmProjects/tts/wav/wavsss/{cols[0]}.wav"
            text = cols[1]


            waveform, sample_rate = torchaudio.load(wav_file)


            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)


            items.append(
                {"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path}
            )

        return items


if __name__ == '__main__':
    torch.cuda.empty_cache()


    output_path = "/mnt/c/Users/Lenovo/PycharmProjects/tts"


    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="transcripttt.csv", path=os.path.join(output_path, "wav/")
    )

    # Audio configuration
    audio_config = VitsAudioConfig(
        sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None, fft_size=1024
    )


    character_config = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890éàè",
        punctuations=" !,.?-:;'",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
    )

    # Model configuration
    config = VitsConfig(
        audio=audio_config,
        characters=character_config,
        run_name="vits_romanized_arabic",
        batch_size=4,
        eval_batch_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=0,
        epochs=500,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=False,
        save_best_after=1000,
        save_checkpoints=True,
        save_all_best=True,
        mixed_precision=False,
        max_text_len=200,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        test_sentences=[
            ["ahla winek"],
            ["mchit lel 9ahwa lyoum?"],
            ["ba3d saret haja okhra f masirti eni teghramet b 9ira2et l kotob tahdidan kotob l business"]
        ]
    )

    # Initialize the audio processor
    ap = AudioProcessor.init_from_config(config)

    # Initialize the tokenizer with the new Romanized character set
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Load the dataset
    train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

    # Initialize the model
    model = Vits(config, ap, tokenizer, speaker_manager=None)


    model = model.to(device)

    # Initialize the trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Start training
    trainer.fit()
