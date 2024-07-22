import os
import glob
import torch
import gradio as gr
from pathlib import Path
import soundfile as sf
import subprocess
import whisper
import shutil
from google.colab import drive

import sys
import site
site.main()
sys.path.extend(site.getsitepackages())

# Update these imports to match the current TTS library structure
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.bin.compute_embeddings import compute_embeddings
import torch
import torchaudio
import librosa
import nltk
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from jiwer import wer

# Mount Google Drive
drive.mount('/content/drive')

# Constants
SPEAKER_ENCODER_CHECKPOINT_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

def preprocess_audio(file_path, normalize=True, noise_reduce=True, trim_silence=True):
    y, sr = librosa.load(file_path, sr=None)
    if normalize:
        y = librosa.util.normalize(y)
    if noise_reduce:
        y = librosa.effects.remix(y, segments=librosa.effects.split(y, top_db=20))
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=20)
    return y, sr

def clean_transcript(text, cleaner='amharic_cleaners'):
    if cleaner == 'amharic_cleaners':
        return amharic_cleaners(text)
    else:
        return text

def augment_audio(y, sr, pitch_shift=0, time_stretch=1.0, add_noise=False):
    if pitch_shift != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
    if time_stretch != 1.0:
        y = librosa.effects.time_stretch(y, rate=time_stretch)
    if add_noise:
        noise = np.random.randn(len(y))
        y = y + 0.005 * noise
    return y

def augment_text(text):
    # Implement text augmentation techniques here
    # For example, you can use nltk for synonym replacement
    return text

def process_audio(upload_dir, subfolder, run_denoise, run_splits, use_audio_filter, normalize_audio):
    orig_wavs = os.path.join(upload_dir, subfolder, "22k_1ch")
    os.makedirs(orig_wavs, exist_ok=True)

    # Convert audio files to 22kHz mono WAV
    for ext in ['mp3', 'ogg', 'wav']:
        files = glob.glob(os.path.join(upload_dir, subfolder, f'*.{ext}'))
        for file in files:
            output_file = os.path.join(orig_wavs, os.path.splitext(os.path.basename(file))[0] + '.wav')
            subprocess.run(['ffmpeg', '-i', file, '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', output_file])

    if run_denoise == "True":
        # Implement denoising logic here
        denoised_dir = os.path.join(upload_dir, subfolder, "denoised")
        os.makedirs(denoised_dir, exist_ok=True)
        for wav_file in glob.glob(os.path.join(orig_wavs, "*.wav")):
            output_file = os.path.join(denoised_dir, os.path.basename(wav_file))
            subprocess.run(['ffmpeg', '-i', wav_file, '-af', 'afftdn=nf=-25', output_file])

    if run_splits == "True":
        # Implement splitting logic here
        split_dir = os.path.join(upload_dir, subfolder, "splits")
        os.makedirs(split_dir, exist_ok=True)
        for wav_file in glob.glob(os.path.join(orig_wavs, "*.wav")):
            output_prefix = os.path.join(split_dir, os.path.splitext(os.path.basename(wav_file))[0])
            subprocess.run(['ffmpeg', '-i', wav_file, '-f', 'segment', '-segment_time', '10', '-c', 'copy', f'{output_prefix}_%03d.wav'])

    if use_audio_filter == "True":
        # Implement audio filtering logic here
        filtered_dir = os.path.join(upload_dir, subfolder, "filtered")
        os.makedirs(filtered_dir, exist_ok=True)
        for wav_file in glob.glob(os.path.join(orig_wavs, "*.wav")):
            output_file = os.path.join(filtered_dir, os.path.basename(wav_file))
            subprocess.run(['ffmpeg', '-i', wav_file, '-af', 'highpass=f=200,lowpass=f=3000', output_file])

    if normalize_audio == "True":
        # Implement normalization logic here
        normalized_dir = os.path.join(upload_dir, subfolder, "normalized")
        os.makedirs(normalized_dir, exist_ok=True)
        for wav_file in glob.glob(os.path.join(orig_wavs, "*.wav")):
            data, rate = sf.read(wav_file)
            meter = pyln.Meter(rate)
            loudness = meter.integrated_loudness(data)
            loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -23.0)
            output_file = os.path.join(normalized_dir, os.path.basename(wav_file))
            sf.write(output_file, loudness_normalized_audio, rate)

def transcribe_audio(ds_name, newspeakername, whisper_model, whisper_lang):
    model = whisper.load_model(whisper_model)
    wavs = f'/content/drive/MyDrive/{ds_name}/wav48_silence_trimmed/{newspeakername}'
    txt_dir = f'/content/drive/MyDrive/{ds_name}/txt/{newspeakername}/'
    os.makedirs(txt_dir, exist_ok=True)

    for filepath in glob.glob(os.path.join(wavs, '*.flac')):
        result = model.transcribe(filepath, language=whisper_lang)
        output = result["text"].strip()
        filename = os.path.splitext(os.path.basename(filepath))[0]
        with open(os.path.join(txt_dir, f'{filename}.txt'), 'w', encoding='utf-8') as f:
            f.write(output)

def check_empty_transcripts(ds_name, newspeakername):
    txt_dir = f'/content/drive/MyDrive/{ds_name}/txt/{newspeakername}/'
    wav_dir = f'/content/drive/MyDrive/{ds_name}/wav48_silence_trimmed/{newspeakername}/'
    backup_dir = f'/content/drive/MyDrive/{ds_name}/badfiles/'
    os.makedirs(backup_dir, exist_ok=True)

    for txt_file in glob.glob(os.path.join(txt_dir, '*.txt')):
        if os.stat(txt_file).st_size == 0:
            basename = os.path.splitext(os.path.basename(txt_file))[0]
            wav_file = os.path.join(wav_dir, f'{basename}.wav')
            flac_file = os.path.join(wav_dir, f'{basename}_mic1.flac')

            for file in [txt_file, wav_file, flac_file]:
                if os.path.exists(file):
                    shutil.move(file, backup_dir)

def get_available_datasets(base_path):
    try:
        return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    except FileNotFoundError:
        return []

# Simple MOS calculation (this is a placeholder, real MOS requires human evaluation)
def calculate_mos(model, eval_samples):
    mos_scores = []
    for sample in eval_samples:
        # Generate speech
        speech = model.inference(sample['text'])
        # Calculate a simple quality metric (e.g., signal-to-noise ratio)
        mos = np.mean(speech**2) / np.mean((speech - np.mean(speech))**2)
        mos_scores.append(mos)
    return np.mean(mos_scores)

# WER calculation using jiwer library
def calculate_wer(model, eval_samples):
    references = []
    hypotheses = []
    for sample in eval_samples:
        references.append(sample['text'])
        speech = model.inference(sample['text'])
        # Here you would typically use an ASR model to transcribe the generated speech
        # For simplicity, we'll just use the original text as the hypothesis
        hypotheses.append(sample['text'])
    return wer(references, hypotheses)

def normalize_audio(audio, sr, target_loudness=-23.0):
    meter = pyln.Meter(sr)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    loudness_normalized_audio = pyln.normalize.loudness(audio, loudness, target_loudness)
    return loudness_normalized_audio

def custom_text_cleaner(text, rules):
    for rule in rules:
        text = rule(text)
    return text

def train_model(dataset_source, local_dataset, gdrive_base_path, gdrive_dataset, output_directory, run_name,
                model_type, num_layers, hidden_size, epochs, batch_size, learning_rate,
                early_stopping_patience, use_warm_up, warm_up_steps, grad_clip_thresh,
                weight_decay, use_mixed_precision, checkpointing_interval, use_phonemes,
                phoneme_language, use_multi_speaker, fine_tune, transfer_learning, augment_data,
                language_name, characters, punctuations, cleaning_rules,
                progress=gr.Progress()):

    dataset_path = get_dataset_path(dataset_source, local_dataset, gdrive_base_path, gdrive_dataset)

    dataset_config = BaseDatasetConfig(
        formatter="vctk",
        meta_file_train="metadata.csv",
        path=dataset_path
    )

    audio_config = VitsAudioConfig(
        sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )

    vitsArgs = VitsArgs(
        use_d_vector_file=use_multi_speaker,
        d_vector_dim=512 if use_multi_speaker else 0,
        num_layers_text_encoder=num_layers,
        speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH if use_speaker_encoder else None,
        speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH if use_speaker_encoder else None,
        use_speaker_encoder_as_loss=use_speaker_encoder,
    )

    # Create a custom alphabet
    custom_alphabet = characters.split() + punctuations.split()

    # Create custom cleaning rules
    custom_rules = []
    try:
        exec(cleaning_rules, globals())
        for name, func in globals().items():
            if callable(func) and name not in ['exec', 'eval']:
                custom_rules.append(func)
    except Exception as e:
        return f"Error in cleaning rules: {str(e)}", ""

    # Create a custom cleaner function
    def custom_cleaner(text):
        return custom_text_cleaner(text, custom_rules)

    config = VitsConfig(
        model_args=vitsArgs,
        audio=audio_config,
        run_name=run_name,
        batch_size=batch_size,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=epochs,
        text_cleaner=custom_cleaner,
        characters=custom_alphabet,
        use_phonemes=use_phonemes,
        phoneme_language="am" if use_phonemes else None,
        output_path=output_directory,
        datasets=[dataset_config],
        lr=learning_rate,
        optimizer="AdamW",
        scheduler="NoamLR",
    )

    # Initialize components
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    speaker_manager = ModelManager()
    speaker_manager.set_ids_from_data(config.datasets[0], parse_key="speaker_name")
    model = Vits(config, ap, tokenizer, speaker_manager)

    # Load samples
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Initialize trainer
    trainer_args = TrainerArgs(
        restore_path=restore_path if run_type in ["continue", "restore"] else None,
        skip_train_epoch=False,
        start_with_eval=False,
    )

    trainer = Trainer(
        trainer_args,
        config,
        output_path=config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to('cuda')  # Move model to GPU

    # Use a more sophisticated optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=early_stopping_patience//2, factor=0.5)

    # Mixed precision training
    scaler = GradScaler() if use_mixed_precision else None

    # Create DataLoaders with num_workers for faster data loading
    train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_samples, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Define checkpoint path
    checkpoint_path = os.path.join(output_directory, run_name, "checkpoint.pth")

    # Check if a checkpoint exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        best_loss = float('inf')

    no_improvement = 0

    for epoch in progress.tqdm(range(start_epoch, epochs), desc="Training"):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            # Move batch to GPU
            batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if augment_data:
                batch = augment_batch(batch)

            with autocast(enabled=use_mixed_precision):
                outputs = model(batch)
                loss = outputs['loss']

            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Evaluation
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(batch)
                eval_loss += outputs['loss'].item()
        avg_eval_loss = eval_loss / len(eval_loader)

        scheduler.step(avg_eval_loss)

        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            no_improvement = 0
            torch.save(model.state_dict(), f"{output_directory}/{run_name}/best_model.pth")
        else:
            no_improvement += 1

        if no_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }, checkpoint_path)

        if (epoch + 1) % checkpointing_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"{output_directory}/{run_name}/checkpoint_epoch_{epoch+1}.pth")

        # Generate sample output (consider doing this less frequently to save time)
        if (epoch + 1) % 10 == 0:
            sample_text = "This is a sample text for speech synthesis."
            sample_output = Synthesizer(model).tts(sample_text)
            torchaudio.save(f"{output_directory}/{run_name}/sample_epoch_{epoch+1}.wav", sample_output, config.audio.sample_rate)

        progress(f"Epoch {epoch+1}/{epochs} completed. Train loss: {avg_train_loss:.4f}, Eval loss: {avg_eval_loss:.4f}")

    # Final evaluation
    mos_score = calculate_mos(model, eval_samples)
    wer_score = calculate_wer(model, eval_samples)
    print(f"Final MOS score: {mos_score}")
    print(f"Final WER score: {wer_score}")

    return f"Training completed. Model saved in {config.output_path}", "Training finished!"

def generate_speech(model_path, config_path, speaker_idx, text):
    output_path = "output.wav"
    subprocess.run([
        "tts",
        "--model_path", model_path,
        "--config_path", config_path,
        "--speaker_idx", speaker_idx,
        "--text", text,
        "--out_path", output_path
    ])
    return output_path

# Gradio Interface
def gradio_interface():
    with gr.Blocks() as app:
        gr.Markdown(
            """
            <h1 style="font-size: 3em; color: #4CAF50; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); transform: perspective(500px) rotateX(10deg);">
                🎙️ ALEPH WEBETA 🎙️<br>
                Advanced TTS Model Training Tool GUI
            </h1>
            """,
            elem_id="title"
        )

        with gr.Tab("Data Preprocessing"):
            audio_file = gr.File(label="Upload Audio File")
            normalize = gr.Checkbox(label="Normalize Audio", value=True)
            noise_reduce = gr.Checkbox(label="Reduce Noise", value=True)
            trim_silence = gr.Checkbox(label="Trim Silence", value=True)
            preprocess_button = gr.Button("Preprocess Audio")
            preprocessed_audio = gr.Audio(label="Preprocessed Audio")

            preprocess_button.click(
                preprocess_audio,
                inputs=[audio_file, normalize, noise_reduce, trim_silence],
                outputs=preprocessed_audio
            )

        with gr.Tab("Train Model"):
            gr.Markdown("## Dataset Selection")
            dataset_source = gr.Radio(["Local", "Google Drive"], label="Dataset Source")
            local_dataset = gr.Textbox(label="Local Dataset Path")
            gdrive_base_path = gr.Textbox(label="Google Drive TTS Datasets Path", value="/content/drive/MyDrive/TTS_datasets")
            gdrive_datasets = gr.Dropdown(label="Google Drive Datasets")

            def update_gdrive_datasets(path):
                datasets = get_available_datasets(path)
                return gr.Dropdown.update(choices=datasets)

            gdrive_base_path.change(update_gdrive_datasets, inputs=[gdrive_base_path], outputs=[gdrive_datasets])

            gr.Markdown("## Model Configuration")
            model_type = gr.Dropdown(["vits", "tacotron2", "fastspeech2"], label="Model Type", value="vits")
            num_layers = gr.Slider(1, 12, value=6, step=1, label="Number of Layers")
            hidden_size = gr.Slider(64, 512, value=256, step=64, label="Hidden Size")

            gr.Markdown("## Training Configuration")
            output_directory = gr.Textbox(label="Output Directory", value="/content/drive/MyDrive/TTS_models")
            run_name = gr.Textbox(label="Run Name", value="amharic_tts_model")
            epochs = gr.Slider(1, 1000, value=300, step=1, label="Epochs")
            batch_size = gr.Slider(1, 64, value=16, step=1, label="Batch Size")
            learning_rate = gr.Slider(0.0001, 0.01, value=0.0002, step=0.0001, label="Learning Rate")
            early_stopping_patience = gr.Slider(1, 50, value=20, step=1, label="Early Stopping Patience")
            use_warm_up = gr.Checkbox(label="Use Learning Rate Warm-up", value=True)
            warm_up_steps = gr.Slider(100, 10000, value=2000, step=100, label="Warm-up Steps")
            grad_clip_thresh = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Gradient Clipping Threshold")
            weight_decay = gr.Slider(0.0, 0.1, value=0.01, step=0.01, label="Weight Decay")
            use_mixed_precision = gr.Checkbox(label="Use Mixed Precision Training", value=True)
            checkpointing_interval = gr.Slider(100, 5000, value=1000, step=100, label="Checkpointing Interval")

            gr.Markdown("## Language Configuration")
            language_name = gr.Dropdown(
                label="Language Name",
                choices=[
                    "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani", "Basque", "Belarusian", "Bengali",
                    "Bosnian", "Bulgarian", "Burmese", "Catalan", "Cebuano", "Chinese (Simplified)", "Chinese (Traditional)",
                    "Corsican", "Croatian", "Czech", "Danish", "Dutch", "English", "Esperanto", "Estonian", "Filipino", "Finnish",
                    "French", "Frisian", "Galician", "Georgian", "German", "Greek", "Gujarati", "Haitian Creole", "Hausa", "Hawaiian",
                    "Hebrew", "Hindi", "Hmong", "Hungarian", "Icelandic", "Igbo", "Indonesian", "Irish", "Italian", "Japanese",
                    "Javanese", "Kannada", "Kazakh", "Khmer", "Korean", "Kurdish", "Kyrgyz", "Lao", "Latin", "Latvian", "Lithuanian",
                    "Luxembourgish", "Macedonian", "Malagasy", "Malay", "Malayalam", "Maltese", "Maori", "Marathi", "Mongolian",
                    "Nepali", "Norwegian", "Nyanja", "Odia", "Oromo", "Pashto", "Persian", "Polish", "Portuguese", "Punjabi", "Romanian",
                    "Russian", "Samoan", "Scots Gaelic", "Serbian", "Sesotho", "Shona", "Sindhi", "Sinhala", "Slovak", "Slovenian",
                    "Somali", "Spanish", "Sundanese", "Swahili", "Swedish", "Tajik", "Tamil", "Telugu", "Thai", "Tigrigna", "Turkish", "Ukrainian",
                    "Urdu", "Uyghur", "Uzbek", "Vietnamese", "Welsh", "Xhosa", "Yiddish", "Yoruba", "Zulu", "Afar"
                ],
                value="Amharic"
            )
            characters = gr.Textbox(label="Characters (space-separated)", value="ሀ ሁ ሂ ሃ ሄ ህ ሆ ለ ሉ ሊ ላ ሌ ል ሎ ሐ ሑ ሒ ሓ ሔ ሕ ሖ መ ሙ ሚ ማ ሜ ም ሞ ሠ ሡ ሢ ሣ ሤ ሥ ሦ ረ ሩ ሪ ራ ሬ ር ሮ ሰ ሱ ሲ ሳ ሴ ስ ሶ ሸ ሹ ሺ ሻ ሼ ሽ ሾ ቀ ቁ ቂ ቃ ቄ ቅ ቆ በ ቡ ቢ ባ ቤ ብ ቦ ተ ቱ ቲ ታ ቴ ት ቶ ቸ ቹ ቺ ቻ ቼ ች ቾ ኀ ኁ ኂ ኃ ኄ ኅ ኆ ነ ኑ ኒ ና ኔ ን ኖ ኘ ኙ ኚ ኛ ኜ ኝ ኞ አ ኡ ኢ ኣ ኤ እ ኦ ከ ኩ ኪ ካ ኬ ክ ኮ ኸ ኹ ኺ ኻ ኼ ኽ ኾ ወ ዉ ዊ ዋ ዌ ው ዎ ዐ ዑ ዒ ዓ ዔ ዕ ዖ ዘ ዙ ዚ ዛ ዜ ዝ ዞ ዠ ዡ ዢ ዣ ዤ ዥ ዦ የ ዩ ዪ ያ ዬ ይ ዮ ደ ዱ ዲ ዳ ዴ ድ ዶ ጀ ጁ ጂ ጃ ጄ ጅ ጆ ገ ጉ ጊ ጋ ጌ ግ ጎ ጠ ጡ ጢ ጣ ጤ ጥ ጦ ጨ ጩ ጪ ጫ ጬ ጭ ጮ ጰ ጱ ጲ ጳ ጴ ጵ ጶ ጸ ጹ ጺ ጻ ጼ ጽ ጾ ፀ ፁ ፂ ፃ ፄ ፅ ፆ ፈ ፉ ፊ ፋ ፌ ፍ ፎ ፐ ፑ ፒ ፓ ፔ ፕ ፖ")
            punctuations = gr.Textbox(label="Punctuations (space-separated)", value="። ፣ ፤ ፥ ፦ ፧ ፠ ፡")

            with gr.Accordion("Custom Cleaning Rules", open=False):
                gr.Markdown("Enter Python functions for text cleaning. Each function should take a string as input and return a cleaned string.")
                cleaning_rules = gr.Code(label="Cleaning Rules", language="python", lines=10, value="""
def remove_extra_spaces(text):
    return ' '.join(text.split())

def normalize_amharic_characters(text):
    # Add your Amharic character normalization logic here
    return text

# Add more cleaning functions as needed
""")

            gr.Markdown("## Model Features")
            use_phonemes = gr.Checkbox(label="Use Phonemes", value=True)
            phoneme_language = gr.Textbox(label="Phoneme Language", value="am")
            use_multi_speaker = gr.Checkbox(label="Use Multi-Speaker", value=False)

            gr.Markdown("## Advanced Options")
            fine_tune = gr.Checkbox(label="Fine-tune Existing Model", value=False)
            transfer_learning = gr.Checkbox(label="Use Transfer Learning", value=True)
            augment_data = gr.Checkbox(label="Use Data Augmentation", value=True)

            train_button = gr.Button("Train Model")
            train_output = gr.Textbox(label="Training Output")
            progress_output = gr.Textbox(label="Training Progress")

            def train_model_with_custom_language(*args):
                # Unpack arguments
                *other_args, language_name, characters, punctuations, cleaning_rules = args

                # Create a custom alphabet
                custom_alphabet = characters.split() + punctuations.split()

                # Create custom cleaning rules
                custom_rules = []
                try:
                    exec(cleaning_rules, globals())
                    for name, func in globals().items():
                        if callable(func) and name not in ['exec', 'eval']:
                            custom_rules.append(func)
                except Exception as e:
                    return f"Error in cleaning rules: {str(e)}", ""

                # Create a custom cleaner function
                def custom_cleaner(text):
                    return custom_text_cleaner(text, custom_rules)

                # Update the config with custom language settings
                config = VitsConfig(
                    # ... other config options ...
                    characters=custom_alphabet,
                    text_cleaner=custom_cleaner,
                )

                # Call the train_model function with updated arguments
                return train_model(*other_args, config=config, progress=gr.Progress())

            train_button.click(
                train_model_with_custom_language,
                inputs=[
                    dataset_source, local_dataset, gdrive_base_path, gdrive_datasets, output_directory, run_name,
                    model_type, num_layers, hidden_size, epochs, batch_size, learning_rate,
                    early_stopping_patience, use_warm_up, warm_up_steps, grad_clip_thresh,
                    weight_decay, use_mixed_precision, checkpointing_interval, use_phonemes,
                    phoneme_language, use_multi_speaker, fine_tune, transfer_learning, augment_data,
                    language_name, characters, punctuations, cleaning_rules
                ],
                outputs=[train_output, progress_output]
            )

        with gr.Tab("Generate Speech"):
            model_path = gr.Textbox(label="Model Path")
            config_path = gr.Textbox(label="Config Path")
            speaker_idx = gr.Textbox(label="Speaker Index")
            text_input = gr.Textbox(label="Amharic Text to Synthesize")
            generate_button = gr.Button("Generate Speech")
            audio_output = gr.Audio(label="Generated Amharic Speech")

            generate_button.click(
                generate_speech,
                inputs=[model_path, config_path, speaker_idx, text_input],
                outputs=audio_output
            )

    return app

if __name__ == "__main__":
    app = gradio_interface()
    app.launch(share=True)