# ALEPH WEBETA: Advanced TTS Model Training Tool GUI

![ALEPH WEBETA Logo](path/to/logo.png)

## Overview

ALEPH WEBETA is an Advanced Text-to-Speech (TTS) Model Training Tool with a Graphical User Interface (GUI). This project aims to simplify the process of training custom TTS models, particularly for low-resource languages like Amharic.

Developed by Robel Adugna, this tool provides a user-friendly interface for data preprocessing, model training, and speech generation using state-of-the-art TTS techniques.

## Features

- **Data Preprocessing**: Easily preprocess audio files with options for normalization, noise reduction, and silence trimming.
- **Model Training**: Train custom TTS models with configurable parameters, including model type, layers, batch size, learning rate, etc.
- **Multi-language Support**: Adaptable to various languages, with a focus on Amharic and other low-resource languages.
- **Custom Alphabet and Cleaning Rules**: Define custom character sets and text cleaning rules for your target language.
- **Speech Generation**: Generate speech from text using trained models.
- **Google Drive Integration**: Seamlessly work with datasets and models stored on Google Drive.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/aleph-webeta.git
   cd aleph-webeta
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) If using Google Colab, mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Usage

1. Launch the Gradio interface:
   ```python
   python TTS_Model.py
   ```

2. Use the GUI to:
   - Preprocess your audio data
   - Configure and train your TTS model
   - Generate speech from text using your trained model

## Contributing

Contributions to ALEPH WEBETA are welcome! Please feel free to submit pull requests, create issues or spread the word.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) for the underlying TTS framework
- [Gradio](https://www.gradio.app/) for the GUI components

## Contact

Robel Adugna - [robadugna19@gmail.com|+251913250168| Ethiopia]

Project Link: [https://github.com/robadugna2/ALEPH_WEBETA-TTSMODEL-GUI)

---

Made with ❤️ for advancing TTS technology in low-resource languages.
