# BDRC Tibetan OCR desktop app

This app is a free and open source desktop app that can be installed on a local compter to run Tibetan OCR on batches of images (including in PDFs). It was developed by Eric Werner for the [Buddhist Digital Resource Center](https://www.bdrc.io) and uses the pipeline developed during the BDRC OCR Project.

### Main features

The app can open one or multiple files and run OCR on them. It can export plain text or [PageXML](https://github.com/PRImA-Research-Lab/PAGE-XML) (a format it shares with [Transkribus](https://www.transkribus.org/)).

It can also optionally dewarp images as well as convert the output to Wylie.

Instead of providing one model that can handle all styles of Tibetan writing, we provide a few different models that we encourage users to experiment with to see what fits their data best.

The models it uses are based on transcriptions available online, from BDRC, [ALL](https://asianlegacylibrary.org/), [Adarsha](https://adarshah.org/), and [NorbuKetaka](http://purl.bdrc.io/resource/PR1ER1), as well as transcriptions made during the project.

See the Huggingface accounts of [BDRC](https://huggingface.co/BDRC) and [OpenPecha](https://huggingface.co/openpecha) for the open access part of our data as well as the models, and the following for the training code:

- https://github.com/buda-base/tibetan-ocr-training
- https://github.com/buda-base/tibetan-ocr-evaluation

### Installation and running

##### Windows

1. Download and unzip https://github.com/buda-base/tibetan-ocr-app/releases/download/v0.3.0/bdrc_ocr_windows_x64_0.3.zip
2. Run the `.exe`

##### OSX

This app has packages for MacOS X on both Intel (x64) processors and Silicon/ARM (M1, M2, etc.) processors available.

1. Download and unzip

- For recent hardware (M1, M2, etc.): https://github.com/buda-base/tibetan-ocr-app/releases/download/v0.3.0/bdrc_ocr_macos_arm64_0.3.zip
- For older x64 (Intel) processors: https://github.com/buda-base/tibetan-ocr-app/releases/download/v0.3.0/bdrc_ocr_macos_x64_0.3.zip

2. Run the app

##### From source (advanced users)

1. Make sure you have [Git LFS](https://git-lfs.com) installed
2. Clone the Github repository: `git clone https://github.com/buda-base/tibetan-ocr-app.git`
3. Run `git lfs pull` to download all LFS files
4. Install dependencies with `pip install -r requirements.txt` (requires at least Python 3.10)
5. Install Poppler with `python scripts/install_poppler.py`
6. Run `pyside6-rcc resources.qrc -o resources.py`
7. Download the OCR models with `curl -L https://github.com/buda-base/tibetan-ocr-app/releases/download/v0.1/bdrc_ocr_models_1.0.zip`
8. Extract the OCR models ZIP archive into a new `OCRModels` directory.
9. Run `python main.py`

### OCR Models

The application comes with pre-installed OCR models that are ready to use. These models are automatically loaded when you start the application.

If you want to use different models:

1. Download and unzip the models in a directory of your choice
2. Open the app, click on the setting icon, click on "import models" and select the `ORCModels/` folder where you extracted the model zip file. Warning! Do not select one of its subfolders (like `Woodblock/`, etc.).
3. The app will immediately start using these custom models.

At that stage we advise you to try the app on a few images. If you're not satisfied with the result, please try setting the "bbox tolerance" setting value to `3.5` or `2.5` and see if it improves the results.

### Building distribution packages

1. `pip install nuitka`
2. (optional) install `ccache` to speed up the compilation (on OSX this can be done through homebrew)
3. run the nuitka command given in main.py that corresponds to your OS
4. zip the files in the corresponding build folder

### Troubleshooting

#### PDF Processing Issues

If you encounter issues with PDF processing, it might be related to Poppler:

1. Make sure Poppler is properly installed using the script provided: `python scripts/install_poppler.py`
2. Verify that the Poppler binaries are in the correct location:
   - Windows: Check that `poppler/bin/pdfinfo.exe` exists
   - macOS/Linux: Check that `poppler/bin/pdfinfo` exists
3. If you installed Poppler manually, make sure the application can find it:
   - Windows: Add the Poppler `bin` directory to your PATH
   - macOS: If using Homebrew, run `brew info poppler` to check the installation path
   - Linux: Ensure `poppler-utils` is installed

#### Missing Dependencies

If you get errors about missing Python modules, make sure you've installed all requirements:

```
pip install -r requirements.txt
```

### Acknowledgements

Our gratitude goes to Jérémy Frère for the OSX packaging and Pentsok W. Rtsang for the translations into Tibetan.
