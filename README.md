
# 🧠 LLM-vision-Captioning

**LLM-vision-Captioning** is a Python-based tool designed to automatically generate natural language captions for images using large language models (LLMs) with vision capabilities. This tool is ideal for researchers, dataset curators, content creators, and developers looking to automate the annotation of image datasets.

---

## 📌 Features

- ✅ Automatic image captioning using LLMs with vision support  
- 📁 Batch processing of images from a directory  
- ⏱ Progress tracking with estimated remaining time  
- 🖼️ Supports common image formats (`.jpg`, `.png`)  
- 🌈 Terminal color output for easier monitoring (with `colorama`)  
- 🧩 Modular codebase for easy extension and customization  

---

## 📂 Folder Structure

```
LLM-vision-Captioning/
├── libs/                  # Core model execution logic
├── ui/                    # User interface logic (if any)
├── captioning/            # Folder for storing image data
├── main.py                # Entry point script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🚀 Installation

To install and run this tool, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/adigayung/LLM-vision-Captioning

# 2. Navigate into the project directory
cd LLM-vision-Captioning

# 3. Install all required Python packages
pip install -r requirements.txt
```

> ✅ Python 3.8 or higher is required.

---

## ▶️ Usage

Once dependencies are installed, run the tool using:

```bash
python main.py
```

By default, it will:

- Load the selected model
- Load prompts for guiding the caption generation
- Iterate through all `.jpg` and `.png` images in the selected folder
- Generate captions
- Display progress in the terminal, including:
  - Iteration status `[3 / 10]`
  - Time taken per image
  - Total elapsed time
  - Estimated remaining time

Example output:

```
[3 / 10] Processing: /images/cat.jpg
✅ Success
⏱ Time taken         : 2.4 seconds
⏳ Elapsed time       : 6.5 seconds
🕒 Estimated remaining: 17.1 seconds
--------------------------------------------------
```

---

## 🧠 How It Works

The core logic resides in `libs/ExecuteModel.py` and `libs/LoadModel.py`, where images are passed to a vision-language model along with a text prompt. The model then returns a natural language description of the visual content. You can modify or extend the prompt logic as needed.

---

## 🛠 Requirements

Key Python libraries used:

- `transformers`
- `Pillow`
- `colorama`
- `gradio` *(optional UI integration)*
- `torch`
- `tqdm`

All dependencies are listed in `requirements.txt`.

---

## 🧪 Model Support

This project is designed to work with checkpoint-based vision-language models. Make sure the model path and configurations are correctly set inside the source code if you're loading a specific architecture (e.g., BLIP, MiniGPT, LLaVA, etc.).

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Submit issues
- Open pull requests
- Suggest new features or enhancements

---

## 👨‍💻 Author

Created by [**@adigayung**](https://github.com/adigayung)

If you use this tool for your work or research, a star ⭐️ on the repo would be appreciated!

---
