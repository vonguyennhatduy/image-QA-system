# Image AI Assistant: Ask Questions to Your Images

An AI-powered web application built with **Streamlit**, **LangChain**, and **Ollama**. This project uses a ReAct Agent to analyze uploaded images through specialized tools like Image Captioning and Object Detection.

## 🚀 Features

- **Interactive UI**: Upload images (JPG, PNG) and chat with them in real-time.
- **AI-Powered Reasoning**: Uses the `phi3` model (via Ollama) to understand user queries and decide which tool to use.
- **Image Captioning**: Generates descriptive text for images using the `Salesforce/blip-image-captioning-large` model.
- **Object Detection**: Identifies and locates objects within the image using the `facebook/detr-resnet-50` model.
- **Conversation Memory**: Remembers the context of the conversation for a better user experience.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **LLM**: [Ollama](https://ollama.com/) (Model: `phi3`)
- **Computer Vision**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (BLIP & DETR)
- **Image Processing**: Pillow (PIL)

## 📋 Prerequisites

1.  **Python 3.10+**
2.  **Ollama**: Install Ollama from [ollama.com](https://ollama.com/) and pull the phi3 model:
    ```bash
    ollama pull phi3
    ```

## 🔧 Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install streamlit langchain langchain-ollama transformers torch torchvision pillow
    ```

## 💻 Usage

1.  **Start the Streamlit app**:
    ```bash
    streamlit run main.py
    ```
2.  Open your browser to `http://localhost:8501`.
3.  Upload an image and ask questions like:
    - *"What is in this picture?"*
    - *"Detect all objects in this image."*
    - *"How many people are there?"*

## 📁 Project Structure

- `main.py`: The entry point of the Streamlit application and Agent logic.
- `tool.py`: Contains the `ImageCaptionTool` and `ObjectDetectionTool` classes.
- `functions.py`: Standalone functions for image processing and testing.

## ⚠️ Important Notes

- **GPU Acceleration**: The application automatically detects if a CUDA-enabled GPU is available. If not, it defaults to CPU (which may be slower for image processing).
- **Temporary Files**: The app creates temporary files to process uploaded images. These are automatically deleted after each query to save disk space.
- **Windows Permission**: If you encounter `PermissionError`, ensure you are using the updated `NamedTemporaryFile` logic provided in the latest version of the code.
