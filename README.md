<h1 align="center"> DICTA-Chat-2.0-LLM</h1>

<h3 align="center">An Interactive Chatbot Powered by DICTA's Hebrew Language Model</h3>

<p align="left">
DICTA-Chat is an intuitive web-based interface working on NVIDIA GPU's that allows users to engage in interactive conversations with a powerful Hebrew language model developed by DICTA. With customizable generation parameters and real-time responses, DICTA-Chat provides a seamless and engaging user experience for generating high-quality Hebrew text.
</p>

![Dicta-chat](https://github.com/ShmuelRonen/DICTA-Chat-2.0-LLM/assets/80190186/a971c385-6c4c-4f01-8ee6-7f6f88343790)


## Features

- **Intuitive Web Interface**: DICTA-Chat offers a user-friendly web interface built with Gradio, making it easy for users to interact with the chatbot.

- **Real-time Responses**: Users can input their messages and receive generated responses from the DICTA language model in real-time, enabling dynamic and engaging conversations.

- **Customizable Generation Parameters**: DICTA-Chat provides a range of adjustable generation parameters, such as max new tokens, min length, temperature, and more, allowing users to fine-tune the generated text according to their preferences.

- **Conversation History**: The chatbot maintains a conversation history, ensuring that the context of the dialogue is preserved throughout the interaction.

- **Paragraph Creation**: Users have the option to enable the creation of paragraphs in the generated responses, enhancing readability and structure.

- **System Prompt Customization**: DICTA-Chat supports custom system prompts, enabling users to define the chatbot's persona, tone, and behavior.

- **System Prompt Management**: Users can save and load custom system prompts, making it convenient to switch between different chatbot personalities.

- **Copy Last Response**: The application provides a button to easily copy the last generated response, facilitating sharing or further processing of the text.

- **Clear Chat**: Users can clear the conversation history with a single click, starting a new interaction from scratch.

- **Right-to-Left Text Support**: DICTA-Chat fully supports right-to-left (RTL) text alignment, ensuring a natural reading experience for Hebrew conversations.


## Installation

### One-click Installation

1. Clone the repository:
```
git clone https://github.com/ShmuelRonen/DICTA-Chat-2.0-LLM.git
cd DICTA-Chat-2.0-LLM
```
2. Click on:
```
init_env_dicta.bat
```
The script will automatically set up the virtual environment and install the required dependencies.

### Manual Installation

1. Clone the repository:
```
git clone https://github.com/ShmuelRonen/DICTA-Chat-2.0-LLM.git
cd DICTA-Chat-2.0-LLM
```
2. Create and activate a virtual environment:
 ```
python -m venv venv
venv\Scripts\activate
```

4. Install the required dependencies:
```
pip install -r requirements.txt
pip install -i https://pypi.org/simple/ bitsandbytes
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
After the installation, you can run the app by executing:
```
python chat_dicta.py
```
This will start the Gradio interface locally, which you can access through the provided URL in your command line interface.

## Acknowledgements

DICTA-Chat leverages the powerful Hebrew language model developed by [DICTA](https://huggingface.co/dicta-il/). We express our gratitude to the DICTA team for their remarkable work in advancing Hebrew natural language processing.

## Feedback and Support

If you have any questions, suggestions, or issues regarding DICTA-Chat, please feel free to [open an issue](https://github.com/ShmuelRonen/DICTA-Chat-2.0-LLM/issues) on the GitHub repository. We appreciate your feedback and contributions to improving the application.

## License

This project is licensed under the [MIT License](https://github.com/your-username/DICTA-Chat/blob/main/LICENSE).
Note: Replace https://your-repo-url/screenshot.png with the actual URL of a screenshot showcasing your DICTA-Chat interface, and replace https://github.com/your-username/DICTA-Chat with the actual URL of your GitHub repository.
This README.md file provides an overview of the DICTA-Chat application, highlighting its key features, installation instructions, acknowledgements, and support information. It follows the markdown format and includes the necessary formatting tags and symbols.
