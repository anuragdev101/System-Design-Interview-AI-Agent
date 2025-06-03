<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">SYSTEM-DESIGN-INTERVIEW-AI-AGENT</h1></p>
<p align="center">
	<em>Designing Tomorrow's Systems with Today's Conversations</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/anuragdev101/System-Design-Interview-AI-Agent?style=flat-square&logo=opensourceinitiative&logoColor=white&color=ff4b4b" alt="license">
	<img src="https://img.shields.io/github/last-commit/anuragdev101/System-Design-Interview-AI-Agent?style=flat-square&logo=git&logoColor=white&color=ff4b4b" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/anuragdev101/System-Design-Interview-AI-Agent?style=flat-square&color=ff4b4b" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/anuragdev101/System-Design-Interview-AI-Agent?style=flat-square&color=ff4b4b" alt="repo-language-count">
</p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=flat-square&logo=OpenAI&logoColor=white" alt="OpenAI">
</p>
<br>

<p> Click on Image below to watch the demo video </p>

[![Watch our demo video](https://github.com/user-attachments/assets/27910c9f-0c5e-471f-a6ee-808d8b2db214)](https://www.canva.com/design/DAGpN7_XAig/j3iLQr52gewgMJkvEWiApg/watch?utm_content=DAGpN7_XAig&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h9ea99f6c0a)

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📁 Project Structure](#-project-structure)
  - [📂 Project Index](#-project-index)
- [🚀 Getting Started](#-getting-started)
  - [☑️ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#-installation)
  - [🤖 Usage](#🤖-usage)
  - [🧪 Testing](#🧪-testing)
- [🔰 Contributing](#-contributing)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

The System-Design-Interview-AI-Agent revolutionizes technical interviews by generating real-time system design diagrams and providing intelligent responses based on conversational context. This tool is invaluable for interviewers and candidates alike, offering a dynamic platform to interact, visualize, and evaluate technical designs efficiently. Perfect for enhancing the clarity and effectiveness of system design validations during interviews.

---

## 👾 Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Centralized orchestration and audio stream management in `main.py`.</li><li>Integrated GUI setup using `<tkinter>` and `<ttkthemes>` in `main_gui.py` for enhanced user interaction.</li><li>Utilizes `<langchain-core>`, `<whisper>`, and `<torch>` for AI-driven processing and speech recognition tasks.</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Modular design with separate functional modules for distinct tasks like audio capture (`audio_capture.py`), diagram generation (`diagram_generator.py`), and AI processing (`ai_processor.py`).</li><li>Robust error handling and logging systems indicating attention to reliability and maintainability.</li><li>Adherence to modern software design principles evident from structured modular files and systematic use of APIs and classes.</li></ul> |
| 📄 | **Documentation** | <ul><li>Well-documented setup and installation procedures using `pip` as seen in `requirements.txt`.</li><li>Code comments in Python modules enhance understandability and maintainability of the system.</li><li>Utilization of Markdown badges for visual enhancement and quick recognition in documentation.</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integrates with various audio and AI libraries such as `<PyAudio>`, `<whisper>`, `<torch>`, and `<torchaudio>` for comprehensive functionality.</li><li>Uses both open-source and custom API integrations to handle complex design tasks in `design_api.py`.</li><li>Connection between GUI and backend processing for real-time data handling and visualization.</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Clear division into modules for specific functionalities like speech recognition (`speech_recognition.py`) and speaker diarization (`speaker_diarization.py`).</li><li>High reusability of modules like `gui.py` which can be adapted for different user interface needs.</li><li>Each module, such as `audio_capture.py`, encapsulates its functionality, enhancing maintainability and testability.</li></ul> |
| 🧪 | **Testing**       | <ul><li>Structured testing commands in the documentation suggest an environment for continuous integration and testing.</li><li>Use of `<pytest>` indicates adoption of automated tests to ensure module integrity.</li><li>Modularity supports unit testing of individual components to verify functionality separately.</li></ul> |

---

## 📁 Project Structure

```sh
└── System-Design-Interview-AI-Agent/
    ├── README.md
    ├── main.py
    ├── main_gui.py
    ├── modules
    │   ├── __init__.py
    │   ├── ai_processor.py
    │   ├── audio_capture.py
    │   ├── console_display.py
    │   ├── design_api.py
    │   ├── design_models.py
    │   ├── diagram_generator.py
    │   ├── gui.py
    │   ├── speaker_diarization.py
    │   └── speech_recognition.py
    └── requirements.txt
```


### 📂 Project Index
<details open>
	<summary><b><code>SYSTEM-DESIGN-INTERVIEW-AI-AGENT/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/main_gui.py'>main_gui.py</a></b></td>
				<td>- The `main_gui.py` file serves as the central user interface component for an integrated software suite that leverages artificial intelligence to enhance audio processing capabilities<br>- Its primary function includes initializing and managing the graphical user interface (GUI) where users interact with the application<br>- The file integrates various modules, including `AIProcessor`, `AudioCapture`, `DiagramGenerator`, `SpeakerDiarization`, and `SpeechRecognition`, creating a streamlined workflow that captures audio, processes it via AI-enhanced methods, and provides visual and textual outputs directly to the users.

This file is designed as the main entry point for the application, indicated by the setup of environment variables and the central logging system aimed at capturing runtime events for debugging and monitoring<br>- The GUI is built using the `tkinter` library, enhanced by `ttkthemes` for aesthetic design, indicating a focus on usability and user experience<br>- The integration of components through this file suggests it is pivotal for application operation, orchestrating interactions between the user interface and underlying processing modules, thereby acting as the backbone of the entire codebase architecture.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/main.py'>main.py</a></b></td>
				<td>- Main.py serves as the central orchestrator for the System Design Interview AI Agent, coordinating various components such as audio capture, speaker diarization, speech recognition, AI processing, and diagram generation<br>- It manages the flow and processing of audio streams, capturing interviewer and candidate dialogues, and integrates AI responses and diagrams based on the conversation context.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td>- Centralizes the primary dependencies for numerical operations, machine learning models, GUI components, audio processing, and AI-driven modules necessary for the project's functionality<br>- The requirements.txt file specifies version-controlled libraries that support tasks ranging from speech recognition and speaker diarization to environmental configuration and GUI enhancements within the application framework.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- modules Submodule -->
		<summary><b>modules</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/modules/design_api.py'>design_api.py</a></b></td>
				<td>- Enables the generation, storage, and management of system designs, components, and connections within a custom architecture<br>- Supports operations to create and retrieve detailed system designs, including components, API endpoints, and data models, with capabilities for translating high-level designs into detailed, low-level specifications.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/modules/console_display.py'>console_display.py</a></b></td>
				<td>- Enhances user interaction within the system by providing an aesthetically pleasing and informative console interface for displaying diverse types of outputs, including error messages, system statuses, and transcriptions from multiple speakers<br>- Key features include visual differentiation of speaker identities and integration of dynamic system feedback, such as process thinking and diagram generation notifications.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/modules/gui.py'>gui.py</a></b></td>
				<td>- The `modules/gui.py` file is integral to the codebase as it implements the graphical user interface (GUI) for the System Design Interview AI Agent<br>- This GUI serves as the primary interaction layer between the user and the AI agent, facilitating the visualization and manipulation of data through a user-friendly interface<br>- Specifically, the GUI offers functionalities such as displaying diagrams, interacting with agent-generated content, and providing updates in real time, enhancing the overall user experience and usability of the system<br>- The design and architectural choices focus on responsiveness and effective visual communication, making it a crucial component in the broader context of the application’s architecture<br>- This file's role is to bridge the backend computations and algorithms with a tangible interface that users can interact with to leverage the capabilities of the AI agent without needing to delve into underlying technical complexities.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/modules/audio_capture.py'>audio_capture.py</a></b></td>
				<td>- The `audio_capture.py` file within the `modules` directory plays a pivotal role in the project's architecture by providing the essential functionality for capturing audio from various system devices<br>- This component integrates various audio libraries to ensure wide compatibility and robust handling across different platforms—capabilities to note include utilizing `sounddevice` and conditional integration with `pyaudio` for basic audio capture, `pyaudiowpatch` specifically for Windows audio loopback, and leveraging `librosa` for audio resampling<br>- The class `AudioCapture` designed in this script serves the primary purpose of capturing system audio, enabling selection of different devices, and processing the audio into usable chunks at a specified sample rate and channel setup.

This module serves as a fundamental building block in the overall project, likely contributing to functionalities that require real-time or recorded audio input, such as voice recognition, sound analysis, or interactive applications requiring audio feedback from users<br>- Its design, which handles various fallback mechanisms when certain dependencies are not available, highlights the robustness and the adaptive nature of the project's architecture<br>- Additionally, the emphasis on logging and condition-based imports demonstrates a mindful approach to cross-compatibility and error management within the project<br>- The presence and handling of different audio libraries ensure that the module can operate across various hardware and software configurations, enhancing the project's utility and user reach.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/modules/diagram_generator.py'>diagram_generator.py</a></b></td>
				<td>- Generates system design diagrams using a library of predefined components that span common programming, on-premise, and AWS services<br>- The module offers functionality to produce both high-level and detailed low-level diagrams customizable based on input specifications regarding components and their interconnections.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/modules/design_models.py'>design_models.py</a></b></td>
				<td>- Defines data models and class structures central to system design, encompassing component types such as servers, databases, and service meshes<br>- The module facilitates structured representations of design components, connections, API details, and security configurations, crucial for building and documenting sophisticated system architectures effectively.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/modules/speaker_diarization.py'>speaker_diarization.py</a></b></td>
				<td>- SpeakerDiarization in the `modules/speaker_diarization.py` identifies and differentiates speakers within audio inputs using the pyannote.audio library<br>- It initializes with model configurations, handles audio processing including resampling, and executes speaker diarization<br>- The module also includes capabilities for dynamic calibration of speaker identities based on audio analysis.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/modules/ai_processor.py'>ai_processor.py</a></b></td>
				<td>- The "ai_processor.py" module serves as a core component within the overall architecture of the system, focused on utilizing advanced large language models (LLMs) for processing interview text<br>- Its primary purpose is to generate system design diagrams and provide answers to follow-up questions related to system designs in technical interviews<br>- This involves leveraging models like GPT-3.5 to interprete and translate complex technical discussions into structured outputs such as DevOps strategies, security architectures, and more.

This module interacts significantly with other components of the system, notably the SystemDesignAPI for managing the designs and various design model classes that represent different aspects of system architecture<br>- The initialization of this module ensures all necessary components, such as the OpenAI API and language chain models, are set up and ready to use, which is crucial for the seamless generation and retrieval of the required information<br>- The presence and configuration of these elements underscore the module's integral role in the synthesized operation of AI-driven design generation within the broader application platform.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/master/modules/speech_recognition.py'>speech_recognition.py</a></b></td>
				<td>- Handles conversion of spoken language into text, primarily utilizing OpenAI's Whisper model<br>- Should Whisper fail, it employs other speech recognition engines as backups<br>- The process also incorporates checks and transformations of audio data to ensure compatibility and enhance accuracy in transcription outputs.</td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with System-Design-Interview-AI-Agent, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


### ⚙️ Installation

Install System-Design-Interview-AI-Agent using one of the following methods:

**Build from source:**

1. Clone the System-Design-Interview-AI-Agent repository:
```sh
❯ git clone https://github.com/anuragdev101/System-Design-Interview-AI-Agent
```

2. Navigate to the project directory:
```sh
❯ cd System-Design-Interview-AI-Agent
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```




### 🤖 Usage
Run System-Design-Interview-AI-Agent using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python main_gui.py
```


### 🧪 Testing
Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pytest
```


---


## 🔰 Contributing

- **💬 [Join the Discussions](https://github.com/anuragdev101/System-Design-Interview-AI-Agent/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/anuragdev101/System-Design-Interview-AI-Agent/issues)**: Submit bugs found or log feature requests for the `System-Design-Interview-AI-Agent` project.
- **💡 [Submit Pull Requests](https://github.com/anuragdev101/System-Design-Interview-AI-Agent/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/anuragdev101/System-Design-Interview-AI-Agent
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/anuragdev101/System-Design-Interview-AI-Agent/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=anuragdev101/System-Design-Interview-AI-Agent">
   </a>
</p>
</details>

---

## 🎗 License

This project is licensed under the **[MIT License](https://opensource.org/licenses/MIT)**.

---

## 🙌 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
