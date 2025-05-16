# codegen
# AI CodeGenie

AI CodeGenie is an intelligent code generation tool that leverages artificial intelligence to assist developers in writing boilerplate code, improving productivity, and reducing errors. It supports a variety of programming languages and development environments, integrating seamlessly into developer workflows.

## Features

- 🧠 **AI-Powered Code Generation**: Generate code snippets using natural language prompts.
- 💬 **Natural Language Interface**: Communicate with the tool using simple and intuitive commands.
- 🚀 **Multi-language Support**: Works with multiple programming languages including Python, Java, JavaScript, and more.
- 🔍 **Smart Suggestions**: Offers context-aware code completions and improvements.
- 🛠️ **IDE Integration**: Easily integrates with popular IDEs like VS Code and IntelliJ.
- 📚 **Documentation and Commenting**: Automatically generates documentation and code comments.
- 🔄 **Code Refactoring**: Identifies code smells and offers suggestions to improve code quality.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/aicodegenie.git
   cd aicodegenie
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Launch the application:

bash
Copy
Edit
python app.py
Usage
Launch the tool and enter your code request using natural language.

View and edit the generated code in your preferred IDE.

Use additional tools for refactoring, commenting, and testing.

Example:

"Generate a Python function to calculate the Fibonacci sequence."

Architecture Overview
AI CodeGenie is composed of the following core modules:

Prompt Processor: Parses and understands user prompts.

Code Generator: Interfaces with an AI model to generate code.

Syntax Validator: Ensures generated code is syntactically correct.

IDE Plugin Interface: Connects with IDEs for real-time interaction.

Use Cases
Rapid prototyping

Learning and teaching programming

Code review and optimization

Automated documentation




yaml
Copy
Edit

---

---

## 🛠️ Technologies Used

- Python 3.x
- PyTorch
- HuggingFace Transformers
- NLTK
- Matplotlib
- Tkinter
- CoNaLa Dataset
- CodeT5-small Model

---

## 📁 Project Structure

```bash
aicodegenie/
│
├── config.py             # Centralized config for model, training, logging
├── dataset.py            # Data loading and preprocessing
├── gui_app.py            # Tkinter-based GUI for real-time generation
├── model.py              # CodeT5 wrapper for initialization and inference
├── train.py              # Model training logic
├── test.py               # Evaluation using BLEU & Exact Match
├── utils.py              # Helper functions and plotting tools
├── requirements.txt      # Python package dependencies
└── README.md             # Project documentation
⚙️ Installation
Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/aicodegenie.git
cd aicodegenie
Install Dependencies

Install PyTorch separately based on your system (CPU/GPU):
https://pytorch.org/get-started/locally/

bash
Copy
Edit
pip install -r requirements.txt
🧪 Usage
Run GUI Application

bash
Copy
Edit
python gui_app.py
Enter a Natural Language Instruction

Example: Sort a list in descending order

Output: mylist.sort(reverse=True)

📊 Results
Metric	Score
BLEU Score	0.74
Exact Match	0.61

📈 Training vs Validation Loss shows consistent learning.

🔵 BLEU Score improved from 0.42 to 0.74.

✅ Exact Match increased steadily across epochs.

📌 Limitations
May struggle with ambiguous or compound prompts.

Performs best on tasks well-represented in the training data.

No runtime execution or validation for generated code (for security reasons).

🚧 Future Scope
🌍 Multi-language support (e.g., JavaScript, Java, C++).

🌐 Web-based interface and REST API.

🧪 Runtime validation and error detection.

🧠 Reinforcement learning from user feedback.

🧩 IDE plugins (e.g., VSCode, Jupyter).

🎓 Educational assistant features for beginners.

📚 References
CodeT5 Model - Salesforce Research

HuggingFace Transformers

CoNaLa Dataset

BLEU Score - Machine Learning Mastery

PyTorch Documentation

🤝 Contributing
Contributions are welcome!

Fork the repo

Create your feature branch (git checkout -b feature/feature-name)

Commit your changes (git commit -am 'Add feature')

Push to the branch (git push origin feature/feature-name)

Open a pull request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👥 Authors
Nitin Manne (U00353741)

Thanvi Lahari Pendyala (U00353645)

Project Guide: Dr. Amirhossein Manzourolajdad

