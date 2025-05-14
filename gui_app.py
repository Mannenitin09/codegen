import tkinter as tk
from tkinter import scrolledtext, messagebox
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import threading
from config import MODEL_CONFIG, OUTPUT_DIR

# --- Global Variables ---
device = None
model = None
tokenizer = None

# --- Rule-Based Handler ---
def rule_based_handler(user_input: str) -> str:
    import re
    input_lower = user_input.lower()

    if "add" in input_lower and re.search(r"\d+", input_lower):
        numbers = list(map(int, re.findall(r"\d+", input_lower)))
        return f"def add_numbers():\n    result = {numbers[0]} + {numbers[1]}\n    print(result)\n\nadd_numbers()"

    elif "subtract" in input_lower and re.search(r"\d+", input_lower):
        numbers = list(map(int, re.findall(r"\d+", input_lower)))
        return f"def subtract_numbers():\n    result = {numbers[0]} - {numbers[1]}\n    print(result)\n\nsubtract_numbers()"

    elif "multiply" in input_lower and re.search(r"\d+", input_lower):
        numbers = list(map(int, re.findall(r"\d+", input_lower)))
        return f"def multiply_numbers():\n    result = {numbers[0]} * {numbers[1]}\n    print(result)\n\nmultiply_numbers()"

    elif "divide" in input_lower and re.search(r"\d+", input_lower):
        numbers = list(map(int, re.findall(r"\d+", input_lower)))
        return f"def divide_numbers():\n    if {numbers[1]} != 0:\n        result = {numbers[0]} / {numbers[1]}\n        print(result)\n    else:\n        print(\"Division by zero error\")\n\ndivide_numbers()"

    elif "factorial" in input_lower and re.search(r"\d+", input_lower):
        n = int(re.search(r"\d+", input_lower).group())
        return f"import math\ndef compute_factorial():\n    print(math.factorial({n}))\n\ncompute_factorial()"

    elif "square root" in input_lower and re.search(r"\d+", input_lower):
        n = int(re.search(r"\d+", input_lower).group())
        return f"import math\ndef compute_sqrt():\n    print(math.sqrt({n}))\n\ncompute_sqrt()"

    return None

# --- Model Loading ---
def load_model_and_tokenizer():
    global device, model, tokenizer
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
        model = T5ForConditionalGeneration.from_pretrained(MODEL_CONFIG['model_name'])

        checkpoint_path = OUTPUT_DIR / "checkpoints" / "best_model.pt"
        if not checkpoint_path.exists():
            messagebox.showerror("Error", f"Best model checkpoint not found at: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return True
    except Exception as e:
        messagebox.showerror("Model Load Error", f"Failed to load model: {e}")
        return False

# --- Code Generation ---
def generate_code():
    if not model or not tokenizer:
        messagebox.showwarning("Model Not Ready", "Model is not loaded yet. Please wait or restart.")
        return

    intent = intent_entry.get("1.0", tk.END).strip()
    if not intent:
        messagebox.showwarning("Input Needed", "Please enter an intent.")
        return

    generate_button.config(state=tk.DISABLED)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "Generating...")
    threading.Thread(target=generation_thread, args=(intent,)).start()

def generation_thread(intent):
    try:
        rule_based = rule_based_handler(intent)
        if rule_based:
            app.after(0, update_output, rule_based)
            return

        inputs = tokenizer(intent, return_tensors="pt", max_length=MODEL_CONFIG["max_source_length"],
                           padding="max_length", truncation=True).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=MODEL_CONFIG['max_target_length'],
                num_beams=MODEL_CONFIG.get('num_beams', 4),
                early_stopping=True
            )

        generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        app.after(0, update_output, generated_code)

    except Exception as e:
        app.after(0, update_output, f"Error during generation: {e}")

def update_output(generated_code):
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, generated_code)
    generate_button.config(state=tk.NORMAL)

# --- GUI Setup ---
app = tk.Tk()
app.title("Code Generation Assistant")
app.geometry("600x400")

input_frame = tk.Frame(app)
input_frame.pack(pady=10, padx=10, fill=tk.X)

intent_label = tk.Label(input_frame, text="Enter Intent:")
intent_label.pack(side=tk.LEFT)

intent_entry = scrolledtext.ScrolledText(input_frame, height=4, width=60, wrap=tk.WORD)
intent_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

generate_button = tk.Button(app, text="Generate Code", command=generate_code, state=tk.DISABLED)
generate_button.pack(pady=5)

output_label = tk.Label(app, text="Generated Code:")
output_label.pack(pady=(5, 0))

output_text = scrolledtext.ScrolledText(app, height=15, width=70, wrap=tk.WORD)
output_text.pack(pady=5, padx=10, expand=True, fill=tk.BOTH)

# --- Main ---
if __name__ == "__main__":
    def startup_load():
        if load_model_and_tokenizer():
            generate_button.config(state=tk.NORMAL)

    threading.Thread(target=startup_load, daemon=True).start()
    app.mainloop()
