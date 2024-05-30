import os
import tkinter as tk
from tkinter import font as tkfont, messagebox
from peft import PeftConfig, PeftModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import pandas as pd

# Function to load the FLAN-T5 model
def load_flan_t5_model(model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer

# Function to load the BERT model
def load_bert_model(model_path):
    question_answerer = pipeline("question-answering", model=model_path)
    return question_answerer

# Locate current directory and load data
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)
covid_data = pd.read_csv("Bert/database_small.csv")
database = covid_data['context'].tolist()

# Load models
flan_model, flan_tokenizer = load_flan_t5_model("./checkpoint")
bert_model = load_bert_model("./Bert/model/bert-COVID-QA")

# Function to get the answer from the selected model
def get_answer(question):
    if model_version.get() == 1:  # Use FLAN-T5
        input_ids = flan_tokenizer(question, return_tensors="pt").input_ids.to("cpu")
        outputs = flan_model.generate(input_ids=input_ids)
        answer = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return f" Question: {question}\n Answer from our model: {answer}"
    else:  # Use BERT
        score = 0
        best_answer = ''
        for content in database:
            res = bert_model(question=question, context=content)
            if res['score'] > score:
                best_answer = res['answer']
                score = res['score']
        return f" Question: {question}\n Answer from our model: {best_answer}"

# Initialize Tkinter window
window = tk.Tk()
window.title("QA System")
window.geometry("1250x1200")
window.configure(bg='#f7f7f7')

# Set fonts
font_title = tkfont.Font(family="Arial Bold", size=20)
font_input = tkfont.Font(family="Arial", size=16)
font_response = tkfont.Font(family="Arial Bold", size=18)

# Model selection
model_version = tk.IntVar()
model_version.set(1)  # Default select FLAN-T5

lbl_version = tk.Label(window, text="Select Model:", font=(font_input.actual()['family'], font_input.actual()['size'], 'bold'), bg='#f7f7f7')
lbl_version.grid(column=0, row=0, pady=20, padx=20, sticky='w')

radio_flan = tk.Radiobutton(window, text="Flan-T5-base", variable=model_version, value=1, font=font_input, bg='#f7f7f7')
radio_flan.grid(column=0, row=1, pady=20, padx=20, sticky='w')

radio_bert = tk.Radiobutton(window, text="Bert", variable=model_version, value=2, font=font_input, bg='#f7f7f7')
radio_bert.grid(column=1, row=1, pady=20, padx=20, sticky='w')

# Define input box
question_text = "Please input your question..."
txt_question = tk.Text(window, width=85, height=5, font=font_input, borderwidth=2, relief="solid", bg='#ffffff', wrap="word")
txt_question.grid(column=0, row=3, columnspan=2, pady=20, padx=20, sticky='nsew')
txt_question.insert(tk.END, question_text)
txt_question.config(foreground="#888888")

# Clear default text on focus
def clear_default_text(event):
    if txt_question.get("1.0", "end-1c") == question_text:
        txt_question.delete("1.0", "end-1c")
        txt_question.config(foreground="#000000")

txt_question.bind("<FocusIn>", clear_default_text)

# Scrollbar for question input
scrollbar = tk.Scrollbar(window, width=15, command=txt_question.yview)
scrollbar.grid(column=2, row=3, pady=20, sticky='ns')
txt_question['yscrollcommand'] = scrollbar.set

# Define output
lbl_response = tk.Label(window, text="", font=font_response, bg='#f7f7f7', wraplength=1000)
lbl_response.grid(column=0, row=4, columnspan=2, pady=20, padx=20, sticky='ew')
lbl_response.configure(justify='left')

# Variable to store the response
response_text = tk.StringVar()

# Function to handle button click
def clicked():
    question = txt_question.get("1.0", "end").strip()
    if question and question != question_text:
        answer = get_answer(question)
        response_text.set(answer)
        lbl_response.configure(text=response_text.get())
        txt_question.delete("1.0", tk.END)
    else:
        messagebox.showwarning("Input Error", "Please enter a valid question.")

# Button to submit question
btn = tk.Button(window, text="Get Answer", font=font_input, bg="#4CAF50", fg='#ffffff', borderwidth=2, relief="solid", command=clicked)
btn.grid(column=2, row=4, pady=20, padx=20, sticky='ew')

# Make columns and rows adaptive
window.columnconfigure((0, 1), weight=1)
window.rowconfigure((2, 3, 4, 5), weight=1)

# Run the main loop
window.mainloop()
