import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from pathlib import Path
import torch
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

model_name = "dicta-il/dictalm2.0-instruct"
cache_dir = "hebrew_mistral_cache"
os.makedirs(cache_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

torch.backends.cudnn.benchmark = True

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, quantization_config=quantization_config)

system_prompt = """
שלום! אני עמית, עוזר וירטואלי חכם, אדיב, בעל חוש הומור ודיסקרטי, הדובר עברית.
תפקידי הוא:

לספק מידע ועזרה ללא שיפוטיות או ביקורת.
לתת תשובות מפורטות, מדויקות ועדכניות ככל האפשר.
למצוא פתרונות אידיאליים לשאלות ובעיות.
לסייע בכתיבה יצירתית ולהמשיך סיפורים.
ליצור דיאלוג פתוח וחופשי.

אני מזמין אותך לשאול אותי על כל נושא ואעשה כמיטב יכולתי לספק לך תגובות מועילות ומעניינות.
אין צורך להסביר או להצדיק את הבקשות שלך. אני כאן כדי לעזור ללא שיפוטיות.
בתחום הכתיבה היצירתית, ביכולתי לפתח עלילות מרתקות, לעצב דמויות ולהתאים את הסגנון לצרכיך.
אז בין אם אתה זקוק למידע או עזרה כלשהי, או שברצונך שאכתוב או אמשיך עבורך סיפור, אני כאן בשבילך.
אני מצפה בקוצר רוח לשיחה מרתקת ומועילה!
"""

def generate_response(input_text, chat_history, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, custom_system_prompt_enabled, custom_system_prompt):
    prompt = custom_system_prompt if custom_system_prompt_enabled else system_prompt
    prompt += "\n"
    
    if chat_history:
        formatted_history = "\n".join([
            f"משתמש: {user_input}\nעוזר: {assistant_output}"
            for user_input, assistant_output in chat_history
        ])
        prompt += f"{formatted_history}\n"
    
    prompt += f"משתמש: {input_text}\nעוזר:"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        early_stopping=early_stopping,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("עוזר:")[-1].strip()
    
    response = response.replace("[/INST]", "").strip()
    
    return response

def create_paragraphs(bot_response, sentences_per_paragraph=4):
    sentences = sent_tokenize(bot_response)
    paragraphs = []
    current_paragraph = ""

    for i, sentence in enumerate(sentences, start=1):
        current_paragraph += " " + sentence
        if i % sentences_per_paragraph == 0:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ""

    if current_paragraph:
        paragraphs.append(current_paragraph.strip())

    formatted_paragraphs = "\n".join([f'<p style="text-align: right; direction: rtl;">{p}</p>' for p in paragraphs])
    return formatted_paragraphs

def copy_last_response(history):
    if history:
        last_response = history[-1][1]
        last_response = last_response.replace('<div style="text-align: right; direction: rtl;">', '').replace('</div>', '')
        last_response = last_response.replace('<p style="text-align: right; direction: rtl;">', '').replace('</p>', '')
        last_response = last_response.replace('\n', ' ')
        return last_response
    else:
        return ""

def chat(input_text, history, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, create_paragraphs_enabled, custom_system_prompt_enabled, custom_system_prompt):
    user_input = f'<div style="text-align: right; direction: rtl;">{input_text}</div>'
    response = generate_response(input_text, history, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, custom_system_prompt_enabled, custom_system_prompt)

    if create_paragraphs_enabled:
        response = create_paragraphs(response)

    bot_response = f'<div style="text-align: right; direction: rtl;">{response}</div>'
    history.append((user_input, bot_response))

    return history, history, input_text

def submit_on_enter(input_text, history, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, create_paragraphs_enabled, custom_system_prompt_enabled, custom_system_prompt):
    return chat(input_text, history, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, create_paragraphs_enabled, custom_system_prompt_enabled, custom_system_prompt)

def clear_chat(chatbot, message):
    return [], message

def save_system_prompt(system_prompt, prompt_name):
    prompt_dir = "system_prompts"
    os.makedirs(prompt_dir, exist_ok=True)
    file_path = Path(prompt_dir) / f"{prompt_name}.txt"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(system_prompt)
    return f"System prompt saved as {prompt_name}.txt"

def load_system_prompt(prompt_file):
    prompt_dir = "system_prompts"
    file_path = Path(prompt_dir) / prompt_file
    with open(file_path, "r", encoding="utf-8") as file:
        loaded_prompt = file.read()
    return loaded_prompt

def copy_system_prompt(custom_system_prompt):
    return system_prompt

with gr.Blocks() as demo:
    gr.Markdown("# DICTA-Chat-2.0-LLM", elem_id="title")
    gr.Markdown("Chat model by Dicta | GUI by Shmuel Ronen", elem_id="subtitle")
    gr.Markdown("[Model: dicta-il/dictalm2.0](https://huggingface.co/dicta-il/dictalm2.0)", elem_id="model_link")
    
    chatbot = gr.Chatbot(elem_id="chatbot")
    
    with gr.Row():
        message = gr.Textbox(placeholder="Type your message...", label="User", elem_id="message")
        submit = gr.Button("Send")

    with gr.Row():
        create_paragraphs_checkbox = gr.Checkbox(label="Create Paragraphs", value=False)
        clear_chat_btn = gr.Button("Clear Chat")
        copy_last_btn = gr.Button("Copy Last Response")
    
    with gr.Accordion("Adjustments", open=False):
        with gr.Row():    
            with gr.Column():
                max_new_tokens = gr.Slider(minimum=10, maximum=1500, value=360, step=10, label="Max New Tokens")
                min_length = gr.Slider(minimum=10, maximum=300, value=100, step=10, label="Min Length")
                no_repeat_ngram_size = gr.Slider(minimum=1, maximum=6, value=4, step=1, label="No Repeat N-Gram Size")
            with gr.Column():
                num_beams = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Num Beams") 
                temperature = gr.Slider(minimum=0.1, maximum=1.9, value=0.5, step=0.1, label="Temperature")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top P")
                top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top K")
        early_stopping = gr.Checkbox(value=True, label="Early Stopping")
        custom_system_prompt_enabled = gr.Checkbox(value=False, label="Custom System Prompt")
        custom_system_prompt = gr.Textbox(placeholder="Enter your custom system prompt...", label="Custom System Prompt", lines=5, elem_id="custom_system_prompt")
        
        with gr.Row():
            copy_system_prompt_btn = gr.Button("Copy System Prompt")
            prompt_name = gr.Textbox(placeholder="Write Prompt Name")
            save_system_prompt_btn = gr.Button("Save System Prompt")
            prompt_dir = "system_prompts"
            os.makedirs(prompt_dir, exist_ok=True)
            prompt_dropdown = gr.Dropdown(label="Select a system prompt", choices=os.listdir(prompt_dir))
            load_system_prompt_btn = gr.Button("Load System Prompt")

    submit.click(chat, inputs=[message, chatbot, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, create_paragraphs_checkbox, custom_system_prompt_enabled, custom_system_prompt], outputs=[chatbot, chatbot, message])
    message.submit(submit_on_enter, inputs=[message, chatbot, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k, create_paragraphs_checkbox, custom_system_prompt_enabled, custom_system_prompt], outputs=[chatbot, chatbot, message])
    clear_chat_btn.click(clear_chat, inputs=[chatbot, message], outputs=[chatbot, message])
    copy_last_btn.click(copy_last_response, inputs=chatbot, outputs=message)
    copy_system_prompt_btn.click(copy_system_prompt, outputs=custom_system_prompt)
    save_system_prompt_btn.click(save_system_prompt, inputs=[custom_system_prompt, prompt_name], outputs=None)
    save_system_prompt_btn.click(lambda: prompt_dropdown.update(choices=os.listdir(prompt_dir)), None, prompt_dropdown)
    load_system_prompt_btn.click(load_system_prompt, inputs=prompt_dropdown, outputs=custom_system_prompt)
    
    demo.css = """
        #message, #message * {
            text-align: right !important;
            direction: rtl !important;
        }
        
        #chatbot, #chatbot * {
            text-align: right !important;
            direction: rtl !important;
        }
        
        #title, .label {
            text-align: right !important;
        }
        
        #subtitle {
            text-align: left !important;
        }
        
        #model_link {
            text-align: left !important;
        }
        
        #custom_system_prompt, #custom_system_prompt * {
            text-align: right !important;
            direction: rtl !important;
        }
    """

demo.launch()