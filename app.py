# Task 1: Import the Libraries
from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from spellchecker import SpellChecker
import contractions
import re
import string
import torch

# Initialize Flask App
app = Flask(__name__)

# Task 3a: Implement Next Word Suggestion -- Create Object
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Task 7a: Implement Spell Fix Functionality -- Create Object
spell = SpellChecker()


@app.route('/')
def index():
    return render_template('index.html')


# Task 3: Implement Next Word Suggestion
@app.route('/next_word', methods=['POST'])
def next_word():
    data = request.get_json()
    prompt = data.get('text', '')

    if not prompt.strip():
        return jsonify({'next_word': ''})

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )

    generated_sequence = output[0]
    generated_text = tokenizer.decode(generated_sequence)
    next_word_only = generated_text[len(prompt):].strip().split(' ')[0]

    return jsonify({'next_word': next_word_only})


# Task 5: Implement Sentence Completion
@app.route('/complete_sentence', methods=['POST'])
def complete_sentence():
    data = request.get_json()
    prompt = data.get('text', '')

    if not prompt.strip():
        return jsonify({'completed_sentence': ''})

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({'completed_sentence': generated_text})


# Task 7: Implement Spell Fix Functionality (Fix by Cursor)
@app.route('/fix_by_cursor', methods=['POST'])
def fix_by_cursor():
    data = request.get_json()
    text = data.get('text', '')

    words = re.findall(r'\b\w+\b', text)
    corrected_words = []

    for word in words:
        if word.lower() in spell:
            corrected_words.append(word)
        else:
            corrected_words.append(spell.correction(word) or word)

    corrected_text = ' '.join(corrected_words)
    return jsonify({'corrected_text': corrected_text})


# Task 9: Implement Fix-All Functionality
@app.route('/fix_all', methods=['POST'])
def fix_all():
    data = request.get_json()
    text = data.get('text', '')

    # Expand contractions
    expanded_text = contractions.fix(text)

    # Tokenize and correct
    words = re.findall(r'\b\w+\b', expanded_text)
    corrected_words = []

    for word in words:
        if word.lower() in spell:
            corrected_words.append(word)
        else:
            corrected_words.append(spell.correction(word) or word)

    corrected_text = ' '.join(corrected_words)
    return jsonify({'corrected_text': corrected_text})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
