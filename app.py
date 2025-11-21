from flask import Flask, render_template, request
from LLM_QA_CLI import get_llm_response, preprocess_input
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = None
    processed_question = None
    
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            processed_question = preprocess_input(question)
            answer = get_llm_response(processed_question)
    
    return render_template('index.html', question=question, processed_question=processed_question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
