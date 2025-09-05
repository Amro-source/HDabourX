from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

chat_history = []  # Stores chat messages

@app.route('/')
def index():
    return render_template('index.html', chat_history=chat_history)

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json['message']
    # Placeholder bot response (replace with xAI API call for real AI)
    bot_message = f"Oh darling, that sounds amazing! You said: '{user_message}'. What else is on your mind? 💕"
    chat_history.append({'sender': 'user', 'message': user_message})
    chat_history.append({'sender': 'bot', 'message': bot_message})
    return jsonify({'status': 'ok', 'bot_message': bot_message})

if __name__ == '__main__':
    app.run(debug=True)