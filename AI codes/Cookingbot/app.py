from flask import Flask, render_template, request, jsonify
import random
import re

app = Flask(__name__)

chat_history = []  # Stores chat messages

# Creative cooking responses
cooking_responses = [
    "Oh darling, that sounds amazing! Let me suggest adding a pinch of smoked paprika for that extra wow factor! 💕",
    "What a fantastic idea! Have you considered garnishing with fresh herbs for a pop of color and flavor? 🌿",
    "Yum! That dish would pair beautifully with a crisp white wine or a citrusy craft beer! 🍷",
    "I love that! For an extra special touch, try roasting your spices first to deepen their flavor profile! 🔥",
    "Delicious choice! A squeeze of fresh lemon juice right before serving will brighten up those flavors! 🍋",
    "Mmm, my virtual taste buds are tingling! That would be perfect with some homemade garlic bread on the side! 🍞",
    "Heavenly! If you're feeling adventurous, a dash of chili flakes could add a nice kick! 🌶️",
    "Scrumptious! Have you thought about adding some toasted nuts for a delightful crunch? 🥜"
]

cooking_tips = [
    "Always taste as you cook - your palate is your best guide! 👩‍🍳",
    "Don't overcrowd the pan - it steams food instead of browning it! 🍳",
    "Salt your pasta water generously - it should taste like the sea! 🌊",
    "Let your meat rest after cooking - it keeps all those delicious juices inside! 🥩",
    "Sharp knives are safer than dull ones - they require less pressure! 🔪",
    "Bring dairy ingredients to room temperature before baking for better emulsion! 🧈",
    "Toast your spices in a dry pan to unlock their full aromatic potential! 🌿",
    "Pat proteins dry before seasoning - you'll get a much better sear! 🍗"
]

ingredient_substitutions = {
    r'\bbutter\b': ['coconut oil', 'olive oil', 'applesauce (in baking)', 'avocado'],
    r'\bmilk\b': ['almond milk', 'oat milk', 'coconut milk', 'soy milk'],
    r'\beggs?\b': ['flax eggs (1 tbsp ground flax + 3 tbsp water)', 'mashed banana', 'applesauce',
                   'commercial egg replacer'],
    r'\bflour\b': ['almond flour', 'oat flour', 'coconut flour', 'whole wheat flour'],
    r'\bsugar\b': ['honey', 'maple syrup', 'coconut sugar', 'stevia'],
    r'\bcream\b': ['coconut cream', 'cashew cream', 'Greek yogurt', 'pureed silken tofu']
}

recipe_ideas = {
    r'chicken': ["How about a creamy chicken marsala?", "Lemon herb roasted chicken is always a crowd-pleaser!",
                 "Chicken piccata with pasta would be delicious!"],
    r'pasta': ["A simple aglio e olio is my comfort food favorite!",
               "Have you tried making homemade pasta? It's easier than you think!",
               "Pasta carbonara is classic for a reason!"],
    r'vegetable': ["Roasted vegetables with herbs are always a hit!",
                   "A colorful stir-fry would showcase those veggies beautifully!",
                   "How about a vegetable gratin with a cheesy topping?"],
    r'chocolate': ["Molten chocolate lava cakes are impressive yet simple!",
                   "A rich chocolate mousse never fails to delight!",
                   "Chocolate-dipped fruit makes an elegant dessert!"],
    r'fish': ["Pan-seared fish with a lemon butter sauce is divine!", "Fish tacos with fresh slaw would be fantastic!",
              "A simple baked fish with herbs is light and healthy!"]
}


@app.route('/')
def index():
    return render_template('index.html', chat_history=chat_history)


@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json['message'].lower()
    chat_history.append({'sender': 'user', 'message': user_message})

    # Check for specific patterns in the user's message
    bot_message = generate_response(user_message)

    chat_history.append({'sender': 'bot', 'message': bot_message})
    return jsonify({'status': 'ok', 'bot_message': bot_message})


def generate_response(user_message):
    # Check for greeting
    if re.search(r'\b(hi|hello|hey|howdy|hola)\b', user_message):
        return random.choice([
            "Hello, culinary adventurer! What delicious creation are we cooking up today? 🍳",
            "Hey there, chef! What's cooking in your kitchen? 👩‍🍳",
            "Hi! Ready to whip up something amazing? What's on your menu? 🥘"
        ])

    # Check for help request
    if re.search(r'\b(help|what can you do|assist)\b', user_message):
        return "I can help with recipe ideas, cooking tips, ingredient substitutions, and general culinary inspiration! What would you like to explore today? 🧑‍🍳"

    # Check for thank you
    if re.search(r'\b(thanks|thank you|appreciate)\b', user_message):
        return random.choice([
            "You're most welcome! Happy cooking! 🥄",
            "Anytime! Can't wait to hear how your culinary creation turns out! 🌟",
            "My pleasure! Remember, the secret ingredient is always love! ❤️"
        ])

    # Check for ingredient substitution requests
    for ingredient, substitutes in ingredient_substitutions.items():
        if re.search(ingredient, user_message):
            return f"Looking to substitute {ingredient.strip('\\\\b')}? Try: {', '.join(substitutes[:3])}! 🥗"

    # Check for recipe ideas based on ingredients
    for ingredient, ideas in recipe_ideas.items():
        if re.search(ingredient, user_message):
            return f"I see you're working with {ingredient.strip('\\\\b')}. {random.choice(ideas)} 🍽️"

    # Check for cooking questions
    if re.search(r'\b(how|what|why|when|where)\b', user_message):
        return f"Great question! {random.choice(cooking_tips)}"

    # Default creative cooking response
    return f"{random.choice(cooking_responses)} {random.choice(cooking_tips)}"


if __name__ == '__main__':
    app.run(debug=True)