import random
import time
import pyjokes

class ChatbotResponse:
    def __init__(self):
        # Define possible responses
        self.responses = {
            "greeting": ["Hello!", "Hi there!", "Hey! How can I help you today?"],
            "goodbye": ["Goodbye!", "See you later!", "Take care!"],
            "thanks": ["You're welcome!", "Anytime!", "Happy to help!"],
            "how_are_you": ["I'm doing well, thanks for asking!", "I'm great, how about you?", "I'm good, ready to help!"],
            "day": [f"Today is {time.strftime('%A')}."],
            "time": [f"The current time is {time.strftime('%H:%M:%S')}."],
            "joke": [pyjokes.get_joke() for _ in range(5)],  # Generate 5 different jokes
            "ai": ["AI, or Artificial Intelligence, is a branch of science focused on creating systems that can perform tasks requiring human intelligence."],
            "python": ["Python is a high-level programming language known for its simplicity and versatility. It is used in web development, AI, data analysis, and more."],
            "machine_learning": ["Machine Learning (ML) is a branch of AI that enables computers to learn from data and improve over time without explicit programming."],
            "face_recognition": ["Facial recognition identifies individuals by analyzing unique features from a captured image and comparing it with a database."],
            "blockchain": ["Blockchain is a decentralized ledger technology used for secure and transparent record-keeping."],
            "knn": ["KNN classifies data points based on their proximity to other points in a dataset, using the majority vote of the nearest neighbors."],
            "socratic_method": ["The Socratic method is a teaching style based on asking questions to stimulate critical thinking and illuminate ideas."],
            "default": ["I'm sorry, I don't understand.", "Can you rephrase that?", "Could you clarify?"],
            "language":["Python","I am written in Python."],
        }

    def get_response(self, user_input):
        user_input = user_input.lower()

        if "hello" in user_input or "hi" in user_input:
            return random.choice(self.responses["greeting"])
        elif "bye" in user_input or "goodbye" in user_input:
            return random.choice(self.responses["goodbye"])
        elif "thanks" in user_input or "thank you" in user_input:
            return random.choice(self.responses["thanks"])
        elif "how are you" in user_input:
            return random.choice(self.responses["how_are_you"])
        elif "what day is it" in user_input or "day" in user_input:
            return random.choice(self.responses["day"])
        elif "what time is it" in user_input or "time" in user_input:
            return random.choice(self.responses["time"])
        elif "joke" in user_input or "tell me a joke" in user_input:
            return random.choice(self.responses["joke"])
        elif "what is ai" in user_input or "ai" in user_input:
            return random.choice(self.responses["ai"])
        elif "python" in user_input:
            return random.choice(self.responses["python"])
        elif "machine learning" in user_input:
            return random.choice(self.responses["machine_learning"])
        elif "face recognition" in user_input:
            return random.choice(self.responses["face_recognition"])
        elif "blockchain" in user_input:
            return random.choice(self.responses["blockchain"])
        elif "knn" in user_input or "nearest neighbor" in user_input:
            return random.choice(self.responses["knn"])
        elif "socratic method" in user_input:
            return random.choice(self.responses["socratic_method"])
        elif "What language are you written in?":
            return random.choice(self.responses["language"])
        else:
            return random.choice(self.responses["default"])


# Example usage:
if __name__ == "__main__":
    chatbot = ChatbotResponse()
    while True:
        user_input = input()
        if "bye" in user_input.lower():
            print("Bot: " + chatbot.get_response(user_input))
            break
        else:
            print("Bot: " + chatbot.get_response(user_input))
