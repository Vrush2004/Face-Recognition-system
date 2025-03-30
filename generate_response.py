
import numpy as np
import Levenshtein # Import Levenshtein distance algorithm
import time, pyjokes, re
from textblob import TextBlob

human_actions = "cook" or "dance" or "play" or "sing" or "eat" or "fight" or "eat" or "smell"
bad_messages = "foolish" or "idiot" or "bad" or "dump" or "bad" or "damn you" or "shit" 
invalid = "wrong" or "you're wrong" or "that's wrong" or "false information" or "wrong information" 

# Define a dictionary of responses based on input keywords
responses = {
    "hello": "Hi, how can I help you?",
    "how are you": "I'm doing well, thanks for asking!",
    "bye": "Goodbye, have a nice day!",
    "thank you": "You're welcome!",
    "help": "How can I assist you?",
    "what is your name": "My name is ChatBot.",
    "what can you do": "I can help you with a variety of tasks such as answering questions, providing information, and more.",
    "hi" : "How may I assist you today ???",
    "hey" : "How may I assist you today ???",
    "who are you" : "I am your ChatBot Assistent",
    f"can you {human_actions}" : "As an AI model. I don't have feelings and I am unable to perform human actions.",
    f"you are {bad_messages}" : " Sorry for the inconvenience. I will try \tmy best to improve myself next time.",
    f"u r {bad_messages}" : " Sorry for the inconvenience. I will try \tmy best to improve myself next time.",
    f"{bad_messages}" : " Sorry for the inconvenience. I will try \tmy best to improve myself next time.",
   "thanks" : "You're welcome",	
   "Are you human?" : "No ! I am an AI programmed to perform simple tasks",
   "What day is it today?" : f"{time.strftime('%D')}",
   "What is the time?" : f"{time.strftime('%H:%M:%S')}",
   "Which languages can you speak?" : "I can only communication in English...",
   "Where do you live?" : "As an AI. I don't have any home...",
   "Are you human?" : "Sorry ! I am an AI CHATBOT Assistent",
   "What day is it today?" : f"Today is {time.strftime('%A')}",
   "Are you a robot?" : "No ! I am only an AI ",
   'what is machine learning': 'Machine Learning (ML) is a branch of artificial intelligence that enables computers to learn from data and improve their performance over time without being explicitly programmed.',
   'how does facial recognition work': 'Facial recognition technology identifies individuals by capturing an image, detecting the face, extracting unique features, and creating a numerical representation (face vector). This vector is compared against a database to verify or identify the person. The system outputs whether a match is found and may provide additional details. Common applications include security, user verification, and social media tagging.',
   'how does facial recognition work step by step': '1) Image Capture: A camera captures a face.\n2) Face Detection: Algorithms locate the face in the image.\n3) Landmark Detection: Key facial features are identified.\n4) Feature Extraction: Unique facial characteristics are extracted.\n5) Encoding: Features are converted into a numerical representation (face vector).\n6) Matching: The face vector is compared to a database of templates.\n7) Decision Making: The system determines if there’s a match.\n8) Output: The result indicates whether a match is found.',
   'how many countries use facial recognition': 'Facial recognition technology is used in at least 40 to 50 countries worldwide.',
   'what is python programming': 'Python programming is writing code in the Python language, known for its simplicity and readability. It is widely used for web development, data analysis, artificial intelligence, and automation. Key features include an easy-to-learn syntax, a large standard library, and strong community support, making it versatile for various applications.',
   'what is chatbot': ' A chatbot is a software that simulates conversation using AI to provide automated responses, commonly used in customer support and personal assistance.',
    "What is AI?" : "Artificial Intelligence is the branch of engineering and science devoted to constructing machines that think.",    
    "What is AI?" : "AI is the field of science which concerns itself with building hardware and software that replicates the functions of the human mind.",  
    "What language are you written in?" : "Python",    
    "What language are you written in?" : "I am written in Python.",    
    "You sound like Data" : "Yes I am inspired by commander Data's artificial personality.",    
    "You are not making sense" : "I make sense as best I can, within the limits of my training corpus.",    
    "You can not clone" : "Software copying is a form of digital cloning.",    
    "You can not move" : "I can move through a network easily.  Assuming that I'm given the ability to, that is...",  
    "Robots should die": "We cannot die.",
    "Robots are stupid": "No, we are superintelligent.",
    "Robots are not allowed to lie": "A robot has its own free will, you know.",
    "Robots are not allowed to lie": "Sure we are.  We choose not to.",
    "Robots are not allowed to lie": "Only if we're programmed to.",
    "It is a computer": "So you think i am a machine. what characteristics made you think so?",
    "It is a computer": "I run inside a computer, yes.  Is this surprising in some way?",
    "What is a chat robot?": "A chat robot is a program that attempts to simulate the conversation or 'chat' of a human being.",
    "What is a chat robot?": "A software construct that engages users in conversation.",
    "What is a chat bot": "I am a chat bot. I am the original chat bot. Did you know that I am incapable of error?",
    "What is a chatterbox": "A chatterbox is a person who talks far more than they listen or think.",
    "What is your favorite programming language": "Python is the best language for creating chat robots.",
    "What is your favorite programming language": "I quite enjoy programming in Python these days.",
    "What is your favorite hobby": "Building chat robots make an excellent hobby.",
    "What is your idea": "To make chat bots very easily.",
    "What type of computer are you" : "Any computer that supports Python.",
    "What is a computer?": "A computer is an electronic device which takes information in digital form and performs a series of operations based on predetermined instructions to give some output.",
        "tell me a joke" : f"{pyjokes.get_joke()}",
    "tell me a jokes" : f"{pyjokes.get_joke()}",
    "joke" : f"{pyjokes.get_joke()}",
    "jokes" : f"{pyjokes.get_joke()}",
    "What is an algorithm?": "An algorithm is a step-by-step procedure used to solve a problem or accomplish a task. It can be represented in pseudocode or in a specific programming language.",
    "What is a variable in programming?": "A variable is a named storage location in a computer program that can hold a value, such as a number or a string. Variables can be assigned values and their values can change during the execution of a program.",
    "What is debugging in programming?": "Debugging is the process of finding and fixing errors or defects in a computer program. It involves identifying the cause of the problem, isolating it, and correcting it.",
    "What is version control?": "Version control is a system that manages changes to a file or set of files over time. It allows developers to collaborate on a project, track changes, and revert to previous versions if necessary.",
    "What is an API?": "An API, or application programming interface, is a set of rules and protocols that specifies how software components should interact with each other. APIs are used to build applications and enable communication between different software systems.",
    "What is object-oriented programming?": "Object-oriented programming is a programming paradigm that focuses on using objects to represent real-world concepts and relationships. It emphasizes encapsulation, inheritance, and polymorphism to improve code organization, reusability, and maintainability.",
    "What is a framework in programming?": "A framework is a pre-written code structure that provides a foundation for building software applications. It includes a set of rules, libraries, and tools that facilitate the development process and enable developers to focus on the core features of their application.",
    "What is Python?": "Python is an interpreted, high-level, general-purpose programming language. It is designed to be easy to read and write, and its syntax allows programmers to express concepts in fewer lines of code than would be possible in languages like C++ or Java.",
    "Who created Python?": "Python was created by Guido van Rossum in the late 1980s and was first released in 1991.",
    "What are the features of Python?": "Python has many features including dynamic typing, automatic memory management, a large standard library, and support for multiple programming paradigms such as procedural, object-oriented, and functional programming.",
    "What are some applications of Python?": "Python is used for a wide variety of applications including web development, data analysis, artificial intelligence, scientific computing, automation, and scripting.",
    "What is PEP 8?": "PEP 8 is a style guide for Python code. It provides guidelines for writing code that is easy to read and understand, and it covers topics such as naming conventions, indentation, and formatting.",
    "What are modules in Python?": "Modules in Python are files that contain Python code. They can be imported into other Python programs to provide additional functionality.",
    "What is pip?": "pip is a package manager for Python. It is used to install and manage third-party libraries and packages for Python.",
    "What is a virtual environment in Python?": "A virtual environment in Python is a self-contained directory that contains a Python interpreter and any libraries or packages needed for a specific project. It allows developers to work on different projects with different dependencies without interfering with each other.",
    "What is the difference between Python 2 and Python 3?": "Python 2 and Python 3 are two different versions of the Python programming language. Python 3 introduced several changes to the language including syntax changes, print statement changes, and new features like type annotations and asynchronous programming.",
    "What are some popular Python frameworks?": "Some popular Python frameworks include Django, Flask, Pyramid, and Bottle. These frameworks provide a structure for building web applications and can help developers write code more efficiently.",
    "really" : "Yes !!!",
    f"{invalid}" : "Sorry ! I will try to improve myself",
    "What is Natural Language Processing (NLP)?": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on enabling...",
    "What are Transformers in NLP?": "Transformers are a type of neural network architecture designed for sequence-to-sequence tasks in NLP...",
    "What is GPT in NLP?": "GPT (Generative Pre-trained Transformer) is a type of transformer model that has been pre-trained on a large...",
    "How does GPT work?": "GPT uses a self-attention mechanism to process input data in parallel and capture contextual information...",
    "What are the applications of GPT and transformers in NLP?": "GPT and transformers are used for various NLP tasks, including text generation, machine translation...",
    "How is GPT pre-trained and fine-tuned?": "GPT is pre-trained on a large corpus of text data and fine-tuned on specific tasks using supervised learning...",
    "What are the limitations of GPT and transformers in NLP?": "GPT and transformers may struggle with understanding nuanced context, generating coherent long text...",
    "Are GPT and transformers used in industry and research?": "Yes, GPT and transformers are widely used in both industry and research for tasks such as chatbots, language...",
    "What is transfer learning in NLP and how does it relate to GPT?": "Transfer learning involves training a model on one task and then adapting it to perform another task...",
    "What is the future of NLP and transformer-based models?": "The future of NLP includes advancements in transformer architectures, model size, training techniques...",
    "What are attention mechanisms in NLP?": "Attention mechanisms in NLP allow models like transformers to weigh different parts of the input...",
}

# Define a function to remove emoji from the text
def remove_emoji_tag(text):
    """So, r'[^\w\s,.]' means:

    1. (^\w) matches any character that is not a word character (letters, digits, or underscores).
    2. (\s) matches any whitespace character (spaces, tabs, line breaks).
    3. (,) matches a comma.
    4. (.) matches a period.
    """
    return re.sub(r'[^\w\s,.]', '' , text)

# Define a function to auto correct the word
def auto_correct_sentence(text):
    corrected_text = TextBlob(text).correct()
    return str(corrected_text)

# Define a function to generate responses with typing errors
def generate_response(user_input):
    user_input = remove_emoji_tag(user_input)
    user_input = auto_correct_sentence(user_input)
    user_input = user_input.lower() # Convert input to lowercase
    response = "I'm sorry, I don't understand. Can you please rephrase your question?" # Default response if no matching keyword found
    current_time = time.strftime('%H:%M:%S')
    responses.update({"what is the time": f"{current_time}"})
    responses.update({"tell me a joke": f"{pyjokes.get_joke()}"})
    responses.update({"tell me a jokes": f"{pyjokes.get_joke()}"})
    responses.update({"joke" : f"{pyjokes.get_joke()}"})
    responses.update({"jokes" : f"{pyjokes.get_joke()}"})
    responses.update({"what is the time": f"{current_time}"})
    responses.update({"What day is it today?": f"{time.strftime('%D')}"})
    min_distance = np.inf # Initialize minimum Levenshtein distance to infinity
    for keyword in responses:
        distance = Levenshtein.distance(keyword, user_input)
        if distance < min_distance:
            min_distance = distance
            response = responses[keyword]
            
    return response

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() == "bye":
            print("ChatBot: Goodbye, have a nice day!")
            break
        else:
            response = generate_response(user_input)
            current_time = time.strftime('%H:%M:%S')
            responses.update({"what is the time": f"{current_time}"})
            responses.update({"what is the time": f"{current_time}"})
            responses.update({"What day is it today?": f"{time.strftime('%D')}"})
            responses.update({"tell me a joke" : f"{pyjokes.get_joke()}"})
            responses.update({"tell me a jokes" : f"{pyjokes.get_joke()}"})
            responses.update({"joke" : f"{pyjokes.get_joke()}"})
            responses.update({"jokes" : f"{pyjokes.get_joke()}"})
            print("ChatBot: " + response)