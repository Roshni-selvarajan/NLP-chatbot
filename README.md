## NLP Chatbot Project

### Overview
<img src="https://images.ctfassets.net/ukazlt65o6hl/1rWCLaP3w1iMUSkZsg9dG4/ab39b7646762b37b2296c07ae08182ff/MicrosoftTeams-image__55_.png?w=1366&h=704&q=70&fm=webp" width='800'>
This project demonstrates the implementation of a conversational chatbot using Natural Language Processing (NLP) techniques in Python. The chatbot is capable of understanding user queries and providing appropriate responses by leveraging machine learning models for text similarity.

### Key Features
- **NLP Techniques Utilized:**
  - Tokenization: Breaking down text into tokens (words or phrases).
  - Lemmatization: Reducing words to their base or root form.
  - TF-IDF (Term Frequency-Inverse Document Frequency): Used for text preprocessing and analysis to understand the importance of words in a document relative to a collection.

- **Chatbot Implementation:**
  - Designed a simple conversational chatbot using cosine similarity and TF-IDF vectorization.
  - Implemented handling for greetings and various user queries.
  - Responses generated based on the similarity of user input with pre-defined text representations using TF-IDF.

- **Python Programming and Machine Learning Integration:**
  - Developed the chatbot using Python and popular libraries such as NLTK (Natural Language Toolkit) and scikit-learn.
  - Demonstrated proficiency in data preprocessing, vectorization, and machine learning model integration.
  - Integrated machine learning concepts for text similarity to provide meaningful responses.


### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Roshni-selvarajan/NLP-chatbot.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the chatbot:
   ```bash
   python src/chatbot.py
   ```

### How It Works
The chatbot uses TF-IDF vectorization to represent user input and predefined responses. Cosine similarity is then computed between the vectorized user input and each response to determine the most relevant response. The chatbot handles greetings and common user queries based on similarity scores.

### Future Improvements
- Enhance the chatbot's understanding by integrating more sophisticated NLP models.
- Implement a more dynamic response generation system using neural networks.
- Explore user context and maintain conversation history for improved interaction.

### Acknowledgements
This project is inspired by the need for interactive conversational agents powered by NLP and machine learning techniques. Special thanks to NLTK and scikit-learn for providing essential tools and resources.

---

