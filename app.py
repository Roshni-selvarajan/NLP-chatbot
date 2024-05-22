import streamlit as st
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Text preprocessing functions
nltk.download('punkt')
nltk.download('wordnet')
lemma = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemma.lemmatize(token) for token in tokens]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Define greeting function
greet_inputs = ('hello', 'hi', 'how are you?', 'whatsup')
greet_responses = ('hi', 'Hey', 'Hey There', 'There there!!')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

# Response generation function
def response(user_response, sentence_tokens):
    robo1_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens + [user_response])
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]
    if req_tfidf == 0:
        robo1_response = "I am sorry. Unable to understand you!"
    else:
        robo1_response = sentence_tokens[idx]
    return robo1_response

# Streamlit app
def main():
    st.title('NLP Chatbot')

    st.write("Upload a text file containing your corpus:")
    uploaded_file = st.file_uploader("Choose a file", type=['txt'], key='file_uploader1')

    if uploaded_file is not None:
        corpus_text = uploaded_file.read().decode('utf-8')
        st.write("Corpus successfully loaded.")
        
        # Text preprocessing
        sentence_tokens = nltk.sent_tokenize(corpus_text)
        word_tokens = nltk.word_tokenize(corpus_text)
        st.write("Performing Text-Preprocessing Steps...")
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = ["Bot: Hello! I am the Retrieval Learning Bot. Start typing your text after greeting to talk to me. For ending convo type bye!"]

        user_response = st.text_input('You:', '', key='user_input')
        user_response = user_response.lower()

        if user_response != 'bye' and user_response:
            st.session_state.conversation_history.append('You: ' + user_response)
            if user_response in ['thank you', 'thanks']:
                st.session_state.conversation_history.append('Bot: You are Welcome..')
            else:
                if greet(user_response) is not None:
                    st.session_state.conversation_history.append('Bot: ' + greet(user_response))
                else:
                    bot_response = response(user_response, sentence_tokens)
                    st.session_state.conversation_history.append('Bot: ' + bot_response)
        elif user_response == 'bye':
            st.session_state.conversation_history.append('You: ' + user_response)
            st.session_state.conversation_history.append('Bot: Goodbye!')

        for message in st.session_state.conversation_history:
            st.text(message)
    else:
        st.write("No file uploaded. Please upload a text file.")

if __name__ == '__main__':
    main()




# st.title('NLP ChatBot')
# st.write("Upload a text file containing your corpus:")
# st.file_uploader("Choose a file", type=['txt'])

