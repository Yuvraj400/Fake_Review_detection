import streamlit as st
import pickle
import re 

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


def clean_text(text):
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    return clean_text

def predict(text_input):
    text_input = clean_text(text_input)
    text_input = tfidf.transform([text_input])
    pred = model.predict(text_input)
    return pred[0]


def main():
    st.title('Fake Review detection')

    text = st.text_input('write your review')
    
    if st.button("Submit"):
        result = predict(text)
        if result == 1:
            st.write("True Review")
        else:
            st.write("Fake Review")


if __name__ == '__main__':
    main()