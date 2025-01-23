# Assignment: Natural Language Processing (NLP) Concepts and Applications

---

## **Part 1: Text Preprocessing and Bag-of-Words Representation**

1. **Text Tokenization and Preprocessing**

   - Use the NLTK library to tokenize the following text into words:
     ```
     "Natural Language Processing (NLP) is a fascinating field that combines linguistics, computer science, and artificial intelligence."
     ```
   - Remove stopwords and punctuation from the tokenized text.

1. **Bag-of-Words Representation**

   - Create a Bag-of-Words (BoW) representation for the following set of documents using `scikit-learn`'s `CountVectorizer`:
     ```
     Document 1: "I love NLP and machine learning."
     Document 2: "NLP is a key part of artificial intelligence."
     Document 3: "Machine learning and NLP are transforming industries."
     ```
   - Print the vocabulary and the BoW matrix.

---

## **Part 2: Language Modeling and Text Classification**

3. **Unigram Language Model**

   - Build a Unigram language model using the following corpus:
     ```
     ["I love NLP", "NLP is fun", "I enjoy learning NLP"]
     ```
   - Calculate the probability of the sentence: `"I love learning NLP"` using the Unigram model.

1. **Text Classification with Naive Bayes**

   - Use the `scikit-learn` library to train a Naive Bayes classifier on the following labeled dataset:
     ```
     Texts: ["I love this product", "This is a great movie", "I hate this weather", "The service was terrible"]
     Labels: ["positive", "positive", "negative", "negative"]
     ```
   - Predict the sentiment of the following text: `"The movie was fantastic"`.

---

## **Part 3: Sentiment Analysis**

5. **Rule-Based vs. Model-Based Sentiment Analysis**
   - Use a sentiment lexicon (e.g., NLTK's VADER) to analyze the sentiment of the following text:
     ```
     "The weather is nice, but the traffic is terrible."
     ```
   - Compare the results with a model-based approach (e.g., using the Naive Bayes classifier from Part 2).

---

## **Part 4: Dependency and Constituency Parsing**

6. **Dependency Parsing**

   - Use a dependency parsing library (e.g., spaCy) to parse the following sentence:
     ```
     "The cat sat on the mat."
     ```
   - Extract and print the dependency relationships between the words.

1. **Constituency Parsing**

   - Use a constituency parsing tool (e.g., NLTK's `nltk.RegexpParser`) to parse the same sentence:
     ```
     "The cat sat on the mat."
     ```
   - Compare the results with the dependency parsing output.

---

## **Part 5: Named Entity Recognition (NER) and Information Extraction**

8. **Named Entity Recognition**

   - Use a pre-trained NER model (e.g., spaCy) to extract named entities from the following text:
     ```
     "Apple Inc. is planning to open a new store in San Francisco next month."
     ```
   - Print the extracted entities and their labels.

1. **Rule-Based Information Extraction**

   - Write a Python program to extract dates from the following text using regular expressions:
     ```
     "The event is scheduled for 2023-10-15, and the deadline is 2023-09-30."
     ```

---

## **Part 6: Machine Translation and Text Generation**

10. **Machine Translation**

    - Use a parallel dataset (e.g., from OPUS) to translate the following English sentence into Arabic:
      ```
      "The weather is beautiful today."
      ```

1.  **Text Generation**

    - Write a Python program to generate text using pre-trained language models (e.g., GPT-2) to generate the completion of the following prompt:
      ```
      "Wealth, fame, power",
      "It all started with a dream",
      "Kids who's never known peace have different"
      ```

---

## **Part 7: Neural Networks for Sentiment Analysis**

12. **Sentiment Analysis with RNN**
    - Implement a simple RNN (e.g., LSTM or GRU) using a deep learning framework (e.g., TensorFlow or PyTorch) to classify the sentiment of the following dataset:
      ```
      Texts: ["I love this product", "This is a great movie", "I hate this weather", "The service was terrible"]
      Labels: ["positive", "positive", "negative", "negative"]
      ```
    - Train the model and evaluate its performance on a test set.

---

## **Part 8: Word Embeddings**

13. **Word2Vec and GloVe**

    - Train a Word2Vec model on the following corpus:
      ```
      ["I love NLP", "NLP is fun", "I enjoy learning NLP"]
      ```
    - Use the trained model to find the most similar words to `"NLP"`.

1.  **Comparing Word Embeddings**

    - Compare the performance of Word2Vec and GloVe embeddings on a simple NLP task (e.g., word similarity or analogy).
