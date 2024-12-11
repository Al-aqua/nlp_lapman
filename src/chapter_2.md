<h1 style="text-align: center;">المحاضرة العملية رقم: 2</h1>

## أولا: مفردات المقرر حسب التوصيف

النمذجة اللغوية وتصنيف النصوص باستخدام Python.

---

## ثانياً: المحتوى

### العنوان: النمذجة اللغوية وتصنيف النصوص باستخدام Naive Bayes

---

### الموضوع الرئيسي من توصيف النظري:

التعرف على مفهوم النماذج اللغوية وتصنيف النصوص باستخدام خوارزميات تعلم الآلة.

---

### أهداف المعمل

يسعى التطبيق العملي لتحقيق الأهداف التالية:

1. بناء نموذج لغوي Unigram لمعالجة النصوص وتحليلها.
1. إنشاء نموذج لتصنيف النصوص باستخدام خوارزمية Naive Bayes مع مكتبة scikit-learn.
1. تعزيز مهارات البرمجة وتحليل البيانات النصية باستخدام Python.

---

### الأدوات

1. جهاز حاسوب.
1. Python
1. مكتبة NLTK.
1. مكتبة scikit-learn.

---

### المحتوى

- **المسألة/المسائل:**

  1. بناء نموذج لغوي Unigram لمعالجة النصوص باستخدام Python.
  1. استخدام مكتبة scikit-learn لبناء نموذج لتصنيف النصوص باستخدام خوارزمية Naive Bayes.

- **التطبيقات**

1. البرنامج الأول: بناء نموذج لغوي Unigram

الشفرة

```python
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk

# Download the NLTK tokenizer models (if not already downloaded)
nltk.download('punkt')

# Sample corpus of English text
corpus = [
    "Artificial intelligence is fascinating.",
    "Machine learning is part of artificial intelligence.",
    "Python is a popular language for data analysis."
]

# Tokenize each document in the corpus using NLTK
tokens = []
for doc in corpus:
    tokens.extend(word_tokenize(doc))

# Count the frequency of words
unigram_counts = Counter(tokens)
total_words = sum(unigram_counts.values())

# Calculate the probability of each word
unigram_model = {word: count / total_words for word, count in unigram_counts.items()}

print("Unigram Language Model:")
for word, prob in unigram_model.items():
    print(f"{word}: {prob:.4f}")
```

المخرجات: عرض احتمالية كل كلمة بناءً على تكرارها في النصوص.

---

2. البرنامج الثاني: تصنيف النصوص باستخدام Naive Bayes

الشفرة

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset: Sentences and their corresponding labels (angry or happy)
documents = [
    "I am so mad at you!",
    "This is the best day ever!",
    "I can't believe this happened, I'm furious!",
    "I am extremely happy right now.",
    "I hate this situation!",
    "What a wonderful experience!",
    "I am very angry with the result!",
    "This is amazing, I feel great!",
    "I am so upset!",
    "I'm so excited for this event!"
]

# Labels: angry or happy
labels = ["angry", "happy", "angry", "happy", "angry", "happy", "angry", "happy", "angry", "happy"]

# Convert the text into numerical format using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Display predictions for test data
for i in range(len(X_test.toarray())):
    print(f"Sentence: '{documents[y_test.index(y_test[i])]}', Predicted: {y_pred[i]}")
```

المخرجات: عرض دقة النموذج ومدى قدرته على تصنيف النصوص.

---

### التكاليف

1. تطبيق نموذج Unigram على مجموعة بيانات جديدة:
   - جمع نصوص إضافية (مثل مقالات أو مراجعات منتجات).
   - بناء نموذج لغوي Unigram جديد، وشرح التوزيع الاحتمالي للكلمات.
1. تصنيف نصوص إضافية باستخدام Naive Bayes:
   - استخدام نموذج Naive Bayes لتصنيف نصوص جديدة.
   - توثيق دقة النموذج وتحليل الأخطاء الناتجة.
