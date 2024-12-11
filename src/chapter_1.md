<h1 style="text-align: center;">المحاضرة العملية رقم: 1</h1>

## أولا: مفردات المقرر حسب التوصيف

معالجة النصوص باستخدام مكتبات Python.

---

## ثانياً: المحتوى

### العنوان: معالجة النصوص - تحليل الجمل والكلمات باستخدام مكتبات Python

---

### الموضوع الرئيسي من توصيف النظري:

تحليل النصوص ومعالجتها باستخدام مكتبات Python مثل NLTK وscikit-learn.

---

### أهداف المعمل

يسعى التطبيق العملي لتحقيق الأهداف التالية:

1. فهم كيفية تقسيم النصوص إلى كلمات باستخدام مكتبة NLTK.
1. إنشاء تمثيل كيس الكلمات (Bag-of-Words) لمجموعة مستندات باستخدام مكتبة scikit-learn.

---

### الأدوات

1. جهاز حاسوب.
1. Python
1. مكتبة NLTK.
1. مكتبة scikit-learn.

---

### المحتوى

- **المسألة/المسائل:**

  1. كتابة برنامج Python يأخذ جملة كمدخل ويستخدم مكتبة NLTK لتجزئة الجملة إلى كلمات منفصلة.
  1. إعطاء مجموعة من المستندات وإنشاء تمثيل كيس الكلمات باستخدام مكتبة scikit-learn في Python.

- **التطبيقات**

1. البرنامج الأول:

الشفرة

```python
import nltk
nltk.download('punkt')

# Input sentence
sentence = input("Enter a sentence: ")

# Tokenize the sentence into words
words = nltk.word_tokenize(sentence)
print("Tokenized words:")
print(words)
```

المخرجات: تُظهر الكلمات الفردية بعد تجزئة الجملة المُدخلة.

---

2. البرنامج الثاني:

الشفرة

```python
from sklearn.feature_extraction.text import CountVectorizer

# Input documents
documents = [
    "Artificial intelligence is fascinating.",
    "Natural language processing is a part of artificial intelligence.",
    "Python is a popular language for machine learning."
]

# Create the bag of words
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(documents)

# Display the result
print("Bag of Words representation:")
print(bag_of_words.toarray())
print("Unique words:")
print(vectorizer.get_feature_names_out())
```

المخرجات: تظهر المصفوفة التي تمثل تمثيل كيس الكلمات وقائمة الكلمات الفريدة.
