<h1 style="text-align: center;">المحاضرة العملية رقم: 2</h1>

## أولا: مفردات المقرر حسب التوصيف

تحليل المشاعر باستخدام التعلم الآلي والمعاجم اللغوية.

---

## ثانياً: المحتوى

### العنوان: تحليل المشاعر باستخدام خوارزميات تعلم الآلة والمعاجم اللغوية

---

### الموضوع الرئيسي من توصيف النظري:

التعرف على كيفية تحليل مشاعر النصوص باستخدام مجموعات بيانات معنونة ومعاجم المشاعر.

---

### أهداف المعمل

يسعى التطبيق العملي لتحقيق الأهداف التالية:

1. استخدام مجموعة بيانات معنونة تحتوي على النصوص وتصنيفات المشاعر لتدريب نماذج تعلم الآلة.
1. استخدام معاجم المشاعر لتحليل النصوص بناءً على الكلمات وتقييم نتائج التحليل.
1. فهم الفرق بين تحليل المشاعر المعتمد على القواعد والمعتمد على النماذج.

---

### الأدوات

- جهاز حاسوب.
- Python.
- مكتبة NLTK.
- مكتبة scikit-learn وpandas.
- مكتبة AFINN-111 أو SentiWordNet.

---

### المحتوى

- **المسألة/المسائل:**

  1. استخدام مجموعة بيانات معنونة (مثل تقييمات نصية مع مشاعر إيجابية أو سلبية) لتدريب نموذج تعلم آلي لتحليل المشاعر.
  1. استخدام معاجم المشاعر، مثل AFINN-111 أو SentiWordNet، لتحليل النصوص واستنتاج مشاعرها.

- **التطبيقات**

1. البرنامج الأول: تدريب نموذج تحليل مشاعر باستخدام مجموعة بيانات معنونة

الشفرة

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset of movie reviews and their sentiment labels
data = {
    "review": [
        "This movie is amazing!",
        "I did not like this movie at all.",
        "It was a fun and enjoyable experience.",
        "The quality was terrible.",
        "The movie was average."
    ],
    "sentiment": ["positive", "negative", "positive", "negative", "negative"]
}
df = pd.DataFrame(data)

# Convert the text data into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

المخرجات: عرض دقة النموذج في تصنيف النصوص إلى إيجابية وسلبية.

---

2. البرنامج الثاني: تحليل النصوص باستخدام معجم المشاعر

الشفرة

```python
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk

nltk.download('sentiwordnet')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Convert words to a form compatible with SentiWordNet
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Analyze sentiment using SentiWordNet
def analyze_sentiment(text):
    tokens = word_tokenize(text)
    sentiment_score = 0
    for token in tokens:
        pos = get_wordnet_pos(token)
        synsets = list(swn.senti_synsets(token, pos))
        if synsets:
            sentiment_score += synsets[0].pos_score() - synsets[0].neg_score()
    return "positive" if sentiment_score > 0 else "negative"

# Test sentences
text = "This movie was great but a bit long."
print("Text:", text)
print("Sentiment:", analyze_sentiment(text))
```

المخرجات: تحليل النصوص بناءً على الكلمات باستخدام SentiWordNet واستنتاج إذا كانت المشاعر إيجابية أو سلبية.

---

### التكاليف

1. توسيع مجموعة البيانات المعنونة:
   1. جمع 10 تقييمات نصية إضافية وتصنيفها كمشاعر إيجابية أو سلبية.
   1. تدريب النموذج على المجموعة الجديدة واختبار دقته.
1. استخدام معجم AFINN-111 لتحليل النصوص:
   1. تحميل واستخدام معجم AFINN-111 لتحليل نصوص إضافية.
   1. مقارنة نتائج AFINN-111 مع SentiWordNet وتوثيق الفرق.
