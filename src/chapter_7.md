<h1 style="text-align: center;">المحاضرة العملية رقم: 7</h1>

## أولا: مفردات المقرر حسب التوصيف

الشبكات العصبية للمعالجة الطبيعية للغات (NLP).

______________________________________________________________________

## ثانيا: المحتوى

### العنوان: الشبكات العصبية للمعالجة الطبيعية للغات

______________________________________________________________________

### الموضوع الرئيسي من توصيف النظري:

فهم كيفية استخدام الشبكات العصبية لتطبيقات المعالجة الطبيعية للغات مثل تصنيف النصوص، وتحليل المشاعر.

______________________________________________________________________

### أهداف المعمل

يسعى التطبيق العملي لتحقيق الأهداف التالية:

تطبيق نموذج لتحليل المشاعر باستخدام شبكة عصبية متكررة (RNN) مثل LSTM أو GRU.

______________________________________________________________________

### الأدوات

- جهاز حاسوب.
- Python.
- TensorFlow أو PyTorch.
- مكتبات numpy و pandas لمعالجة البيانات.

______________________________________________________________________

### المحتوى

- **المسألة/المسائل:**

  1. تنفيذ نموذج تحليل مشاعر باستخدام RNN بناءً على مكتبات التعلم العميق.
  1. تصنيف النصوص باستخدام نموذج ثنائي الطبقات (Bi-LSTM) ونموذج ثنائي العصبية (RNN).

- **التطبيقات:**

1. البرنامج الأول: تنفيذ نموذج تحليل المشاعر باستخدام RNN

الشفرة:

```bash
pip install tensorflow pandas
```

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
data = {
    "text": [
        "I love this product!",
        "This is the worst service I've ever had.",
        "The experience was amazing.",
        "I do not recommend this at all.",
        "Absolutely fantastic!",
        "This place is horrible.",
        "I hate this place!",
        "The service was terrible.",
        "I do not recommend this at all.",
        "Absolutely fantastic!",
        "It was really good.",
    ],
    "sentiment": [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1]  # 1 for positive, 0 for negative
}
df = pd.DataFrame(data)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index

# Pad sequences
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = np.array(df["sentiment"])

# Build the RNN model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=16, input_length=max_length),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=2)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print("Accuracy:", accuracy)

# Test the model with new samples
test_texts = [
    "This is a fantastic product!",
    "I hate this place!",
    "The service was terrible.",
    "I do not recommend this at all.",
    "Absolutely fantastic!"
]

# Tokenize and pad the test samples
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_X = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Predict sentiments
predictions = model.predict(test_X)
for text, pred in zip(test_texts, predictions):
    print(f"Text: {text}\nPredicted Sentiment: {'Positive' if pred > 0.5 else 'Negative'}\n")
```

المخرجات: عرض دقة النموذج في تحليل المشاعر.

______________________________________________________________________

2. البرنامج الثاني: تصنيف النصوص باستخدام نموذج ثنائي الطبقات (Bi-LSTM)

الشفرة:

```bash
pip install tensorflow pandas
```

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# بيانات افتراضية
data = {
    "text": [
        "I love this product!",
        "This is the worst service I've ever had.",
        "The experience was amazing.",
        "I do not recommend this at all.",
        "Absolutely fantastic!",
        "This place is horrible.",
        "I hate this place!",
        "The service was terrible.",
        "I do not recommend this at all.",
        "Absolutely fantastic!",
        "It was really good.",
    ],
    "sentiment": [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1]  # 1 for positive, 0 for negative
}

# توكنيزة النصوص
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["text"])
sequences = tokenizer.texts_to_sequences(data["text"])
word_index = tokenizer.word_index

# تجهيز البيانات للإدخال
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = np.array(data["sentiment"])

# بناء النموذج
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=16, input_length=max_length),
    Bidirectional(LSTM(32)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(X, y, epochs=10, batch_size=2)

# تقييم النموذج
loss, accuracy = model.evaluate(X, y)
print("Accuracy:", accuracy)

# Test the model with new samples
test_texts = [
    "This is a fantastic product!",
    "I hate this place!",
    "The service was terrible.",
    "I do not recommend this at all.",
    "Absolutely fantastic!"
]

# Tokenize and pad the test samples
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_X = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Predict sentiments
predictions = model.predict(test_X)
for text, pred in zip(test_texts, predictions):
    print(f"Text: {text}\nPredicted Sentiment: {'Positive' if pred > 0.5 else 'Negative'}\n")

```

المخرجات: نموذج يمكنه توقع مشاعر النصوص باستخدام نموذج RNN ثنائي الاتجاه لزيادة دقة التنبؤ.

______________________________________________________________________

### التكاليف

1. تجربة نموذج LSTM ونموذج GRU لمعرفة أيهما يقدم أداءً أفضل على نفس البيانات.
1. مقارنة نتائج تصنيف المشاعر بين Bi-LSTM وRNN التقليدية.
