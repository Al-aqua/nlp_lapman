<h1 style="text-align: center;">المحاضرة العملية رقم: 7</h1>

## أولا: مفردات المقرر حسب التوصيف

الشبكات العصبية للمعالجة الطبيعية للغات (NLP).

---

## ثانيا: المحتوى

### العنوان: الشبكات العصبية للمعالجة الطبيعية للغات

---

### الموضوع الرئيسي من توصيف النظري:

فهم كيفية استخدام الشبكات العصبية لتطبيقات المعالجة الطبيعية للغات مثل تصنيف النصوص، وتحليل المشاعر، وإنشاء النصوص.

---

### أهداف المعمل

يسعى التطبيق العملي لتحقيق الأهداف التالية:

1. تطبيق نموذج لتحليل المشاعر باستخدام شبكة عصبية متكررة (RNN) مثل LSTM أو GRU.
1. كتابة برنامج يتفاعل مع نموذج لغوي لإنتاج نصوص منطقية ومتسقة بناءً على مدخلات من المستخدم.

---

### الأدوات

- جهاز حاسوب.
- Python.
- TensorFlow أو PyTorch.
- مكتبات numpy و pandas لمعالجة البيانات.
- NLTK لاستخراج المعاني اللغوية.

---

### المحتوى

- **المسألة/المسائل:**

  1. تنفيذ نموذج تحليل مشاعر باستخدام RNN بناءً على مكتبات التعلم العميق.
  1. كتابة برنامج يولد نصوصاً منطقية وموافقة للسياق بناءً على مدخلات المستخدم.

- **التطبيقات:**

1. البرنامج الأول: تنفيذ نموذج تحليل المشاعر باستخدام RNN

الشفرة:

```bash
pip install numpy pandas scikit-learn tensorflow
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
        "Absolutely fantastic!"
    ],
    "sentiment": [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative
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
```

المخرجات: عرض دقة النموذج في تحليل المشاعر.

---

2. البرنامج الثاني: إنشاء نظام يولد نصوصاً بناءً على مدخلات المستخدم

الشفرة:

```bash
pip install torch
```

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Generate text based on user input
prompt = "Once upon a time in a small village"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
```

المخرجات: نصوص منطقية ومتسقة تُنتج بناءً على المدخلات النصية.

---

### التكاليف

1. تحسين نموذج تحليل المشاعر:

   1. استخدام مجموعة بيانات أكبر وأكثر تنوعاً.
   1. تحسين معمارية النموذج باستخدام GRU بدلاً من LSTM.

1. توسيع نظام توليد النصوص:

   1. تجربة مدخلات متنوعة.
   1. مقارنة نتائج النماذج المختلفة مثل GPT-3 أو BERT.
