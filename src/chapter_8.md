<h1 style="text-align: center;">المحاضرة العملية رقم: 8</h1>

## أولا: مفردات المقرر حسب التوصيف

مقدمة في الشبكات العصبية والتعلم العميق: التضمينات اللفظية (مثل Word2Vec و GloVe).

---

## ثانياً: المحتوى

### العنوان: الشبكات العصبية لمعالجة اللغات الطبيعية: التضمينات اللفظية

---

### الموضوع الرئيسي من توصيف النظري:

فهم أساسيات الشبكات العصبية والتعلم العميق وتطبيقها في معالجة اللغات الطبيعية، مع التركيز على التضمينات اللفظية.

---

### أهداف المعمل

يسعى التطبيق العملي لتحقيق الأهداف التالية:

1. التعرف على مفهوم التضمينات اللفظية وأهميتها في معالجة اللغات الطبيعية.
1. تطبيق نماذج Word2Vec و GloVe لإنشاء تضمينات لفظية.
1. استكشاف العلاقات الدلالية بين الكلمات باستخدام التضمينات اللفظية.
1. مقارنة أداء نماذج التضمين المختلفة في مهام معالجة اللغات الطبيعية.

---

### الأدوات

- جهاز حاسوب.
- Python.
- مكتبة gensim.
- مكتبة numpy.
- مكتبة matplotlib.
- مكتبة scikit-learn.
- مجموعة بيانات نصية (مثل مجموعة أخبار أو مقالات).

---

### المحتوى

- **المسألة/المسائل:**

  1. إنشاء تضمينات لفظية باستخدام نموذج Word2Vec.
  1. استخدام التضمينات اللفظية لاستكشاف العلاقات بين الكلمات.
  1. مقارنة أداء Word2Vec و GloVe في مهمة تصنيف النصوص.

- **التطبيقات**

1. البرنامج الأول: إنشاء تضمينات لفظية باستخدام Word2Vec

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import pandas as pd

# Load the data (example: news collection)
df = pd.read_csv('Articles.csv')
corpus = df['Article'].apply(simple_preprocess).tolist()

# Train Word2Vec model
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Save the model
model.save("word2vec.model")

# Extract similar words
similar_words = model.wv.most_similar('technology', topn=10)
print("Words similar to 'technology':")
for word, score in similar_words:
    print(f"{word}: {score}")
```

**المخرجات:** قائمة بالكلمات الأكثر تشابهًا مع كلمة "technology" بناءً على نموذج Word2Vec.

---

2. البرنامج الثاني: استكشاف العلاقات بين الكلمات

```python
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the saved model
model = KeyedVectors.load("word2vec.model")

# Choose some words for visualization
words = [
    'computer', 'technology', 'science', 'art', 'music',
    'programming', 'data'
]

# Extract embeddings
embeddings = np.array([model.wv[w] for w in words])

# Reduce dimensions using t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=0)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.title("Word Embeddings Visualization")
plt.savefig("word_embeddings_visualization.png")
```

**المخرجات:** رسم بياني يوضح العلاقات بين الكلمات المختارة في فضاء ثنائي الأبعاد.

---

### التكاليف

1. تجربة تقنيات مختلفة لتحسين جودة التضمينات اللفظية:

   1. تعديل معلمات نموذج Word2Vec (مثل حجم النافذة وحجم المتجه) وملاحظة التأثير على جودة التضمينات.
   1. استخدام تقنيات معالجة مسبقة مختلفة للنصوص وتقييم تأثيرها على التضمينات الناتجة.

1. تطبيق التضمينات اللفظية في مهمة أخرى لمعالجة اللغات الطبيعية:

   1. استخدام التضمينات اللفظية في مهمة تحليل المشاعر.
   1. مقارنة أداء النموذج باستخدام التضمينات اللفظية مع نموذج يستخدم تمثيلات نصية بسيطة (مثل bag-of-words).

1. استكشاف نماذج تضمين متقدمة:

   1. تجربة نموذج FastText وتقييم أدائه مقارنة بـ Word2Vec و GloVe.
   1. البحث عن نماذج تضمين أحدث (مثل BERT أو ELMo) وتقديم تقرير موجز عن مزاياها وكيفية استخدامها.
