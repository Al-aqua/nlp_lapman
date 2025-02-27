<h1 style="text-align: center;">المحاضرة العملية رقم: 5</h1>

## أولا: مفردات المقرر حسب التوصيف

التعرف على الكيانات المسماة واستخراج المعلومات.

______________________________________________________________________

## ثانياً: المحتوى

### العنوان: التعرف على الكيانات المسماة واستخراج المعلومات باستخدام بايثون

______________________________________________________________________

### الموضوع الرئيسي من توصيف النظري:

فهم كيفية استخراج الكيانات المسماة من النصوص (مثل الأسماء، المواقع، المؤسسات) واستخدام تقنيات استخراج المعلومات لتحليل النصوص.

______________________________________________________________________

### أهداف المعمل

يسعى التطبيق العملي لتحقيق الأهداف التالية:

1. استخدام مجموعة بيانات معنونة تحتوي على نصوص بها كيانات مسماة لتحليل النصوص.
1. كتابة برنامج يعتمد على القواعد لاستخراج معلومات محددة من النصوص المنظمة.
1. فهم كيفية تطبيق استخراج المعلومات في سيناريوهات حقيقية.

______________________________________________________________________

### الأدوات

- جهاز حاسوب.
- Python.
- مكتبة spaCy.
- مكتبة pandas.
- مجموعة بيانات معنونة تحتوي على كيانات مسماة.

______________________________________________________________________

### المحتوى

- **المسألة/المسائل:**

  1. استخدام مجموعة بيانات معنونة تحتوي على نصوص وكيانات مسماة (مثل أسماء الأشخاص، المواقع، المنظمات) لاستخراج الكيانات المسماة.
  1. كتابة برنامج يعتمد على القواعد لاستخراج أنواع محددة من المعلومات من مستندات نصية تحتوي على بيانات منظمة (مثل السير الذاتية، المقالات الإخبارية).

- **التطبيقات**

1. البرنامج الأول: استخراج الكيانات المسماة باستخدام مجموعة بيانات معنونة

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

```python
import spacy
from spacy.tokens import DocBin

# تحميل نموذج لغة spaCy
nlp = spacy.load("en_core_web_sm")

# مثال على نصوص تحتوي على كيانات مسماة
texts = [
    "Barack Obama was born in Hawaii.",
    "Apple is looking at buying a UK startup for $1 billion.",
    "Google's headquarters are located in Mountain View, California."
]

# معالجة النصوص واستخراج الكيانات المسماة
for text in texts:
    doc = nlp(text)
    print(f"Text: {text}")
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
```

**المخرجات:** استخراج الكيانات المسماة (مثل الأشخاص، المواقع، المنظمات) وعرض تصنيفاتها.

______________________________________________________________________

2. البرنامج الثاني: استخراج المعلومات باستخدام نهج قائم على القواعد

```python
import re

# قائمة بيانات السير الذاتية
resumes = [
    "Name: John Doe, Experience: 5 years, Skills: Python, Machine Learning",
    "Name: Jane Smith, Experience: 3 years, Skills: Data Analysis, SQL",
    "Name: Emily Davis, Experience: 7 years, Skills: Java, Project Management"
]

# استخراج المعلومات باستخدام التعبيرات المنتظمة
for resume in resumes:
    name = re.search(r"Name: ([\w ]+)", resume).group(1)
    experience = re.search(r"Experience: (\d+) years", resume).group(1)
    skills = re.findall(r"Skills: (.+)", resume)[0].split(', ')
    print(f"\nName: {name}\nExperience: {experience} years\nSkills:")
    for skill in skills:
        print(f"- {skill}")
```

**المخرجات:** تحليل السير الذاتية وإظهار كل مهارة في قائمة واضحة، مما يجعل النتائج أسهل قراءة وفهماً.

______________________________________________________________________

### التكاليف

1. كتابة برنامج يعتمد على القواعد:

   1. تطبيق البرنامج على مستندات أخرى (مثل المقالات الإخبارية) لاستخراج معلومات مثل التاريخ، الأشخاص، والأحداث.
   1. مقارنة دقة النهج القائم على القواعد مع نتائج أدوات مثل spaCy وتوثيق الفرق.
