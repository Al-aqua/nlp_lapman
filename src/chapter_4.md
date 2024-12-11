<h1 style="text-align: center;">المحاضرة العملية رقم: 4</h1>

## أولا: مفردات المقرر حسب التوصيف

النحو ومعالجة الجمل باستخدام تقنيات التحليل النحوي.

---

## ثانياً: المحتوى

### العنوان: النحو والتحليل النحوي باستخدام المكتبات والأدوات البرمجية

---

### الموضوع الرئيسي من توصيف النظري:

التعرف على كيفية تحليل الجمل لغوياً باستخدام تقنيات التحليل النحوي لمعرفة العلاقات النحوية والبنائية بين الكلمات في الجمل.

---

### أهداف المعمل

يسعى التطبيق العملي لتحقيق الأهداف التالية:

1. استخدام مكتبة لتحليل الاعتمادية (Dependency Parsing) لاستخراج العلاقات بين الكلمات في جملة معينة.
1. استخدام أداة تحليل التركيب (Constituency Parsing) لفهم البنية الهرمية للجمل.
1. مقارنة النتائج المستخرجة من التحليل الاعتمادي والتحليل التركيبي.

---

### الأدوات

- جهاز حاسوب.
- Python.
- مكتبة spaCy.
- مكتبة NLTK.
- Stanford Parser (اختياري).

---

### المحتوى

- **المسألة/المسائل:**

  1. استخدام مكتبة تحليل الاعتمادية، مثل spaCy أو NLTK، لتحليل جملة واستخراج العلاقات بين الكلمات.
  1. استخدام أداة تحليل التركيب، مثل Stanford Parser أو NLTK، لإجراء تحليل تركيبي على جملة معينة.

- **التطبيقات**

1. البرنامج الأول: تحليل الاعتمادية باستخدام مكتبة spaCy

الشفرة:

```
pip install spacy
python -m spacy download en_core_web_sm
```

```python
import spacy

# تحميل نموذج اللغة الإنجليزية في spaCy
nlp = spacy.load("en_core_web_sm")

# الجملة المراد تحليلها
sentence = "The quick brown fox jumps over the lazy dog."

# تحليل الجملة
doc = nlp(sentence)

# استخراج العلاقات بين الكلمات
for token in doc:
    print(f"Word: {token.text}, Dependency: {token.dep_}, Head: {token.head.text}")
```

| Dependency (Abbreviation) | Dependency (Full Description)    |
| ------------------------- | -------------------------------- |
| `det`                     | Determiner                       |
| `amod`                    | Adjective Modifier               |
| `nsubj`                   | Nominal Subject                  |
| `root`                    | Root (Main Verb)                 |
| `case`                    | Case (Prepositional Case Marker) |
| `obl`                     | Oblique Argument                 |

المخرجات: عرض العلاقات بين الكلمات (مثل الكلمة الرئيسية والكلمات التابعة).

---

2. البرنامج الثاني: التحليل التركيبي باستخدام مكتبة NLTK

الشفرة:

```python
import nltk
from nltk import CFG


def print_tree_structure(tree, level=0):
    """Print a tree structure with indentation"""
    if isinstance(tree, nltk.Tree):
        print("  " * level + tree.label())
        for child in tree:
            print_tree_structure(child, level + 1)
    else:
        print("  " * level + str(tree))


# Define the grammar
grammar = CFG.fromstring(
    """
    S -> NP VP
    NP -> DT NN | DT JJ NN
    VP -> VBZ PP
    PP -> IN NP
    DT -> 'the' | 'The'
    JJ -> 'quick' | 'lazy'
    NN -> 'fox' | 'dog'
    VBZ -> 'jumps'
    IN -> 'over'
"""
)

# Create the parser
parser = nltk.ChartParser(grammar)

# The sentence to be parsed
sentence = ["The", "quick", "fox", "jumps", "over", "the", "lazy", "dog"]

# Parse the sentence and display the results
print("Bracket notation:")
for tree in parser.parse(sentence):
    print(tree)
    print("\nDetailed tree structure:")
    print_tree_structure(tree)
```

المخرجات: عرض الشجرة النحوية التي تمثل البنية الهرمية للجملة.

---

### التكاليف

1. تمرين 1: تحليل جمل باستخدام تحليل الاعتمادية:

   1. اختر ثلاث جمل مختلفة من اختيارك.
   1. استخدم مكتبة spaCy لتحليل العلاقات بين الكلمات.
   1. وثق النتائج في تقرير.

1. تمرين 2: تحليل جمل باستخدام التحليل التركيبي:

   1. اختر ثلاث جمل مختلفة من اختيارك.
   1. استخدم مكتبة NLTK أو Stanford Parser لإجراء تحليل تركيبي.
   1. قارن النتائج المستخرجة مع التحليل الاعتمادي.
