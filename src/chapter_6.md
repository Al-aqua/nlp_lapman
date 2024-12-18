<h1 style="text-align: center;">المحاضرة العملية رقم: 6</h1>

## أولا: مفردات المقرر حسب التوصيف

الترجمة الآلية وتوليد النصوص.

______________________________________________________________________

## ثانياً: المحتوى

### العنوان: الترجمة الآلية وتوليد النصوص باستخدام بايثون

______________________________________________________________________

### الموضوع الرئيسي من توصيف النظري:

فهم كيفية استخدام الترجمة الآلية لتوليد نصوص وتحليلها باستخدام نماذج تعلم الآلة.

______________________________________________________________________

### أهداف المعمل

يسعى التطبيق العملي لتحقيق الأهداف التالية:

1. استخدام مجموعة بيانات موازية مترجمة لتطبيق الترجمة الآلية.
1. كتابة برنامج لتوليد نصوص باستخدام النماذج اللغوية.
1. مقارنة أداء النماذج المختلفة في الترجمة وتوليد النصوص.

______________________________________________________________________

### الأدوات

- جهاز حاسوب.
- Python.
- مكتبة transformers (Hugging Face).
- مكتبة pandas.
- مجموعة بيانات موازية للترجمة.

______________________________________________________________________

### المحتوى

- **المسألة/المسائل:**

  1. استخدام مجموعة بيانات موازية تحتوي على جمل في لغة المصدر وترجمتها في لغة الهدف لتطبيق الترجمة الآلية.
  1. كتابة برنامج يقوم بتوليد نصوص باستخدام نموذج لغة.

- **التطبيقات**

1. البرنامج الأول: الترجمة الآلية باستخدام مجموعة بيانات موازية

```bash
pip install transformers
pip install sentencepiece
pip install torch
```

```python
from transformers import MarianMTModel, MarianTokenizer

# تحميل النموذج والمحول الخاص بالترجمة
model_name = 'Helsinki-NLP/opus-mt-en-ar'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# النصوص المراد ترجمتها
texts = [
    "The weather is nice today.",
    "I am learning Python programming.",
    "Machine translation is a fascinating field."
]

# ترجمة النصوص
for text in texts:
    encoded_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
    translated = model.generate(**encoded_text)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(f"Original: {text}\nTranslated: {translated_text}\n")
```

**المخرجات:** ترجمة النصوص من الإنجليزية إلى العربية باستخدام نموذج MarianMT.

______________________________________________________________________

2. البرنامج الثاني: توليد النصوص باستخدام نموذج لغوي

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# تحميل النموذج والمحول الخاص بـ GPT-2
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# النصوص التمهيدية لتوليد النصوص
prompts = [
    "Once upon a time in a small village,",
    "Artificial intelligence has changed the way we",
    "In the future, humans will"
]

# توليد النصوص
for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}\nGenerated: {generated_text}\n")
```

**المخرجات:** إنتاج نصوص مكملة للجمل التمهيدية باستخدام نموذج GPT-2.

______________________________________________________________________

### التكاليف

1. توليد النصوص باستخدام النماذج اللغوية:

   1. تجربة نماذج مختلفة (مثل GPT-2 وT5) لتوليد النصوص.
   1. مقارنة النصوص المولدة من حيث الجودة والطول والارتباط بالسياق.
