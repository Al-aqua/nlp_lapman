<h1 style="text-align: center;">المحاضرة العملية رقم: 8</h1>

## أولا: مفردات المقرر حسب التوصيف

الشبكات العصبية للمعالجة الطبيعية للغات (Neural Networks for NLP).

______________________________________________________________________

## ثانيا: المحتوى

### العنوان: الشبكات العصبية للمعالجة الطبيعية للغات

______________________________________________________________________

### الموضوع الرئيسي من توصيف النظري:

استكشاف تطبيق الشبكات العصبية العميقة في معالجة اللغة الطبيعية، بما في ذلك الكشف عن الكيانات، تصنيف النصوص، واكتشاف الأنماط باستخدام الشبكات المتقدمة مثل Faster R-CNN.

______________________________________________________________________

### أهداف المعمل

يسعى التطبيق العملي لتحقيق الأهداف التالية:

1. تثبيت وتكوين بيئة عمل تدعم التعلم العميق باستخدام TensorFlow أو PyTorch.
1. تدريب نموذج Faster R-CNN لاكتشاف الكائنات باستخدام مجموعة بيانات مُعلّمة مثل COCO أو PASCAL VOC.

______________________________________________________________________

### الأدوات

- جهاز حاسوب.
- Python.
- TensorFlow أو PyTorch.
- مكتبة OpenCV.
- مجموعة بيانات مُعلّمة (مثل COCO أو PASCAL VOC).

______________________________________________________________________

### المحتوى

- **المسألة/المسائل:**

  1. تثبيت وتكوين بيئة عمل مناسبة للتعلم العميق.
  1. تدريب نموذج Faster R-CNN لاكتشاف الكائنات.

- **التطبيقات:**

#### البرنامج الأول: تثبيت وتكوين بيئة عمل التعلم العميق

الشفرة:

```bash
pip install tensorflow
# أو
pip install torch torchvision
```

```python
# التأكد من تثبيت TensorFlow أو PyTorch
try:
    import tensorflow as tf
    print("TensorFlow is installed. Version:", tf.__version__)
except ImportError:
    print("TensorFlow is not installed.")

try:
    import torch
    print("PyTorch is installed. Version:", torch.__version__)
except ImportError:
    print("PyTorch is not installed.")
```

المخرجات: عرض الإطار الذي تم تثبيته بنجاح والإصدار.

______________________________________________________________________

#### البرنامج الثاني: تدريب نموذج Faster R-CNN لاكتشاف الكائنات

الشفرة:

```bash
pip install torch torchvision opencv-python
```

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2

# تحميل النموذج المدرب مسبقاً
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# قراءة الصورة
image = cv2.imread("sample_image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# تحويل الصورة إلى تنسيق مناسب
image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

# إجراء التنبؤات
with torch.no_grad():
    predictions = model(image_tensor)

# عرض النتائج
for element in predictions[0]['boxes']:
    x1, y1, x2, y2 = element.int()
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

المخرجات: عرض الصورة مع المربعات المحيطة بالكائنات المكتشفة.

______________________________________________________________________

### التكاليف

1. تثبيت وتكوين بيئة العمل:

   - مقارنة الأداء بين TensorFlow و PyTorch.
   - تحليل واجهات برمجة التطبيقات (APIs) لكل إطار عمل.

1. تدريب نموذج Faster R-CNN:

   - استخدام مجموعة بيانات أكبر وأكثر تنوعاً.
   - تحسين الأداء باستخدام GPU أثناء التدريب.
