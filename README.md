# 🪖 Helmet 🪖 Detection using YOLOv8 💡

### Introduction 🌟

Welcome to the fascinating world of helmet detection using the powerful YOLOv8 (You Only Look Once) object detection algorithm! 🚀 In this comprehensive guide, we will walk you through the process of implementing helmet detection with utmost professionalism. YOLOv8, known for its real-time object detection capabilities, will enable you to swiftly identify helmets in images or video streams. 🎯

### Prerequisites 📚

Before embarking on this exciting journey, make sure you meet the following prerequisites:
* A system with Python 3.x installed. 🖥️
* Basic knowledge of Python programming and deep learning concepts. 🐍
* Familiarity with the YOLO (You Only Look Once) object detection algorithm. 👀
* Installation of required packages, such as OpenCV, PyTorch, and others. 📦

```bash
pip install ultralytics
```

Pip install the ultralytics package including
all [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a
[**3.10>=Python>=3.7**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).



### Steps 📝

#### 1. Dataset Collection 📂

To train a stellar YOLOv8 model for helmet detection, you must begin by assembling a captivating dataset. Collect images or videos showcasing individuals wearing helmets, and diligently annotate the bounding boxes around these protective headgears. You can either manually annotate the data or source existing annotated datasets available [Roboflow](https://universe.roboflow.com/bike-helmets/bike-helmet-detection-2vdjo) . 📸

#### 2. Data Preparation 🛠️

Prepare your dataset meticulously by following these steps:
    Delicately divide the dataset into training, Testing and validation sets. Arrange the data in the YOLO format, ✂️ If you have downloaded dataset from Roboflow it's already divided into yolo     format.which traditionally consists of an image file paired with a corresponding text file containing annotated bounding boxes. 📦

#### 3. Model Training 🚀

To embark on the training journey of your YOLOv8 model, navigate through these essential steps:

* Acquire the YOLOv8 architecture and pre-trained weights from the official repository or a trustworthy source. 🏛️
* Configure the network architecture and hyperparameters according to your specific requirements. 🧰
* Initialize your YOLOv8 model with the pre-trained weights, laying the foundation for exceptional performance. 🌟
* Train the model using your meticulously prepared dataset, optimizing the loss function to hone its detection prowess. ⚙️
* Stay vigilant during the training process, constantly monitoring progress and adjusting hyperparameters if necessary. 👁️
* Preserve the trained weights for future inference tasks, ensuring your model's brilliance persists. 💾

#### CLI

YOLOv8 may be used directly in the Command Line Interface (CLI) with a `yolo` command:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 
```


#### Python

YOLOv8 may also be used directly in a Python environment, and accepts the
same [arguments](https://docs.ultralytics.com/cfg/) as in the CLI example above:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data="data.yaml", epochs=100)  # train the model
```

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models) download automatically from the latest
Ultralytics [release](https://github.com/ultralytics/assets/releases). See
YOLOv8 [Python Docs](https://docs.ultralytics.com/python) for more examples.


#### 4. Inference 👁️‍🗨️

Once your model is prepared to showcase its talents, follow these enchanting steps for inference:

* Summon the trained YOLOv8 weights, enabling your model to shine. ✨
* Prepare the input images or video frames with utmost care, setting the stage for a captivating performance. 🖼️
* Allow the preprocessed data to gracefully pass through the YOLOv8 model, unraveling the mystery of object detection. 🌌
* Enchant your audience by applying post-processing techniques to filter and refine the detected bounding boxes, unveiling the true beauty of helmet detection. ✨🎩
* With an artistic touch, visualize the detected helmets by adorning the input images or video frames with elegant bounding boxes. 🎨🖼️
* Optionally, capture the essence of the moment by preserving the output images or videos, immortalizing the marvels of detected helmets. 📸📹
  
<font style="color:green">You can also skip all these steps and use prediction on the weight trained on Roboflow dataset used in this repositoray.</font>

#### CLI

YOLOv8 may be used directly in the Command Line Interface (CLI) with a `yolo` command:

```bash
yolo predict model=best.pt source="helmet.mp4"
```

#### Python

```python
from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model("helmet.mp4")  # predict on an image
```

#### 5. Evaluation 📊

To appraise the sheer brilliance of your trained model, embark on this captivating evaluation journey:

* Curate a separate test dataset adorned with annotated ground truth bounding boxes, enabling the stage to be set for a grand performance. 🎭
* Evoke the trained model's grace by performing inference on the test dataset, unveiling its true potential. 🌟
* Measure the model's performance through evaluation metrics such as precision, recall, and mean Average Precision (mAP), allowing its magnificence to be quantified. 📈
* Dive deep into the results, extracting insights, and identifying areas for improvement, paving the way for even more captivating performances. 🔍📈

### Conclusion 🎉

Congratulations on mastering the art of helmet detection using the magical YOLOv8 algorithm! By following these meticulously crafted steps, you have unlocked the ability to train a YOLOv8 model that gracefully identifies helmets in images or video streams. Brace yourself for remarkable applications, such as safety monitoring in construction sites, sports, or industrial environments. Remember, like an artist refining their masterpiece, fine-tune the model and experiment with different hyperparameters to achieve breathtaking performance tailored to your specific use case. Embrace the beauty of helmet detection and continue pushing the boundaries of what is possible! 🌟✨


## You can also skip all these steps and use prediction on the weight trained on Roboflow dataset used in this repositoray.</font> 🪖🪖🪖🪖🪖🪖

#### CLI

YOLOv8 may be used directly in the Command Line Interface (CLI) with a `yolo` command:

clone this repository in personal machine 
```bash
git clone https://github.com/prince0310/Helmet-Detection-using-YOLOv8.git
```
Move into the folder Helmet-Detection-using-YOLOv8
```bash
cd Helmet-Detection-using-YOLOv8
```
Start prediction on video 
```bash
yolo predict model=best.pt source="helmet.mp4"
```

#### Python

```python
from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model("helmet.mp4")  # predict on an image
```
Result on predition can be shown 



https://github.com/prince0310/Helmet-Detection-using-YOLOv8/assets/85225054/848a8219-b3c0-4293-823c-13046415fb79



