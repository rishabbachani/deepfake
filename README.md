# Deepfake Detection with MTCNN and EfficientNet-B0

An end-to-end system that decides whether a face image or video is **real** or **AI-generated / manipulated**
and returns a confidence score through a web interface.

**Team:** Rishab Bachani · Anish Verma Bhupathiraju · Chetandeep Singh Phull · Ashar Khan · Vedha Thota
**Mentor:** Dr. B. Sanjai Prasad

---

## How it works

```
 image / video ──► OpenCV frame sampling ──► MTCNN face detection ──► square crop + 30 % margin
                                                                              │
                                                                       resize 224×224×3
                                                                              │
  Flask API ◄── Platt-calibrated P(fake) ◄── sigmoid head ◄── EfficientNet-B0 (ImageNet, fine-tuned)
      │
      ▼
  Web UI: verdict, confidence, the face crops that were scored, per-frame scores for videos
```

* **Face isolation (MTCNN).** Most of a frame is background. Cropping the largest detected face (with a
  30 % margin so the blending boundary is kept) makes the classifier look only at the region that
  manipulation methods change. The *same* crop function is used for training data and for uploads.
* **Classifier (EfficientNet-B0).** ImageNet-pretrained backbone + custom head
  (GlobalAveragePooling → Dropout → Dense 256 ReLU → Dropout → Dense 1 sigmoid). Output = P(fake).
* **Two-stage transfer learning.** Stage 1 trains only the head on frozen features (LR 1e-3).
  Stage 2 unfreezes the top 60 backbone layers (BatchNorm kept frozen) and fine-tunes with LR 1e-5.
  The checkpoint with the best validation ROC-AUC is kept.
* **Videos.** 16 evenly spaced frames are sampled, faces are scored independently and averaged.
* **Calibration.** Platt scaling is fitted on the validation split so the confidence shown in the app is a
  calibrated probability, not a raw sigmoid value.

## Dataset

[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (Kaggle):
70k real faces (Flickr-Faces-HQ) and 70k StyleGAN-generated faces, already split into train / valid / test.
By default we sample **7,000 / 1,500 / 1,500 images per class** (20,000 total) from the official splits, so
no test image is ever seen in training.

`prepare_dataset.py` also accepts any `{real,fake}/` image folder, or a DFDC-style folder of videos with
`metadata.json` (frames are sampled per video and split **per video** to avoid leakage).

## Project structure

```
deepfake_detector/        reusable package
    config.py             image size, class order, paths, thresholds
    faces.py              MTCNN wrapper, square face crop, video frame sampling
    model.py              EfficientNet-B0 model, unfreezing, robust loading
    data.py               tf.data loaders for the prepared dataset
    inference.py          DeepfakeDetector: image / video -> verdict
prepare_dataset.py        raw dataset -> data/processed/{train,val,test}/{real,fake}
train.py                  two-stage training -> models/best_model.keras
evaluate.py               test metrics, plots, calibration -> reports/
predict.py                command-line prediction
app.py                    Flask web app
templates/, static/       web UI (HTML / CSS / JS)
notebooks/train_on_colab.ipynb   one-click training on a free Colab GPU
```

## Running it

### 1. Train (Google Colab, recommended)

Open `notebooks/train_on_colab.ipynb` in Colab, set **Runtime → T4 GPU**, and *Run all*.
It downloads the dataset, prepares it, trains, evaluates and saves `models/` + `reports/` to Google Drive
and as a zip download.

### 1b. Train locally (needs a GPU to be practical)

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (source venv/bin/activate on macOS / Linux)
pip install -r requirements.txt

python prepare_dataset.py --source path/to/real_vs_fake
python train.py
python evaluate.py
```

### 2. Run the web app

Put the trained `models/` folder (from Colab) in the project root, then:

```bash
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>, drop an image or a video, press **Analyse**.

Command line alternative: `python predict.py some_face.jpg some_video.mp4`

## Outputs of `evaluate.py`

| File | Content |
|---|---|
| `reports/metrics.json` | accuracy, ROC-AUC, macro and per-class precision / recall / F1, confusion matrix, calibration error |
| `reports/classification_report.txt` | scikit-learn classification report |
| `reports/confusion_matrix.png` | confusion matrix on the test set |
| `reports/roc_curve.png` | ROC curve with AUC |
| `reports/training_curves.png` | accuracy / AUC / loss per epoch, with the fine-tuning boundary marked |
| `reports/calibration.png` | reliability diagram before and after Platt scaling |

## Limitations

* Trained on StyleGAN face images; face-swap videos (DFDC, FaceForensics++) and newer diffusion
  generators are a different distribution, so accuracy on them is expected to be lower.
* Video frames are scored independently; there is no temporal model (e.g. LSTM / 3D-CNN).
* Cross-dataset evaluation (Celeb-DF, FaceForensics++) is future work.
* Heavy compression, very small faces or extreme poses reduce reliability. When MTCNN finds no face,
  the app scores the whole frame and shows a warning.

## References

1. Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019.
2. Nirkin et al., *DeepFake Detection Based on the Discrepancy Between the Face and its Context*, 2020.
3. Khan & Dang-Nguyen, *Deepfake Detection: A Comparative Analysis*, 2023.
4. *Comprehensive Evaluation of Deepfake Detection Models*, Applied Sciences, 2025.
5. Kumar et al., *A Hybrid Spatial-Frequency Attention-Based Algorithm Using EfficientNet*, 2026.
6. Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
7. Zhang et al., *Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks*, 2016.
