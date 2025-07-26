# Bone Fracture Classification using YOLOv8

A deep learning project for classifying different types of bone fractures using YOLOv8 computer vision model. This system can automatically identify and classify 10 different types of bone fractures from X-ray images.

## 🦴 Fracture Types Detected

The model can classify the following bone fracture types:

- **Avulsion fracture**
- **Comminuted fracture**
- **Compression-Crush fracture**
- **Dislocation Fracture**
- **Greenstick fracture**
- **Hairline Fracture**
- **Impacted fracture**
- **Intra-articular fracture**
- **Longitudinal fracture**
- **Oblique fracture**
- **Pathological fracture**
- **Spiral Fracture**

## 🚀 Features

- **High Accuracy Classification**: Uses YOLOv8 for state-of-the-art bone fracture detection
- **Multi-class Support**: Identifies 10+ different fracture types
- **GPU Acceleration**: CUDA support for faster inference
- **Batch Prediction**: Process multiple images at once
- **Confidence Scoring**: Get prediction confidence for each classification
- **Easy to Use**: Simple scripts for training and prediction

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/parthib22/YOLO-Bone-Injury-Classification.git
cd YOLO-Bone-Injury-Classification
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download the dataset (optional, for training):

```bash
python dataset.py
```

## 📊 Dataset

This project uses the Bone Break Classification dataset from Roboflow:

- **Training images**: Various bone fracture X-rays
- **Validation images**: Separate validation set
- **Test images**: Test set for evaluation
- **License**: CC BY 4.0

The dataset contains X-ray images of different bone fractures, organized by fracture type for supervised learning.

## 🎯 Usage

### 📷 Making Predictions

1. **Add your X-ray images** to the `content/` folder
2. **Run prediction**:
   ```bash
   python predict.py
   # or
   python predict_arrg.py
   # if X-rays are of the same bone
   ```

### 🔄 Optional: Training Your Own Model

⚠️ **Training Requirements:**

- **Minimum**: 8GB GPU VRAM
- **Recommended**: 12GB+ GPU VRAM
- **RAM**: 12GB+ system memory
- **Storage**: enough free space for datasets and training outputs

1. **Download dataset from [Roboflow](https://universe.roboflow.com/search?q=bone+model%3Ayolov8)** (optional):

   ```bash
   python dataset.py
   ```

2. **Train the model**:
   ```bash
   python train.py
   ```
   ⚠️ **Warning**: Training can take 2-6 hours depending on your GPU

### 🛠️ Utility Scripts

- **Check GPU compatibility**:

  ```bash
  python gpu.py
  ```

## 📁 Project Structure

```
yolo_bone_injury_classification/
├── main.py              # Main prediction script
├── train.py             # Model training script
├── predict.py           # Interactive prediction script
├── predict_aggr.py      # Aggregate prediction analysis
├── check_preds.py       # Prediction validation
├── dataset.py           # Dataset download script
├── gpu.py               # GPU check utility
├── best.pt              # Trained model weights
├── yolov8n-cls.pt       # Base YOLOv8 model
├── content/             # Sample images for testing
├── Bone-Break-Classification-2/  # Dataset folder
└── Classification-2/    # Additional dataset
```

## 🏥 Medical Disclaimer

⚠️ **Important**: This tool is for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The dataset used is licensed under CC BY 4.0.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the excellent computer vision framework
- [Roboflow](https://roboflow.com/) for providing the bone fracture dataset
- Medical imaging community for advancing AI in healthcare

## 📈 Model Performance

The trained model achieves high accuracy on the validation set. For detailed performance metrics, check the training logs generated during model training.

## 🔧 Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model not found**: Ensure `best.pt` is in the project directory
3. **Dataset path errors**: Check dataset folder structure

### GPU Setup:

- Install CUDA toolkit compatible with PyTorch
- Verify GPU availability with `python gpu.py`

## 📞 Contact

If you have any questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for advancing medical AI**
