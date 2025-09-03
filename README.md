# 19F NMR Spectrum Predictor

[![Live Demo](https://img.shields.io/badge/Live%20Demo-AWS%20Lambda-blue)](https://wqdx9jslij.execute-api.us-east-2.amazonaws.com/prod/)

An AI-powered machine learning system for predicting 19F NMR spectra of fluorinated compounds, with a focus on PFAS (Per- and Polyfluoroalkyl Substances). This project combines multiple machine learning approaches including Ridge Regression, Feed-Forward Neural Networks, XGBoost, and HOSE-based nearest neighbors to provide accurate NMR predictions.

## 🌐 Live Web Application

**Try the live application:** [19F NMR Spectrum Predictor](https://wqdx9jslij.execute-api.us-east-2.amazonaws.com/prod/)

The web application allows you to:
- Enter SMILES notation for fluorinated compounds
- Get AI-powered 19F NMR spectrum predictions
- View confidence levels and analysis results
- Process compounds in 50ms to 1 second

### 🔗 Related Applications

**FFNN Web Application**: For a dedicated Feed-Forward Neural Network implementation with a modern web interface, see the separate repository: [Application_19F_NMR_Spectra_predictor_FFNN](https://github.com/Dandan-Rao/Application_19F_NMR_Spectra_predictor_FFNN)

This FFNN application features:
- Modern Bootstrap 5 UI with real-time validation
- Molecular structure visualization with fluorine highlighting
- Confidence level indicators (L1-L6) for each predicted peak
- Docker deployment support
- Flask-based web framework

## 🚀 Features

- **Multi-Model Ensemble**: Combines Ridge Regression, FFNN, XGBoost, and HOSE-based models
- **2D & 3D Descriptors**: Utilizes both 2D molecular descriptors and 3D structural features
- **HOSE Code Integration**: Implements Hierarchical Organization of Spherical Environments for chemical similarity
- **Real-time Predictions**: Fast processing with AWS Lambda deployment
- **Comprehensive Analysis**: Provides confidence levels and uncertainty quantification
- **PFAS Focus**: Specialized for Per- and Polyfluoroalkyl Substances analysis

## 📊 Model Performance

The system uses an ensemble approach combining:
- **Ridge Regression** with 2D/3D feature sets
- **Feed-Forward Neural Networks** for non-linear relationships
- **XGBoost** for gradient boosting
- **HOSE-based Nearest Neighbors** for chemical similarity

## 🛠️ Installation

### Prerequisites

- **Python 3.6+**
- **Java** (openjdk version "23.0.2" or compatible)
- **MATLAB R2021b** (for certain molecular descriptor calculations)

### Quick Start with GitHub Codespaces

1. **Fork the repository**

2. **Open GitHub Codespaces**:
   - Click 'Code' → 'Codespaces' → 'Create Codespace on main'

3. **Set up the environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate
   
   # Install dependencies
   make install
   ```

4. **Run Jupyter notebooks**:
   - Open any notebook (e.g., `1_Ridge_and_FFNN_models_use_2D_and_3D_feature_sets.ipynb`)
   - Select kernel: `.venv (Python 3.12.1) .venv/bin/python`

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd 19F_NMR_Spectrum_Predictor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
19F_NMR_Spectrum_Predictor/
├── notebooks/                    # Jupyter notebooks for model development
│   ├── 1_Ridge_and_FFNN_models_use_2D_and_3D_feature_sets.ipynb
│   ├── 2_XGBoost_model_2D_and_3D_descriptors.ipynb
│   ├── 3_HOSE_based_nearest_neighbors_model.ipynb
│   ├── 4_compare_the_model_performance_and_create_an_ensembled_model.ipynb
│   ├── 5_application.ipynb
│   └── 6_rule_verification_and_new_rules.ipynb
├── src/                         # Source code
│   ├── atomic_features_2D.py    # 2D molecular descriptors
│   ├── atomic_features_3D.py    # 3D structural features
│   ├── hose_code.py            # HOSE code generation
│   └── common.py               # Common utilities
├── dataset/                     # Training and test datasets
│   ├── descriptors/            # Molecular descriptors
│   ├── neighbors/              # 3D neighbor information
│   └── sdf/                    # 3D molecular structures
├── artifacts/                   # Model artifacts and results
│   ├── models/                 # Trained models (.pkl, .h5, .json)
│   └── results/                # Performance results
├── external/                    # External tools and dependencies
└── web_page.png                # Web application screenshot
```

## 🔬 Usage

### Web Application

1. Visit the [live application](https://wqdx9jslij.execute-api.us-east-2.amazonaws.com/prod/)
2. Enter a SMILES string for your fluorinated compound
3. Click "Predict 19F NMR Spectrum"
4. View the predicted spectrum and confidence levels

### Example SMILES

- **PFBA**: `C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O`
- **PFBS**: `C(C(C(F)(F)S(=O)(=O)O)(F)F)(C(F)(F)F)(F)F`

### Programmatic Usage

```python
from src.atomic_features_2D import get_2D_descriptors
from src.atomic_features_3D import get_3D_descriptors
from src.hose_code import generate_hose_codes

# Example: Get descriptors for a SMILES string
smiles = "C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O"
descriptors_2D = get_2D_descriptors(smiles)
descriptors_3D = get_3D_descriptors(smiles)
hose_codes = generate_hose_codes(smiles)
```

## 🧪 Requirements

### SMILES Input Requirements

- Must contain at least one fluorine atom (F)
- Use valid SMILES syntax
- Compound should be fluorinated
- Processing time: 50ms to 1 second depending on complexity

### Java Compilation (if modifying Java code)

```bash
# Compile Java program
javac -cp .:cdk-1.4.13.jar:cdk.jar:jmathio.jar:jmathplot.jar GetCDKDescriptors.java

# Run Java program
java -cp .:cdk-1.4.13.jar:cdk.jar:jmathio.jar:jmathplot.jar GetCDKDescriptors temp
```

## 📈 Model Development

The project includes comprehensive notebooks for:

1. **Model Training**: Ridge and FFNN models with 2D/3D features
2. **XGBoost Implementation**: Gradient boosting for enhanced performance
3. **HOSE Integration**: Chemical similarity-based predictions
4. **Ensemble Methods**: Combining multiple models for robust predictions
5. **Application Examples**: Real-world usage scenarios
6. **Rule Verification**: Validation and new rule discovery

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Dandan Rao** - *Initial work* - [kiluarao@gmail.com](mailto:kiluarao@gmail.com)

## 🙏 Acknowledgments

- CDK (Chemistry Development Kit) for molecular descriptor calculations
- RDKit for cheminformatics functionality
- TensorFlow/Keras for neural network implementations
- XGBoost for gradient boosting
- AWS Lambda for web application deployment

## 📞 Support

For questions or support, please contact [kiluarao@gmail.com](mailto:kiluarao@gmail.com) or open an issue on GitHub.

---

**© 2024 19F NMR Spectrum Predictor. Powered by AI and Machine Learning.**