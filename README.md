# Federated Learning Simulation with Differential Privacy
This repository implements a simulation framework for federated learning with integrated differential privacy. The simulation trains an Electronic Health Record (EHR) classification model over multiple clients, applying privacy-preserving techniques and visualizing key performance metrics.

## Features
- Simulates distributed training over multiple clients.

- Applies differential privacy through a custom DPAdam optimizer and per-example gradient clipping.

- Uses gradient clustering techniques to aggregate client updates robustly.

## Visualization

Generates plots for:

1. Accuracy vs. Privacy (ε)

2. ROC Curve

3. Resource Consumption

4. Comparative Performance Table

## Interactive UI

A Streamlit web application provides an interactive interface for setting parameters and running simulations.

## Project Structure
- **streamlit_app.py:**
The Streamlit UI that allows you to configure simulation parameters and trigger federated training.

- **main.py:**
Contains the core simulation code including data loading, federated training, gradient aggregation, and plotting.

- **data_loader.py:**
Loads and preprocesses the EHR data from a CSV file (expects data/patient_treatment.csv).

- **model.py:**
Defines the deep neural network model used for EHR classification.

- **dp_optimizer.py:**
Implements the DPAdam optimizer and functions for differentially private gradient computation.

- **gradient_clustering.py:**
Provides functions for clustering and aggregating gradients from clients.

- **plots/:**
Directory where simulation output images (e.g., roc_curve.png, accuracy_privacy.png, comparative_table.png, resource_consumption.png) are saved.

## Requirements
- Python 3.7 or higher

- TensorFlow

- Streamlit

- NumPy

- Pandas

- scikit-learn

- psutil

- Optional: TensorFlow Privacy (for computing DP epsilon)

## Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Setup Instructions
1. Clone the Repository:

```bash
git clone [<repository-url>](https://github.com/iamrishabruh/difflearn/)
cd your/path/
```

2. Prepare the Data:

```bash
1. Place your EHR data file (e.g., patient_treatment.csv) into a folder named data in the root directory.
2. Ensure the CSV file contains the necessary columns (e.g., SEX, HAEMATOCRIT, HAEMOGLOBINS, AGE, etc.). See data_loader.py for details on preprocessing.
```

3. Run the Simulation:

  - **Using the Command Line:**

```bash
python main.py
This will execute federated training and generate plots saved in the plots folder.
```

  - **Using the Streamlit UI:**

```bash
streamlit run streamlit_app.py
Use the sidebar controls to adjust simulation parameters, then click Run Federated Training. The app displays the simulation progress, evaluation metrics, and generated plots.
```

## Output
After running the simulation, you will see:

**Global Model Evaluation:**
The summary of the updated global model along with evaluation metrics (loss and accuracy) on a validation set.

**Generated Plots:**
The following plots will be saved in the plots/ directory:

- **accuracy_privacy.png:** Accuracy vs. Privacy (ε) trade-off.

- **roc_curve.png:** Receiver Operating Characteristic curve.

- **comparative_table.png:** Comparative performance table.

- **resource_consumption.png:** Resource consumption during training.

## Customization

- Modify default parameters in streamlit_app.py or main.py to adjust the number of clients, local epochs, learning rates, and other hyperparameters.

- Update model.py if you wish to change the neural network architecture.

- Modify data_loader.py to suit the format and requirements of your dataset.

## License
This project is licensed under the MIT License.

## Acknowledgments
This project leverages TensorFlow, Streamlit, scikit-learn, and other open-source libraries to simulate federated learning with differential privacy. Special thanks to the contributors and open-source community for their invaluable tools and resources.

