# 🧬 Nested ComBat Harmonization Tool

This repository provides an implementation of the **Nested ComBat** harmonization method for radiomics data. It is designed to correct variations in extracted radiomic features due to differences in imaging protocols, centers, or acquisition parameters.

The tool performs harmonization at the **feature level**, aligning radiomic features across sites or scanners while preserving biological variability. It includes support for:
- **Standard ComBat**
- **Modified ComBat (M-ComBat)** using a reference batch
- **Nested ComBat** to harmonize multiple batch effects sequentially, automatically determining the optimal order

---

## 📂 Input Files

The tool operates in both **training mode** (to estimate harmonization parameters) and **test mode** (to apply the harmonization model to new data) The tool expects a specific folder structure and input files, which differ slightly depending on the mode:

### Folder Structure
project_directory/
├── input_train/
│ ├── radiomics.csv # Radiomic features per patient
│ └── metadata.csv # Acquisition metadata (e.g., manufacturer, slice thickness, etc.)
├── config/
│ └── config.yaml # Configuration file specifying harmonization settings
├── input_test/ # (Only for test mode)
│ └── radiomics.csv # New radiomic features to harmonize using trained model


### File Descriptions

- `radiomics.csv`:
  - Must include a `PatientID` column and one column per radiomic feature.
- `metadata.csv`:
  - Must include `PatientID`, `Manufacturer`, `Slice Thickness`, `Tube Voltage`, and `Center` columns.
- `config.yaml`:
  - Specifies the harmonization method (`ComBat`, `M-ComBat`, or `Nested ComBat`), batch effects to be corrected, covariates, reference batch (if any), and options to filter out poorly harmonized features.

> ⚠️ **Important Notes:**
> - Column names must match exactly as described.
> - In test mode, new batch categories not present in the training set are not allowed.
> - All test features must have the same names and format as those used during training.


---

## 🧪 Output

After harmonization, the tool generates different outputs depending on the selected mode:

### Training Mode
The following files will be saved in the output directory:
- `nested_combat_best_order.csv` – The optimal order of batch effects used in Nested ComBat
- `features_to_harmo_list.pkl` – List of features that were successfully harmonized
- `estimates.pkl` – Estimated parameters used for feature-wise harmonization

### Test Mode
To run the tool in test mode, the following files **must be available** (produced from the training phase):
- `nested_combat_best_order.csv`
- `features_to_harmo_list.pkl`
- `estimates.pkl`
- A new test `radiomics.csv` in the input test folder

The output will include:
- `harmonized_radiomics_test.csv` – The harmonized radiomic features for the test set

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/nested-combat.git
cd nested-combat
pip install -r requirements.txt

---

## 🐳 Docker
This tool is containerized using Docker. You can build and run the Docker image as follows:

# Build the image
docker build -t nested-combat .

# Run the container
docker run -v /path/to/data:/app/input -v /path/to/output:/app/output nested-combat

---

## 🧪 Example
An example dataset and config file are provided in the examples/ directory. You can test the tool as follows:
python run.py --mode train --input_dir ./examples --output_dir ./output

For test mode (after training):
python main.py --mode test --input_dir ./examples/test --model_path ./output/model.pkl --output_dir ./output/test

---

## ⚠️ Common Errors
- radiomics.csv missing "PatientID" column
- metadata.csv missing one of: "Manufacturer", "Slice Thickness", "Tube Voltage", "Center"
- Trying to harmonize a batch effect that was not used in the training phase
- Adding new batch effect categories in test mode

---

## 📫 Contact
For questions or support, please contact: benito.farina@upm.es

---

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
