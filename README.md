<div align="center">

# 🎓 MHT-CET College Predictor & Admission Analytics System

### Machine Learning–Powered College Discovery | Historical Cutoff Analysis | Admission Decision Support

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Application-000000?logo=flask)
![CatBoost](https://img.shields.io/badge/ML-CatBoost-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Render](https://img.shields.io/badge/Deployment-Render-46E3B7)

**Explore engineering college options using MHT-CET percentile, admission category, branch preference, and historical cutoff trends.**

🌐 **[Live Application](https://mht-cet-college-predictor-admission.onrender.com/)** · 💻 **[GitHub Repository](https://github.com/yashchaudhari372-lab/MHT-CET-College-Predictor-Admission-Analytics-System)**

</div>

---

## 📌 Project Overview

The **MHT-CET College Predictor & Admission Analytics System** is a data-driven admission decision-support application built with **Python, Flask, Pandas, and CatBoost**. It helps students explore potential engineering college and branch options by comparing their MHT-CET percentile with historical admission cutoff records.

The application combines historical cutoff filtering, a trained **CatBoost regression model**, and an interactive interface to display ranked college options and admission-related analytics.

> **Disclaimer:** This is an independent educational project, not an official Maharashtra CET Cell admission portal. Historical cutoffs and model estimates do **not** guarantee admission. Always verify the latest official CAP-round rules, seat matrix, category eligibility, and cutoffs before making admission decisions.

## 🎯 Problem Statement

Students often struggle to interpret large cutoff tables and compare engineering college options across branches, reservation categories, genders, and seat types. This project organizes historical admission information into a searchable, percentile-based recommendation experience.

## ✨ Key Features

- **Percentile-based search:** Enter an MHT-CET percentile between 0 and 100.
- **Advanced filters:** Narrow results by college, engineering branch, category, gender, and seat type.
- **Historical cutoff comparisons:** Review minimum, 10th-percentile, median, and maximum cutoff statistics where available.
- **Machine learning estimates:** Uses a saved CatBoost regressor to estimate a cutoff-related target for matching college/branch combinations.
- **Recommendation grouping:** Displays *Safe Zone*, *Target / Reach*, and *Ambitious* categories based on comparisons with historical cutoff thresholds.
- **Admission analytics:** Summarizes matching records and cutoff distributions.
- **CSV export:** Download selected recommendation records for further comparison.
- **Responsive web interface:** Access the application through a browser.

**Important:** The recommendation groups are heuristic comparisons with historical data, not calibrated probabilities of admission.

## 🛠️ Technology Stack

| Area | Technology |
|---|---|
| Programming | Python |
| Web framework | Flask |
| Machine learning | CatBoost Regressor |
| Data processing | Pandas, NumPy |
| Model metadata | Joblib / Pickle |
| Web server | Gunicorn |
| Deployment | Render |
| Data source | Historical admission cutoff CSV |

## 📊 Dataset & Input Features

The application loads `historical_cutoffs.csv` to filter and compare historical college admission records.

| Field | Purpose |
|---|---|
| `college` | Engineering college name |
| `branch` | Engineering specialization |
| `category` | Admission / reservation category |
| `gender` | Gender-related admission classification |
| `seat_type` | Seat or quota classification |
| `allotment_count` | Historical allotment count |
| `min_percentile` | Minimum recorded percentile |
| `cutoff_p10` | Historical 10th-percentile cutoff statistic |
| `median_percentile` | Median recorded percentile |
| `max_percentile` | Maximum recorded percentile |

The saved model metadata identifies **`cutoff_p10`** as the prediction target, with `college`, `branch`, `category`, `gender`, `seat_type`, and `allotment_count` as features.

## 🧠 Machine Learning Approach

The system loads the trained model from `mhtcet_catboost.cbm` using `CatBoostRegressor`. CatBoost is a gradient-boosting algorithm suited to tabular datasets containing categorical features.

### Prediction workflow

```text
Student enters percentile and admission preferences
                       |
                       v
             Validate input values
                       |
                       v
         Filter historical cutoff data
                       |
                       v
        Rank relevant college/branch rows
                       |
                       v
        Run CatBoost cutoff estimation
                       |
                       v
     Compare percentile with historical P10
                       |
                       v
    Label Safe / Target-Reach / Ambitious
                       |
                       v
     Display analytics and exportable results
```

If the model cannot be loaded, the application can still provide historical-cutoff-based comparisons; model predictions may be unavailable.

## 📈 Model Performance

The application source includes the following **reported** regression metrics:

| Metric | Value | Interpretation |
|---|---:|---|
| Training RMSE | **12.70** | Root mean squared prediction error on training data |
| Test RMSE | **12.16** | Root mean squared prediction error on test data |

**Note:** These values are recorded in `app.py` as `VERIFIED_METRICS`; the repository does not include the full training/evaluation notebook needed to independently reproduce them. RMSE is expressed in the target's percentile-point units. Do not interpret these results as classification accuracy or an admission success rate.

## 📁 Repository Structure

```text
MHT-CET-College-Predictor-Admission-Analytics-System/
├── app.py                  # Flask application, UI and API routes
├── historical_cutoffs.csv  # Historical college cutoff records
├── mhtcet_catboost.cbm     # Trained CatBoost regression model
├── model_metadata.pkl      # Saved feature/target metadata
├── requirements.txt        # Python dependencies
├── procfile.txt            # Deployment-related process file
└── README.md               # Project documentation
```

## ⚙️ Installation & Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/yashchaudhari372-lab/MHT-CET-College-Predictor-Admission-Analytics-System.git
cd MHT-CET-College-Predictor-Admission-Analytics-System
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
python app.py
```

Open the local URL printed in your terminal. The application requires the model, metadata, and CSV files to remain alongside `app.py`.

## 🔌 API Overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/options` | GET | Retrieve available categories, branches, genders, seat types, colleges, and dataset counts |
| `/api/predict` | POST | Return filtered college recommendations and admission analytics |
| `/api/download` | POST | Export selected recommendation rows as CSV |

### Example prediction request

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"percentile":85,"limit":10,"margin":3}'
```

Optional filters include `category`, `gender`, `branch`, `seat_type`, and `college`. Use valid filter values returned by `/api/options`.

## 🌐 Live Demo

**[Launch the MHT-CET College Predictor](https://mht-cet-college-predictor-admission.onrender.com/)**

The live demo is hosted on Render. A free hosting instance may take some time to wake up after inactivity.

## 📷 Screenshots

Add application screenshots to an `assets/` directory, then uncomment and update the examples below:

```markdown
<!-- ![Search and filters](assets/search-filters.png) -->
<!-- ![College recommendations](assets/college-recommendations.png) -->
<!-- ![Admission analytics](assets/admission-analytics.png) -->
```

## 💡 Practical Applications

- Shortlisting engineering colleges and branches before CAP preference planning.
- Exploring historical admission cutoff trends across categories and seat types.
- Comparing safer and more ambitious options based on percentile differences.
- Demonstrating applied machine learning, data analytics, REST APIs, and deployment in an end-to-end portfolio project.

## 🚀 Future Enhancements

- Integrate newer officially published CAP-round cutoff datasets.
- Add year-wise and round-wise comparisons.
- Include clear data provenance, last-updated dates, and validation checks.
- Provide explainable model predictions and uncertainty estimates.
- Publish a reproducible training notebook and expanded regression evaluation (MAE, RMSE, R²).
- Improve accessibility and mobile usability.

## 👨‍💻 Developer

**Yash Chaudhari**  
Aspiring Data Scientist & Data Analyst

- **GitHub:** [yashchaudhari372-lab](https://github.com/yashchaudhari372-lab)
- **LinkedIn:** [yashchaudhari08](https://www.linkedin.com/in/yashchaudhari08/)

---

<div align="center">

**Built to make historical MHT-CET admission data easier to explore and understand.**

⭐ If you find this project useful, consider starring the repository.

</div>
