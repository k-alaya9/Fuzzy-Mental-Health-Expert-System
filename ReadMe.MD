# 🧠 Fuzzy Mental Health Expert System

A fuzzy logic-based expert system to assess the **mental health status** of university students using academic, lifestyle, and social indicators.

The system predicts four psychological outcomes:

- **Depression**
- **Anxiety**
- **Isolation**
- **Future Insecurity**

It uses a rule-based fuzzy inference engine and compares the output against real-world data using machine learning models like Decision Trees and Random Forests.

---

## 🎯 Project Goals

- Build a fuzzy expert system for soft reasoning on mental health.
- Provide approximate, human-like decisions instead of binary predictions.
- Visualize and test the system using real student data.
- Compare fuzzy results with ML regressors for performance validation.
- Deliver the system through a Django web interface (without REST API).

---

## 🛠️ Tools & Technologies

| Technology        | Purpose                               |
|-------------------|----------------------------------------|
| **Python**        | Core development language              |
| **scikit-fuzzy**  | Fuzzy logic system and membership funcs|
| **pandas / sklearn** | Data preprocessing and evaluation   |
| **Django**        | Backend logic and templating engine    |
| **HTML + CSS**    | Frontend for simple input/output       |

---

## 📥 Inputs

The system uses the following input features:

- `academic_workload`
- `academic_pressure`
- `social_relationships`
- `average_sleep`
- `financial_concerns`
- `age`

All inputs are fuzzified into categories such as: **low**, **moderate**, **high**, etc.

---

## 📤 Outputs

The fuzzy system estimates the following outcomes:

| Output              | Fuzzy Labels                            |
|---------------------|------------------------------------------|
| **Depression**       | none, mild, moderate, severe             |
| **Anxiety**          | low, medium, high                        |
| **Isolation**        | connected, neutral, isolated             |
| **Future Insecurity**| secure, unsure, very insecure            |

---

## 📊 ML Comparison (Optional)

The outputs were also predicted using regression models for comparison:

- **Decision Tree Regressor**
- **Random Forest Regressor**

Model performance was evaluated using:

- `R² Score`
- `Accuracy`

---

## 🖥️ Running Locally

1. Clone the repo:

```bash
git clone https://github.com/your-username/fuzzy-mental-health.git
cd fuzzy-mental-health
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
python manage.py runserver
```

4. Open your browser at:

```
http://127.0.0.1:8000/
```

---

## 📁 Project Structure

```
fuzzy_backend/
├── diagnosis/
│   ├── templates/
│   │   └── diagnosis/
│   │       └── index.html
│   ├── static/
│   │   └── diagnosis/
│   │       └── style.css
│   ├── fuzzy_engine.py     ← Fuzzy system engine
│   ├── views.py            ← Django views for input/output
│   └── urls.py
├── manage.py
└── requirements.txt
```

---

## ✅ Features

* 40 Fuzzy rules across 6 input variables.
* Multi-output diagnosis using Mamdani fuzzy inference.
* Clean, responsive UI using Django templates and CSS.
* Optional ML benchmarking for validation.

---

## 👨‍💻 Author

* **m Khaled Alaya** – AI student and fuzzy logic enthusiast

---

## 📝 License

This project is for academic and research purposes only.

---

## 🙏 Thanks

If you found this useful or want to contribute, feel free to fork and star ⭐ the repo.

```

---

Would you like me to also prepare the `requirements.txt` or a `.gitignore` file?
```
