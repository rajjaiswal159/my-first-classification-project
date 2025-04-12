# Student Career Path Classifier

This project is a multi-class classification model that predicts the career path of a Computer Science student based on their skills and preferences.

## Problem Statement
Many students decide which field they want to enter, like Software Development, Data Science, or Cybersecurity—but they often struggle to choose a specific career path within that field. This confusion leads to uncertainty and poor career decisions. The goal of this project is to build a machine learning model that helps students by recommending a suitable and relevant career path based on their skills and interests.

## Project Overview
- Multi-class classification with 35 possible career paths
- Balanced dataset with features like programming skills, interests, and project types
- Used model: Decision Tree.
- Evaluated using accuracy, F1-score

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit (for UI)

## How to Run
1. Clone the repo or download the `.ipynb` file
2. Open it in Jupyter Notebook or Google Colab
3. Run all cells

## Results
- Best model: Decision Tree
- Accuracy: 96%

## License
Free to use for educational purposes


🌐 Streamlit Web App
I built an interactive web app using Streamlit so users can input their skills and interests to get a predicted career path.

⚙️ How to Run Locally:
git clone https://github.com/rajjaiswal159/my-first-classification-project.git
cd my-first-classification-project
pip install -r requirements.txt
streamlit run app.py
