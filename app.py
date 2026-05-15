from flask import Flask, render_template, request
import pandas as pd
import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "resumes"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load dataset
jobs = pd.read_csv("resumes/monster_com-job_sample.csv")

# Common skills list (you can expand this)
skills_list = [
    "python", "java", "c++", "machine learning", "data science",
    "sql", "html", "css", "javascript", "deep learning",
    "excel", "communication", "teamwork", "flask", "django"
]

# Extract text
def extract_text(pdf):
    text = ""
    reader = PyPDF2.PdfReader(pdf)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text.lower()
    return text

# Extract skills
def extract_skills(text):
    found = []
    for skill in skills_list:
        if skill in text:
            found.append(skill)
    return found


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/match', methods=['POST'])
def match():
    file = request.files['resume']

    if file:
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        with open(path, "rb") as f:
            resume_text = extract_text(f)

        resume_skills = extract_skills(resume_text)

        job_descriptions = jobs['job_description'].astype(str).tolist()
        documents = [resume_text] + job_descriptions

        tfidf = TfidfVectorizer(stop_words='english').fit_transform(documents)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        jobs['score'] = scores

        top_jobs = jobs.sort_values(by='score', ascending=False).head(5)

        results = []

        for _, row in top_jobs.iterrows():
            job_text = row['job_description'].lower()
            job_skills = extract_skills(job_text)

            matched = list(set(resume_skills) & set(job_skills))
            missing = list(set(job_skills) - set(resume_skills))

            results.append({
                "description": row['job_description'],
                "score": row['score'],
                "matched": matched,
                "missing": missing
            })

        return render_template("result.html", results=results)

    return "No file uploaded"


if __name__ == '__main__':
    app.run(debug=True)