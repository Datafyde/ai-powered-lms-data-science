from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load model once at startup
model = SentenceTransformer('all-MiniLM-L6-v2')
course_list = [
    "Introduction to Python Programming",
    "Machine Learning Basics",
    "Data Analysis with Excel",
    "Natural Language Processing with Transformers",
    "Deep Learning with PyTorch",
    "Database Management Systems",
    "Web Development with Django",
    "Data Visualization with Tableau",
    "Cloud Computing Fundamentals",
    "Artificial Intelligence for Beginners"
]
course_embeddings = model.encode(course_list, convert_to_tensor=True)


class RequestData(BaseModel):
    input: str


@app.post("/recommend")
def recommend_courses(data: RequestData):
    if not data.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    user_embedding = model.encode(data.input, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, course_embeddings)[0]
    top_results = cosine_scores.topk(3)

    recommendations = [
        {"course": course_list[idx], "score": float(score)}
        for score, idx in zip(top_results.values, top_results.indices)
    ]
    return {"recommendations": recommendations}