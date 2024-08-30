from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict


from src.utils import embedder, search_engine, get_content

app = FastAPI()

# Define request and response models
class SearchRequest(BaseModel):
    relief: List[str]
    positive_effects: List[str]
    query: str

class ScoreResult(BaseModel):
    relief: List[str]
    positive_effects: List[str]
    query: str

class SearchResult(BaseModel):
    title: str
    explanation: str

class ScoreItem(BaseModel):
    title: str
    relevant: bool

class ScoreRequest(BaseModel):
    items: List[ScoreItem]

# Mock database or search logic
mock_data = [
    {"title": "Strain 1", "explanation": "Helps with anxiety and stress."},
    {"title": "Strain 2", "explanation": "Good for uplifting mood."},
    {"title": "Strain 3", "explanation": "Relieves depression and insomnia."},
]

@app.post("/search", response_model=List[SearchResult])
async def search_items(request: SearchRequest):
    # Placeholder search logic based on the request
    reliefs = f"reliefs: {', '.join(request.relief)}"
    positive_effects = f"positive effects: {', '.join(request.positive_effects)}"
    query = f"{reliefs}; {positive_effects}; {request.query}"
    q = embedder.encode(query)
    result = search_engine.search(q, num_results=5)
    results = get_content([i['doc'] for i in result])
    # results = []
    # for item in mock_data:
    #     if any(relief in item['explanation'] for relief in request.relief) or \
    #        any(effect in item['explanation'] for effect in request.positive_effects) or \
    #        request.query.lower() in item['explanation'].lower():
    #         results.append(item)
    if not results:
        raise HTTPException(status_code=404, detail="No matching items found")
    
    return results


@app.post("/score")
async def score(data: ScoreRequest):
    # Process the scoring data here
    # For now, let's just return the received data
    return {"received_items": data.items}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
