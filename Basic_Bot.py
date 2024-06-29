from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=os.environ.get("OPEN_AI_KEY"),
    project="Customer_Service"
)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
  
@app.post("/chat", response_model = ChatResponse)
async def chat(request: ChatRequest):
    try:
        completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": request.message,
                   "role": "system", "content": "You are a helpful assistant designed to output JSON.",
                   "role": "assistant", "content": "You are a helpful customer center staff"
                   }],
        temperature=0.75,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.75,
        presence_penalty=0.5
        )
        answer = completion.choices[0].message.content
        return {"response": answer}
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__": 
    import uvicorn 
    uvicorn.run(app, host="localhost", port=8000)