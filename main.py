from fastapi import FastAPI,Request
import uvicorn
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/truefoundary")
def read_root():
    return {"Hello": "fine-tunned-RoBERTa"}

def get_model():
    model_x = AutoModelForSequenceClassification.from_pretrained("velvrix/truefoundary_sentimental_RoBERTa")
    tokenizer_x = AutoTokenizer.from_pretrained("velvrix/truefoundary_sentimental_RoBERTa")
    return tokenizer_x,model_x

d = {
    
  1:'Postive Sentiment',
  0:'Negative Sentiment'
}

tokenizer,model = get_model()

@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    print(data)
    if 'text' in data: #checking for the payload (json format) case sensitive
        user_input = data['text']
        test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
        output = model(**test_sample)
        y_pred = np.argmax(output.logits.detach().numpy(),axis=1)  
        response = {"Recieved Text": user_input,"Prediction": d[y_pred[0]]} #dictionary
    else:
        response = {"Recieved Text": "No Text Found"}
    return response

if __name__ == "__main__":
    uvicorn.run("main:app",host='0.0.0.0', port=8080, reload=True)
    #above uvicorn.run takes the name of our app instance. 
    #convention is first we need to give the file name where our app instance is there and the name of the app.








