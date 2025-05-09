from fastapi import FastAPI,Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # to allow our api to connect to anything such as html file


class bmioutput(BaseModel):#make the code simpler and easy to understand it
    bmi:float
    message:str
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],#what is allowed
    allow_headers=["*"],
    )


@app.get("/")
def Hi():
    return{"message":"Hel"}

@app.get("/code")
def code(number1:float=Query(... ,gt=20,lt=200),
        number2:float=Query(... ,gt=20,lt=200)):
    bmi=number1/number2
    if number1<number2:
        message="results: 2"
    else:
        message="results: 1"
    return bmioutput(bmi=bmi,message=message)
