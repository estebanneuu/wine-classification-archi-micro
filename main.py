from fastapi import FastAPI

app = FastAPI(
    title='WinApi',
    description='APIs for red wine classification',
    version='0.1'
)


@app.get("/api")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
