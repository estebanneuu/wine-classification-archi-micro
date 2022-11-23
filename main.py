from fastapi import FastAPI

app = FastAPI(
    title='WinApi',
    description='APIs for red wine classification',
    version='0.1'
)


@app.get("/api")
async def root():
    return {"message": "Hi there !"}



@app.on_event("startup")
async def startup():
    print("Application is starting...")


@app.on_event("shutdown")
async def shutdown():
    print("Application is closing...")
