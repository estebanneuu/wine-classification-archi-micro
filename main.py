from fastapi import FastAPI
from routers import predict,model

app = FastAPI(
    title='WinApi',
    description='APIs for red wine classification',
    version='0.1'
)


@app.get("/api")
async def root():
    return {"message": "Hi there !"}

app.include_router(predict.router_predict)
app.include_router(model.router_model)

@app.on_event("startup")
async def startup():
    print("Application is starting...")


@app.on_event("shutdown")
async def shutdown():
    print("Application is closing...")
