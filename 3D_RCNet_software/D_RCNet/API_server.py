from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import torch
from io import BytesIO
from D_RCNet.inference import predict

app = FastAPI()


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    try:
        # 正确读取npy文件
        content = await file.read()
        data = np.load(BytesIO(content))  # 使用BytesIO解析

        # 数据维度校验
        if data.shape != (32, 1, 200, 27, 27):
            raise ValueError(f"数据维度需为 (32,1,200,27,27)，当前维度 {data.shape}")

        tensor_data = torch.from_numpy(data).float()
        result = predict(tensor_data)
        return JSONResponse({
            "prediction": result.numpy().tolist(),
            "class": int(np.argmax(result))
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9001)


if __name__ == "__main__":
    main()