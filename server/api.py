from fastapi import FastAPI
from server.routes import router as NoteRouter
from .model.gan import Generator
import numpy as np

app = FastAPI()


@app.get("/", tags=["Root"])
async def read_root():
    Generator.load_weights("server/model/pixel_gen_weights.npz")
    batch_size = 3
    z = np.random.randn(batch_size, 100, 1, 1).astype(np.float32)
    x = Generator(z)
    gen_result = ((x.data + 1) / 2) * 255
    gen_result = np.transpose(gen_result, (0, 2, 3, 1))
    # the result is 50 * 50
    # original data was 500 * 500 with 25 pixel size
    # if the result is 50 * 50, then the pixel size is 2.5

    results = []

    # TODO: implement for loop
    for gen_i in range(len(gen_result)):
        test = []
        for i in range(25):
            row = []
            for j in range(25):
                # average r, g, b separate,y
                r = np.mean(gen_result[gen_i]
                            [i * 2:i * 2 + 2, j * 2:j * 2 + 2, 0])
                g = np.mean(gen_result[gen_i]
                            [i * 2:i * 2 + 2, j * 2:j * 2 + 2, 1])
                b = np.mean(gen_result[gen_i]
                            [i * 2:i * 2 + 2, j * 2:j * 2 + 2, 2])
                row.append([r, g, b])
            test.append(row)
        results.append(test)
    return {
        "result": gen_result.tolist(),
    }

app.include_router(NoteRouter, prefix="/note")
