from fastapi import FastAPI
from server.routes import router as NoteRouter
from server.schema import PostStrokeBody
from server.utils.thinning import thinning
from server.model.gan import Generator
import numpy as np
from PIL import Image
from server.utils.data import normalize
import shortuuid
import os
from dotenv import load_dotenv
import vtracer

app = FastAPI()

load_dotenv()


@app.post("/", tags=["Root"])
async def read_root(body: PostStrokeBody):
    model = Generator()
    img = np.zeros([256, 256, 3], dtype=np.uint8)
    img.fill(255)  # numpy array!
    im = Image.fromarray(img)  # convert numpy array to image
    pixels = body.strokes

    strokes = body.strokes
    stroke_array = np.array(strokes)
    # make 1bit to 8bit
    stroke_array = stroke_array

    model.load_weights(path="server/model/NanumPenScript_gen.npz")
    # change 1 to True and 0 to False
    stroke_array = stroke_array.astype(bool)
    skeleton = thinning(stroke_array)

    x = 0
    y = 0
    for row in skeleton:
        y = 0
        for pixel in row:
            if pixel == 1:
                im.putpixel((y, x), (0, 0, 0))
            y += 1
        x += 1

    rgb = np.asarray(im.convert("RGB"))
    array = rgb / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    normalized = normalize(array, mean, std)
    transposed = np.transpose(normalized, (2, 0, 1))
    generated = model(np.array([transposed]))
    gen_result = ((generated.data + 1) / 2) * 255
    gen_result = np.transpose(gen_result, (0, 2, 3, 1))
    environment = os.getenv("SERVER_ENVIRONMENT", "production")

    save_dir = "/tmp/"

    if environment == "development":
        save_dir = "./samples/"

    gen_random_ids = []
    svg_results = []
    for results in gen_result:
        sample_image = Image.fromarray(results.astype(np.uint8))
        # save temporarily
        random_id = shortuuid.uuid()
        file_dir = f"{save_dir}{random_id}.png"
        sample_image.save(file_dir)
        svg_dir = f"{save_dir}{random_id}.svg"

        vtracer.convert_image_to_svg_py(
            file_dir,
            svg_dir,
            colormode="binary",  # ["color"] or "binary"
            hierarchical="stacked",  # ["stacked"] or "cutout"
            mode="spline",  # ["spline"] "polygon", or "none"
            filter_speckle=50,  # default: 4
            color_precision=10,  # default: 6
            layer_difference=16,  # default: 16
            corner_threshold=60,  # default: 60
            length_threshold=4,  # in [3.5, 10] default: 4.0
            max_iterations=10,  # default: 10
            splice_threshold=80,  # default: 45
            path_precision=2,  # default: 8
        )
        with open(svg_dir, "r") as f:
            svg = f.read()
            gen_random_ids.append(random_id)
            svg_results.append(svg)

    return {"result": svg_results}


app.include_router(NoteRouter, prefix="/note")
