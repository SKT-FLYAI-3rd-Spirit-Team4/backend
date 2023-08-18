import io
import warnings

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from music_generator import genearate_music, load_model, generate_prompt

warnings.simplefilter(action='ignore', category=FutureWarning)

app = FastAPI(title="SKT FLY AI Melovision Internal Music GAN Service",
              redoc_url=None)


@app.get("/")
async def root():
    return {"message": "SKT FLY AI Melovision Internal Music GAN Service"}


@app.get("/music/generate")
async def generate_music(genre: str, instrument: str, speed: str, duration: int, title: str = None, desc: str = None):
    model = load_model(duration=duration)
    options = {
        'genre': genre,
        'instrument': instrument.split(','),
        'speed': speed
    }

    prompt = generate_prompt(options)
    output_music = genearate_music(prompt, model)
    output_music = io.BytesIO(output_music)

    return StreamingResponse(output_music)


if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=45678, reload=True)
