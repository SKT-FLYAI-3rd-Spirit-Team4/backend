#@title Imports
from diffusers import DiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
import numpy as np
from io import BytesIO
from IPython.display import Audio
import scipy

pipe = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1")
pipe = pipe.to("cuda")

#@title Define a `predict` function

params = SpectrogramParams()
converter = SpectrogramImageConverter(params)

def predict(prompt, negative_prompt):
    spec = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=3072,
    ).images[0]
    
    wav = converter.audio_from_spectrogram_image(image=spec)
    wav.export('output.wav', format='wav')
    return 'output.wav', spec, wav

# 768
#@title Run with Colab interface
prompt = "solo piano piece, classical"#@param {type:"string"}
prompt = "This song features an electric guitar as the main instrument. The guitar plays a descending run in the beginning then plays an arpeggiated chord followed by a double stop hammer on to a higher note and a descending slide followed by a descending chord run. The percussion plays a simple beat using rim shots. The percussion plays in common time. The bass plays only one note on the first count of each bar. The piano plays backing chords. There are no voices in this song. The mood of this song is relaxing. This song can be played in a coffee shop."
negative_prompt = "drums"#@param {type:"string"}

path, spec, wav = predict(prompt, negative_prompt)

Audio('output.wav')
scipy.io.wavfile.write('output2.wav', rate=32000, data=np.array(wav.get_array_of_samples()))
