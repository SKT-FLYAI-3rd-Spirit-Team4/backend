from audiocraft.models import musicgen
import scipy
import time


def generate_prompt(options : dict):
    return 'a ' + options['genre'] + ' track with ' + ' and '.join(options['instrument']) + ' at ' +\
            options['speed'] + ' bpm ' + options['mood']

def load_model(model_size = 'small', duration = 30):
    model = musicgen.MusicGen.get_pretrained(model_size, device='cuda')
    model.set_generation_params(duration=duration)
    return model

def genearate_music(prompt, model, sampling_rate = 32000, save_path = ''):
    output_audio = model.generate([
        prompt
    ], 
        progress=True)
    scipy.io.wavfile.write(save_path, rate=sampling_rate, data=output_audio[0, 0].cpu().numpy())
    return output_audio

if __name__ == "__main__":
    model = load_model()
    # An 80s driving pop song with heavy drums and synth pads in the background
    options = {'genre' : 'classic', 'instrument' : ['piano'],
               'speed' : 'slow', 'mood' : 'calm'}
    # prompt = generate_prompt(options)
    prompt = 'This song features an electric guitar as the main instrument. The guitar plays a descending run in the beginning then plays an arpeggiated chord followed by a double stop hammer on to a higher note and a descending slide followed by a descending chord run. The percussion plays a simple beat using rim shots. The percussion plays in common time. The bass plays only one note on the first count of each bar. The piano plays backing chords. There are no voices in this song. The mood of this song is relaxing. This song can be played in a coffee shop.'
    file_name = prompt[:10].replace(' ', '_') + '.wav'
    print(file_name)
    start = time.time()
    genearate_music(prompt, model, save_path=file_name)
    print('spending time :', time.time() - start)