from audiocraft.models import musicgen
import scipy
import time
import pandas as pd
import openpyxl
from datetime import datetime


def generate_prompt(options: dict):
    return 'a ' + options['genre'] + ' track with ' + ' and '.join(options['instrument']) + ' at ' +\
           options['speed'] + ' bpm ' + options['mood']

def load_model(model_size='small', duration=30):
    model = musicgen.MusicGen.get_pretrained(model_size, device='cuda')
    model.set_generation_params(duration=duration)
    return model

def generate_music(prompt, model, sampling_rate=32000):
    output_audio = model.generate([prompt], progress=True)
    return output_audio[0, 0].cpu().numpy()

if __name__ == "__main__":
    model_size = 'small'
    duration = 30
    sampling_rate = 32000
    model = load_model(model_size = model_size, duration=duration)
    
    # 여러 개의 options를 정의
    options_list = [
        {'genre': 'Jazz', 'instrument': ['slap bass', 'powerful saxophone'], 'speed': 'slow', 'mood': 'powerful'},
        {'genre': 'Rock', 'instrument': ['electric guitar', 'drums'], 'speed': 'fast', 'mood': 'energetic'}
        # 추가적인 options를 원하는 만큼 추가 가능
    ]
    
    excel_path = "./music_generation_results.xlsx"
    
    try:
        # 기존 엑셀 파일이 있으면 로드
        result_df = pd.read_excel(excel_path)
    except FileNotFoundError:
        # 기존 엑셀 파일이 없으면 빈 DataFrame을 생성
        result_df = pd.DataFrame(columns=['Genre', 'Instrument', 'Speed', 'Mood', 'Model Size', 'Duration', 'Sampling Rate', 'Generation Time', 'Generation Date', 'Insight'])

    for options in options_list:
        prompt = generate_prompt(options)
        print("Prompt:", prompt)
        
        start = time.time()
        generated_audio = generate_music(prompt, model, sampling_rate=sampling_rate)
        generation_time = time.time() - start
        print('Spending time:', generation_time)

        id = datetime.now().strftime('%Y%m%d%H%M%S')
        print(id)
        
        # 결과를 딕셔너리로 저장합니다.
        result_dict = {
            'Genre': options['genre'],
            'Instrument': ', '.join(options['instrument']),
            'Speed': options['speed'],
            'Mood': options['mood'],
            'Model Size' : model_size,
            'Duration' : duration,
            'Sampling Rate' : sampling_rate,
            'Generation Time' : str(duration),
            'Generation Date(id)': id
        }
        result_df.loc[len(result_df)] = result_dict
        # 오디오를 저장하려면 아래 코드 주석을 해제하세요.
        file_name = f"./{id}_{options['genre']}_{options['speed']}_{options['mood']}.wav"
        save_path = file_name.replace(' ', '_')
        scipy.io.wavfile.write(save_path, rate=sampling_rate, data=generated_audio)
    
    # 결과를 엑셀 파일로 저장합니다.
    result_df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")
