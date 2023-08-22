from audiocraft.models import musicgen
import scipy
import time
import pandas as pd
import openpyxl
import os
from datetime import datetime


def generate_prompt(options: dict):
    # a chill blues with piano at fast bpm
    return 'a ' + options['mood'] + ' ' + options['genre'] + ' with ' + options['instrument'] + ' at ' +\
           options['speed'] + ' bpm '

def load_model(model_size='small', duration=30):
    model = musicgen.MusicGen.get_pretrained(model_size, device='cuda')
    model.set_generation_params(duration=duration)
    return model

def generate_music(prompt, model, sampling_rate=32000):
    output_audio = model.generate([prompt], progress=True)
    return output_audio[0, 0].cpu().numpy()


if __name__ == "__main__":
    model_size = 'small'
    duration = 30 # 음악 생성 시간
    sampling_rate = 32000
    model = load_model(model_size = model_size, duration=duration)


    # option_list 생성
    file = open("1_genre.txt", "r")
    file2 = open("2_instrument.txt", "r")
    file3 = open("3_speed.txt", "r")
    file4 = open("4_mood.txt", "r")

    genre = file.readlines()
    instrument = file2.readlines()
    speed = file3.readlines()
    mood = file4.readlines()

    option_list = []

    for sub_genre in genre:
            option_genre = sub_genre.strip()
            for sub_instrument in instrument:
                    option_instrument = sub_instrument.strip()
                    for sub_speed in speed:
                        option_speed = sub_speed.strip()
                        for sub_mood in mood:
                               option_mood = sub_mood.strip()
                               option = {'genre': option_genre, 'instrument': option_instrument, 'speed': option_speed, 'mood': option_mood}
                               option_list.append(option)
    
    excel_path = "./music_generation_results.xlsx"
    
    try:
        # 기존 엑셀 파일이 있으면 로드
        result_df = pd.read_excel(excel_path)
    except FileNotFoundError:
        # 기존 엑셀 파일이 없으면 빈 DataFrame을 생성
        result_df = pd.DataFrame(columns=['Genre', 'Instrument', 'Speed', 'Mood', 'Model Size', 'Duration', 'Sampling Rate', 'Generation Time', 'Generation Date', 'Insight'])

    for options in option_list:
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
            'Instrument': options['instrument'],
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
        file_name = f"./{id}_{options['genre']}_{options['instrument']}_{options['speed']}_{options['mood']}.wav"
        save_path = file_name.replace(' ', '_')
      
        # 오디오 생성 결과 test 폴더 따로 지정
        output_dir_path = '/home/sunmin/audiocraft/train_test01/' + save_path
      
        scipy.io.wavfile.write(output_dir_path, rate=sampling_rate, data=generated_audio)
    
    # 결과를 엑셀 파일로 저장합니다.
    result_df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")
