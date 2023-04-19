from __future__ import annotations
import argparse
import os
import configparser
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
import PIL.Image
import datetime
import pyexiv2
#import shutil
from Utils import dbimutils


def create_config_file():
    config = configparser.ConfigParser()

    # 파일에서 설정값 로드
    if os.path.exists('config.ini'):
        config.read('config.ini')

    # 모델 설정 추가
    if not config.has_section('model'):
        config.add_section('model')
    if not config.has_option('model', 'selected_model'):
        config.set('model', 'selected_model', 'ViT')
    if not config.has_option('model', 'threshold'):
        config.set('model', 'threshold', '0.2')

    # 일반 설정 추가
    if not config.has_section('settings'):
        config.add_section('settings')
    #if not config.has_option('settings', 'backup'):
    #    config.set('settings', 'backup', 'false')
    if not config.has_option('settings', 'replace_tags'):
        config.set('settings', 'replace_tags', 'false')
    if not config.has_option('settings', 'modify_utime'):
        config.set('settings', 'modify_utime', 'false')

    # 파일에 설정 저장
    with open('config.ini', 'w', encoding='utf-8') as configfile:
        config.write(configfile)
create_config_file()

# config.ini
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')
selected_model = config.get("model", "selected_model")#, fallback="ViT")
threshold = float(config.get("model", "threshold"))#, fallback="0.2"))
#backup = config.get('settings', 'backup')#, fallback="false")
replace_tags = config.get('settings', 'replace_tags')#, fallback="false")
modify_utime = config.get('settings', 'modify_utime')#, fallback="false")


# 이미 처리된 파일 목록을 로드합니다.
def load_processed_files(processed_files_list):
    if os.path.exists(processed_files_list):
        with open(processed_files_list, 'r', encoding='utf-8') as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()
    return processed_files

processed_files_list = 'processed_files.txt'
processed_files = load_processed_files(processed_files_list)



# 코드의 메타데이터를 정의합니다.
TITLE = "WaifuDiffusion v1.4 Tags"
DESCRIPTION = """
Demo for:
- [SmilingWolf/wd-v1-4-swinv2-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)
- [SmilingWolf/wd-v1-4-convnext-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)
- [SmilingWolf/wd-v1-4-vit-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2)
"""

HF_TOKEN = ""
SWIN_MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
VIT_MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# 인자를 분석하는 함수를 정의합니다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.85)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()

# 모델을 로드하는 함수를 정의합니다.
def load_model(model_repo: str, model_filename: str) -> rt.InferenceSession:
    path = huggingface_hub.hf_hub_download(
        model_repo, model_filename, use_auth_token=HF_TOKEN
    )
    model = rt.InferenceSession(path)
    return model

# 사용할 모델을 변경하는 함수를 정의합니다.
def change_model(model_name):
    global loaded_models

    if model_name == "SwinV2":
        model = load_model(SWIN_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "ConvNext":
        model = load_model(CONV_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "ViT":
        model = load_model(VIT_MODEL_REPO, MODEL_FILENAME)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")

    loaded_models[model_name] = model
    return loaded_models[model_name]

# 레이블을 로드하는 함수를 정의합니다.
def load_labels() -> list[str]:
    path = huggingface_hub.hf_hub_download(
        SWIN_MODEL_REPO, LABEL_FILENAME, use_auth_token=HF_TOKEN
    )
    df = pd.read_csv(path)

    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

#경로를 설정
while True:
    def get_folder_path():
        if os.path.exists("path.txt"):
            with open("path.txt", "r", encoding="utf-8") as f:
                folder_path = f.read()
                folder_path = os.path.abspath(folder_path)

                print("불러온 경로는 다음과 같습니다:")
                print(folder_path)
                answer = input("이 경로가 맞습니까? Y/n): ")
                while answer.lower() not in ["y", ""]:
                    folder_path = input("폴더 경로를 다시 입력하세요: ")
                    folder_path = os.path.abspath(folder_path)
                    with open("path.txt", "w", encoding="utf-8") as f:
                        f.write(folder_path)
                    print("입력한 경로는 다음과 같습니다:")
                    print(folder_path)
                    answer = input("이 경로가 맞습니까? (Y/n): ")

        else:
            folder_path = input("폴더 경로를 입력하세요: ")
            folder_path = os.path.abspath(folder_path)
            with open("path.txt", "w", encoding="utf-8") as f:
                f.write(folder_path)

            while not os.path.exists(folder_path):
                print("잘못된 경로입니다.")
                folder_path = input("폴더 경로를 다시 입력하세요: ")
                folder_path = os.path.abspath(folder_path)
                with open("path.txt", "w", encoding="utf-8") as f:
                    f.write(folder_path)

                print("입력한 경로는 다음과 같습니다:")
                print(folder_path)
                answer = input("이 경로가 맞습니까? (Y/n): ")
                while answer.lower() not in ["y", ""]:
                    folder_path = input("폴더 경로를 다시 입력하세요: ")
                    folder_path = os.path.abspath(folder_path)
                    with open("path.txt", "w", encoding="utf-8") as f:
                        f.write(folder_path)
                    print("입력한 경로는 다음과 같습니다:")
                    print(folder_path)
                    answer = input("이 경로가 맞습니까? (Y/n): ")

            with open("path.txt", "w", encoding="utf-8") as f:
                f.write(folder_path)

        return folder_path
    folder_path = get_folder_path()

    # 하위 디렉토리도 포함할지 확인합니다.
    include_subdirs = input("하위 디렉토리도 포함합니까? (y/N) (기본값 N):").lower() not in ["n", ""]

    # 폴더 경로를 순회하며 이미지 파일을 찾습니다.
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        # 하위 디렉토리를 포함하지 않으려면, root를 folder_path와 같은 경우만 처리하도록 합니다.
        if not include_subdirs and root != folder_path:
            continue

        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".psd")):
                # 이미지 파일을 발견하면 절대 경로를 리스트에 추가합니다.
                image_files.append(os.path.join(root, file))

    # 로그 파일 경로 생성
    log_folder = "log"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    log_path = os.path.join(log_folder, log_filename)

    # 작업이 완료된 파일과 실패한 파일의 개수를 저장할 변수 초기화
    success_count = 0
    skipped_count = 0
    fail_count = 0
    total_files = len(image_files)


    # 이미지를 입력받아 예측 결과를 반환하는 함수를 정의합니다.
    def predict(
        image: PIL.Image.Image,
        model_name: str,
        general_threshold: float,
        character_threshold: float,
        tag_names: list[str],
        rating_indexes: list[np.int64],
        general_indexes: list[np.int64],
        character_indexes: list[np.int64],
    ):
        global loaded_models

        model = loaded_models[model_name]
        if model is None:
            model = change_model(model_name)

        _, height, width, _ = model.get_inputs()[0].shape

        # Alpha to white
        try:
            image = image.convert("RGBA")
            new_image = PIL.Image.new("RGBA", image.size, "WHITE")
            new_image.paste(image, mask=image)
            image = new_image.convert("RGB")
            image = np.asarray(image)

            # PIL RGB to OpenCV BGR
            image = image[:, :, ::-1]

            image = dbimutils.make_square(image, height)
            image = dbimutils.smart_resize(image, height)
            image = image.astype(np.float32)
            image = np.expand_dims(image, 0)

            input_name = model.get_inputs()[0].name
            label_name = model.get_outputs()[0].name
            probs = model.run([label_name], {input_name: image})[0]

            labels = list(zip(tag_names, probs[0].astype(float)))

            # First 4 labels are actually ratings: pick one with argmax
            ratings_names = [labels[i] for i in rating_indexes]
            rating = dict(ratings_names)

            # Then we have general tags: pick any where prediction confidence > threshold
            general_names = [labels[i] for i in general_indexes]
            general_res = [x for x in general_names if x[1] > general_threshold]
            general_res = dict(general_res)

            # Everything else is characters: pick any where prediction confidence > threshold
            character_names = [labels[i] for i in character_indexes]
            character_res = [x for x in character_names if x[1] > character_threshold]
            character_res = dict(character_res)

            b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
            a = (
                ", ".join(list(b.keys()))
                .replace("_", " ")
                .replace("(", "\(")
                .replace(")", "\)")
            )
            c = ", ".join(list(b.keys()))
            rating = max(rating, key=rating.get)
            rating = "rating:" + rating
            cd = dict(sorted(character_res.items(), key=lambda item: item[1], reverse=True))
            character_res = ", ".join(list(cd.keys()))
            return (a, c, rating, character_res, general_res)
        except OSError:
            return None, None, None, None

    #가중치를 로드합니다
    general_threshold = threshold
    character_threshold = threshold

    # 로그를 프린트합니다
    def log_print(message, log_path):
        print(message, flush=True)
        with open(log_path, "a") as log_file:
            print(message, file=log_file)

    # 예측된 태그를 Metadata에 입력합니다
    if __name__ == "__main__":
        global loaded_models
        loaded_models = {"SwinV2": None, "ConvNext": None, "ViT": None}

        change_model(selected_model)

        tag_names, rating_indexes, general_indexes, character_indexes = load_labels()

        for i, filename in enumerate(image_files):
            input_image_path = os.path.join(folder_path, filename)
            # 이미지 파일이 이미 처리되었는지 확인하고 건너뜁니다.
            if filename in processed_files:
                log_print(f"[건너뜀] ({i+1}/{total_files}) 이미 처리된 파일입니다: {filename}", log_path)
                skipped_count += 1  # 카운트를 증가시킵니다.
                continue

            try:
                input_image = PIL.Image.open(input_image_path)
                
            except PIL.UnidentifiedImageError:
                log_print(f"[에러] ({i+1}/{total_files}) {filename} 이미지를 건너뜁니다: 이미지 파일이 손상되었거나 인식할 수 없습니다.", log_path)
                fail_count += 1
                continue

            result = predict(
                input_image,
                selected_model,
                general_threshold,
                character_threshold,
                tag_names,
                rating_indexes,
                general_indexes,
                character_indexes,
            )

            # 예측된 태그
            if result[0] is None or result[1] is None or result[2] is None or result[3] is None:
                log_print(f"[에러] ({i+1}/{total_files}) {filename} 이미지 변환 실패: 이미지 파일이 손상되었습니다.", log_path)
                fail_count += 1
                continue

            predicted_tags = [result[2]] + result[3].split(', ') + sorted(result[1].split(', '))
        
            try:
                img = pyexiv2.Image(input_image_path, encoding='cp949')
                #utime 저장
                old_time = os.path.getmtime(input_image_path)
                '''
                def backup_file_with_structure(input_image_path, backup_folder, log_path):
                    # 원본 파일의 상대 경로를 계산
                    rel_path = os.path.relpath(input_image_path, folder_path)

                    # 백업 파일 경로 생성
                    backup_path = os.path.join(backup_folder, rel_path)

                    # 백업 폴더 생성 (상위 폴더 포함)
                    backup_file_folder = os.path.dirname(backup_path)
                    os.makedirs(backup_file_folder, exist_ok=True)

                    # 원본 파일을 백업 폴더로 복사 (파일 수정한 날짜를 보존)
                    try:
                        shutil.copy2(input_image_path, backup_path)
                    except Exception as e:
                        print(f"[에러] {input_image_path} 파일의 백업 실패: {e}", file=open(log_path, "a"))
                        return False

                    # 백업 파일 수정한 날짜 보존
                    os.utime(backup_path, (os.path.getatime(input_image_path), os.path.getmtime(input_image_path)))
                    return True


                if backup.lower() == "true":
                    backup_folder = folder_path + "_backup"
                    success = backup_file_with_structure(input_image_path, backup_folder, log_path)
                    if not success:
                        img.close()
                        continue
                '''
                # 기존 태그를 남기고 새로운 태그를 추가하는 경우
                if replace_tags == "false":
                    xmp_data = img.read_xmp()
                    if xmp_data is not None and "Xmp.dc.subject" in xmp_data:
                        existing_tags = xmp_data["Xmp.dc.subject"]
                        if existing_tags is not None:
                            predicted_tags = list(set(existing_tags + predicted_tags))

                # XMP 데이터 수정
                img.modify_xmp({'Xmp.dc.subject': predicted_tags})
                log_print(f"[완료] ({i+1}/{total_files}) (예측된 태그 개수: {len(predicted_tags)}) {filename} 파일 작업 완료", log_path)
                
                # utime를 수정
                if modify_utime == "false":	
                    os.utime(input_image_path, (old_time, old_time))

                # 이미지 파일 닫기
                img.close()
                success_count += 1

                # 작업이 완료된 이미지 파일을 저장하는 부분을 추가합니다.
                with open(processed_files_list, 'a', encoding='utf-8') as f:
                    f.write(f'{filename}\n')
                    processed_files.add(filename)
                


            except Exception as e:
                log_print(f"[에러] ({i+1}/{total_files}) {filename} 파일 작업 실패: {e}", log_path)
                fail_count += 1


    with open(log_path, "r") as f:
        log_text = f.read()


    # 작업이 완료된 파일과 실패한 파일의 개수를 CUI에 출력
    log_print(f"success: {success_count}/{total_files}, fail: {fail_count}, skip: {skipped_count}", log_path)
    user_input = input("작업을 계속 진행하시려면 'c'를 입력하고, 종료하려면 'q'를 입력하세요: ")

    if user_input.lower() == 'q':
        break

# 프로그램 종료
#input("Enter키를 누르면 프로그램이 종료됩니다...")