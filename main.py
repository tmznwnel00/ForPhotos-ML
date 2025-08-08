import os
import pandas as pd
from analysis import get_num_people, get_gender_distribution, get_pose_type
from metadata import create_dataframe, add_data

csv_path = 'metadata.csv'

# 기존 CSV 불러오기 또는 새로 생성
if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
    df = pd.read_csv(csv_path)
    print("[INFO] 기존 metadata.csv 불러오기 완료.")
else:
    df = create_dataframe()
    print("[INFO] 새 데이터프레임 생성.")

# 분석할 사진들
image_folder = 'photos'
image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

# 각 사진 분석 후 DataFrame에 추가
for i, filename in enumerate(image_files, start=1):
    image_path = os.path.join(image_folder, filename)
    print(f"[INFO] 분석 중: {filename}")

    df = add_data(
        df,
        photo_id=len(df) + 1,
        num_people=get_num_people(image_path),
        gender_distribution=get_gender_distribution(image_path),
        pose_type=get_pose_type(image_path)
    )

# 결과 출력 및 저장
print("[INFO] 최종 데이터프레임:")
print(df)

df.to_csv(csv_path, index=False)
print(f"[INFO] metadata.csv 저장 완료 (총 {len(df)}건).")