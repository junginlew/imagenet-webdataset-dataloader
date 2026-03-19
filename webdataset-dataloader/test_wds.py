import time
import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 data 폴더
src_dir = os.path.dirname(os.path.dirname(current_dir))  # 두 칸 위인 src 폴더
sys.path.append(src_dir)

from aidall_seg.data.imagenet_wds import ImageNetWDSDataModule

def test_pipeline():
    print(" WebDataset 스트리밍 파이프라인 테스트를 시작합니다.")

    # 데이터 모듈 초기화
    datamodule = ImageNetWDSDataModule(
        data_dir="",           
        train_batch_size=16,   
        val_batch_size=16,
        train_num_workers=4,    
        v2=False         
    )

    # Setup 실행 (S3 Presigned URL 발급 및 파이프라인 조립)
    print(" S3 인증 및 임시 URL(Presigned URL) 발급 중...")
    start_time = time.time()
    datamodule.setup(stage="fit")
    print(f"Setup 완료 ({time.time() - start_time:.2f}초 소요)")

    #  Train DataLoader 테스트
    print("\n Train DataLoader에서 첫 번째 배치(16장)를 스트리밍해옵니다...")
    train_loader = datamodule.train_dataloader()
    
    loader_start_time = time.time()
    
    # batch 1개만 꺼내고 멈춤
    for batch_idx, (images, labels) in enumerate(train_loader): #for문이 시작될때 가져오기 시작함
        print(f"\n데이터 로드 성공 ({time.time() - loader_start_time:.2f}초 소요)")
        print(f"이미지 텐서 형태 (B, C, H, W) : {images.shape}")
        print(f"정답 라벨 형태 (B)           : {labels.shape}")
        print(f"이미지 데이터 타입           : {images.dtype}")
        print(f"라벨 데이터 타입             : {labels.dtype}")
        print(f"첫 5개 이미지의 정답 라벨    : {labels[:5].tolist()}")
        break  

    print("\n테스트 완료")

if __name__ == "__main__":
    test_pipeline()
