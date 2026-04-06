import time
import sys
import os
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 폴더 (scripts)
project_root = os.path.dirname(current_dir)              # 한 칸 위 (프로젝트 최상위)
src_dir = os.path.join(project_root, "src")              # 최상위 폴더 안의 src 폴더
sys.path.append(src_dir)

from aidall_seg.data.imagenet_wds import ImageNetWDSDataModule

NUM_BATCHES = 50  # 처리량 측정에 사용할 배치 수


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
    start_time = time.perf_counter()
    datamodule.setup(stage="fit")
    print(f"Setup 완료 ({time.perf_counter() - start_time:.2f}초 소요)")

    #  Train DataLoader 테스트
    print(f"\n Train DataLoader에서 {NUM_BATCHES}배치를 스트리밍합니다...")
    train_loader = datamodule.train_dataloader()

    first_batch_time = None
    sustained_start = None
    sustained_images = 0
    overall_start = time.perf_counter()
    batch_throughputs = []  # 배치별 처리량 기록
    batch_start = None

    for i, (images, labels) in enumerate(train_loader): #for문이 시작될때 가져오기 시작함
        elapsed = time.perf_counter() - overall_start

        if i == 0:
            first_batch_time = elapsed
            print(f"\n[첫 번째 배치] 수신 완료 ({first_batch_time:.2f}초, 버퍼 채움 대기 포함)")
            print(f"이미지 텐서 형태 (B, C, H, W) : {images.shape}")
            print(f"정답 라벨 형태 (B)           : {labels.shape}")
            print(f"이미지 데이터 타입           : {images.dtype}")
            print(f"라벨 데이터 타입             : {labels.dtype}")
            print(f"첫 5개 이미지의 정답 라벨    : {labels[:5].tolist()}")
            print(f"\n[처리량 측정] 2번째 배치부터 {NUM_BATCHES}배치까지 측정...")
            sustained_start = time.perf_counter()
            batch_start = time.perf_counter()
        else:
            batch_elapsed = time.perf_counter() - batch_start
            batch_throughputs.append(images.shape[0] / batch_elapsed)
            sustained_images += images.shape[0]
            batch_start = time.perf_counter()

        if i + 1 >= NUM_BATCHES:
            break

    sustained_elapsed = time.perf_counter() - sustained_start
    throughput = sustained_images / sustained_elapsed
    total_elapsed = time.perf_counter() - overall_start

    print(f"\n{'=' * 45}")
    print(f"측정 결과 (배치 2~{NUM_BATCHES} 기준, 버퍼 안정화 이후)")
    print(f"{'=' * 45}")
    print(f"  첫 배치 대기 시간   : {first_batch_time:.2f}초")
    print(f"  안정화 후 처리량    : {throughput:.1f} img/s")
    print(f"  배치당 평균 시간    : {sustained_elapsed / (NUM_BATCHES - 1) * 1000:.1f} ms")
    print(f"  전체 소요 시간      : {total_elapsed:.2f}초 ({NUM_BATCHES}배치 / {NUM_BATCHES * 16}장)")
    print(f"{'=' * 45}")

    # 처리량 그래프 저장
    save_path = os.path.join(project_root, "throughput.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(2, NUM_BATCHES + 1)
    ax.plot(x, batch_throughputs, color="steelblue", linewidth=1, alpha=0.6, label="per-batch throughput")
    ax.axhline(throughput, color="tomato", linewidth=1.5, linestyle="--", label=f"avg {throughput:.1f} img/s")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Throughput (img/s)")
    ax.set_title("S3 Streaming Throughput (WebDataset)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n그래프 저장 완료: {save_path}")
    print("\n테스트 완료")

if __name__ == "__main__":
    test_pipeline()
