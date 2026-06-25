# Deepfake Image Detection — Hand-Crafted Feature Engineering + Decision Tree

딥러닝 없이 25개 수작업 이미지 피처를 설계·추출하고 의사결정나무로 REAL/FAKE를 분류하는 머신러닝 프로젝트.

## Overview

이미지의 물리적 속성(RGB 채널 상관관계, 노이즈 패턴, 텍스처, 색공간)을 직접 수치화해 딥페이크 특유의 미세한 왜곡을 탐지한다.

## Feature Engineering (25개)

| 그룹 | 피처 | 수 |
|---|---|---|
| RGB 채널 | 채널 간 상관계수, 표준편차 | 9 |
| 노이즈 / 텍스처 | SNR, 첨도, Laplacian, Sobel 엣지 선명도 | 9 |
| HSV 색공간 | 밝기·채도·색조 통계, 색상 엔트로피 | 7 |

- ThreadPoolExecutor 16스레드 병렬 이미지 다운로드 및 피처 추출
- EDA에서 분류력 최상으로 예측한 `brightness_mean`이 실제 모델 루트 노드로 선택됨

## Results

| 지표 | 값 |
|---|---|
| Accuracy | 0.98 |
| Balanced Accuracy | 0.9757 |

GridSearchCV 6파라미터, 5-Fold CV 하이퍼파라미터 최적화 수행.

## Tech Stack

- Python, scikit-learn, OpenCV, Pillow
- Google Colab

## Course

한신대학교 머신러닝 이해와 활용 (ML_2601) 기말 프로젝트
