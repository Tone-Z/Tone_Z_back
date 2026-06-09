# Tone-Z Backend

## 프로젝트 소개

Tone-Z는 실시간 카메라 기반 퍼스널 컬러 진단 서비스입니다.

본 레포지토리는 Tone-Z의 백엔드 서버로, 사용자의 얼굴 이미지를 분석하여 퍼스널 컬러를 진단하고 결과를 반환하는 기능을 담당합니다.

## 주요 기능

* 실시간 카메라 이미지 분석
* 얼굴 영역 검출
* 피부 색상 추출
* 퍼스널 컬러 진단
* 진단 결과 API 제공

## 기술 스택

### Backend

* Python
* FastAPI

### AI / Image Processing

* OpenCV
* MediaPipe
* NumPy
* Pillow

## 프로젝트 구조

```text
Tone_Z_back
│
├── main.py
├── requirements.txt
├── .gitignore
│
├── models
├── routers
├── services
└── utils
```

## 설치 방법

```bash
git clone [repository-url]
cd Tone_Z_back

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## 실행 방법

```bash
.\venv311\Scripts\Activate.ps1 #가상환경활성화
uvicorn main:app --reload
```

서버 실행 후:

```text
http://127.0.0.1:8000
```

API 문서:

```text
http://127.0.0.1:8000/docs
```

## 개발 목표

사용자의 얼굴 이미지를 분석하여 퍼스널 컬러를 진단하고, 진단 결과를 프론트엔드에 제공하는 AI 기반 분석 서버를 구축하는 것을 목표로 합니다.
