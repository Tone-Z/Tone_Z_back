from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PERSONAL_COLOR_TYPES = {
    "spring-light": {
        "tone": "warm",
        "toneName": "웜톤",
        "seasonName": "봄 웜 라이트",
        "description": "맑고 밝은 따뜻한 파스텔 색상이 잘 어울리는 타입입니다.",
        "bestColors": ["#FFD6A5", "#FFB5C2", "#FFF1A8", "#B8E6C1"],
        "worstColors": ["#2E2E2E", "#4B3F72", "#6B4E3D"]
    },
    "spring-bright": {
        "tone": "warm",
        "toneName": "웜톤",
        "seasonName": "봄 웜 브라이트",
        "description": "생기 있고 선명한 따뜻한 색상이 얼굴을 또렷하게 살려주는 타입입니다.",
        "bestColors": ["#FF6F61", "#FFB000", "#7ED957", "#00C2A8"],
        "worstColors": ["#6D6875", "#8D99AE", "#4A4A4A"]
    },
    "spring-soft": {
        "tone": "warm",
        "toneName": "웜톤",
        "seasonName": "봄 웜 소프트",
        "description": "부드럽고 산뜻한 따뜻한 색상이 자연스럽게 어울리는 타입입니다.",
        "bestColors": ["#F6C8A8", "#E8B7A2", "#D9C589", "#A8CFA0"],
        "worstColors": ["#111827", "#6B21A8", "#0F4C5C"]
    },
    "autumn-muted": {
        "tone": "warm",
        "toneName": "웜톤",
        "seasonName": "가을 웜 뮤트",
        "description": "차분하고 탁도가 있는 따뜻한 색상이 고급스럽게 어울리는 타입입니다.",
        "bestColors": ["#C58B5C", "#A98467", "#8FA86E", "#B08D57"],
        "worstColors": ["#BDEBFF", "#D8C7FF", "#FF4FA3"]
    },
    "autumn-deep": {
        "tone": "warm",
        "toneName": "웜톤",
        "seasonName": "가을 웜 딥",
        "description": "깊고 진한 따뜻한 색상이 분위기와 대비감을 살려주는 타입입니다.",
        "bestColors": ["#6B3F2A", "#8B5E34", "#556B2F", "#7A4E2D"],
        "worstColors": ["#FADADD", "#B7D7F0", "#EAF4FF"]
    },
    "summer-light": {
        "tone": "cool",
        "toneName": "쿨톤",
        "seasonName": "여름 쿨 라이트",
        "description": "밝고 부드러운 차가운 파스텔 색상이 맑게 어울리는 타입입니다.",
        "bestColors": ["#B7D7F0", "#D8C7FF", "#F6CFE1", "#C9E4DE"],
        "worstColors": ["#7A3E1D", "#D96C2C", "#6B3F2A"]
    },
    "summer-muted": {
        "tone": "cool",
        "toneName": "쿨톤",
        "seasonName": "여름 쿨 뮤트",
        "description": "회색기가 섞인 차분한 차가운 색상이 세련되게 어울리는 타입입니다.",
        "bestColors": ["#9FB3C8", "#B8A9C9", "#D6AFC2", "#8FA3A3"],
        "worstColors": ["#FFB000", "#FF6F61", "#7A4E2D"]
    },
    "winter-bright": {
        "tone": "cool",
        "toneName": "쿨톤",
        "seasonName": "겨울 쿨 브라이트",
        "description": "차갑고 선명한 고채도 색상이 강한 생기를 주는 타입입니다.",
        "bestColors": ["#0057FF", "#E6007E", "#00A3FF", "#FFFFFF"],
        "worstColors": ["#C58B5C", "#A98467", "#D9C589"]
    },
    "winter-deep": {
        "tone": "cool",
        "toneName": "쿨톤",
        "seasonName": "겨울 쿨 딥",
        "description": "어둡고 깊은 차가운 색상과 강한 대비가 잘 어울리는 타입입니다.",
        "bestColors": ["#111827", "#1E1B4B", "#B0005A", "#EAF4FF"],
        "worstColors": ["#FFD6A5", "#A98467", "#F6C8A8"]
    },
}


PERSONAL_COLOR_PROFILES = {
    # value: HSV V, saturation: HSV S, lightness: LAB L, chroma: LAB a/b distance from neutral.
    # 봄 타입은 웜톤 중에서도 밝고 비교적 맑은 축입니다.
    # spring-light: 밝기와 명도가 높고 채도는 중간 정도인 따뜻한 타입입니다.
    "spring-light": {"value": 214, "saturation": 58, "lightness": 178, "chroma": 24, "warmth": 24},
    # spring-bright: 밝기, 명도, 채도가 모두 높은 맑고 선명한 따뜻한 타입입니다.
    "spring-bright": {"value": 220, "saturation": 86, "lightness": 174, "chroma": 37, "warmth": 30},
    # spring-soft: 봄의 밝음은 남아 있지만 채도와 대비가 낮아 부드러운 따뜻한 타입입니다.
    "spring-soft": {"value": 202, "saturation": 45, "lightness": 164, "chroma": 18, "warmth": 18},
    # 가을 타입은 웜톤 중 명도가 낮거나 채도가 차분한 축입니다.
    # autumn-muted: 중간 이하 명도, 낮은 채도, 노란기/흙빛이 느껴지는 따뜻한 타입입니다.
    "autumn-muted": {"value": 174, "saturation": 43, "lightness": 145, "chroma": 18, "warmth": 24},
    # autumn-deep: 웜톤이면서 밝기가 낮고 색감의 깊이가 강한 타입입니다.
    "autumn-deep": {"value": 144, "saturation": 66, "lightness": 122, "chroma": 31, "warmth": 30},
    # 여름 타입은 쿨톤 중 밝거나 부드럽고 채도가 낮은 축입니다.
    # summer-light: 밝고 명도가 높으며 채도가 낮아 부드러운 차가운 타입입니다.
    "summer-light": {"value": 210, "saturation": 38, "lightness": 178, "chroma": 14, "warmth": -16},
    # summer-muted: 쿨톤이면서 채도와 대비가 낮고 회색기가 도는 타입입니다.
    "summer-muted": {"value": 178, "saturation": 34, "lightness": 154, "chroma": 12, "warmth": -18},
    # 겨울 타입은 쿨톤 중 선명도와 대비가 강한 축입니다.
    # winter-bright: 밝기와 채도가 모두 높아 또렷한 차가운 타입입니다.
    "winter-bright": {"value": 214, "saturation": 84, "lightness": 166, "chroma": 34, "warmth": -28},
    # winter-deep: 밝기와 명도는 낮고 채도/대비가 강한 깊은 차가운 타입입니다.
    "winter-deep": {"value": 136, "saturation": 72, "lightness": 112, "chroma": 30, "warmth": -30},
}


def rgb_to_lab(r: int, g: int, b: int):
    rgb_pixel = np.uint8([[[r, g, b]]])
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)[0][0]

    return {
        "l": int(lab_pixel[0]),
        "a": int(lab_pixel[1]),
        "b": int(lab_pixel[2])
    }


def rgb_to_hsv(r: int, g: int, b: int):
    rgb_pixel = np.uint8([[[r, g, b]]])
    hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0][0]

    return {
        "h": int(hsv_pixel[0]),
        "s": int(hsv_pixel[1]),
        "v": int(hsv_pixel[2])
    }


def analyze_tone(lab, r: int, g: int, b: int):
    lab_a = lab["a"]
    lab_b = lab["b"]

    # OpenCV LAB의 a/b는 128이 중립입니다.
    # b가 128보다 높으면 노란기, a가 b보다 상대적으로 높으면 붉은기/푸른기 쪽으로 해석합니다.
    # 피부 사진은 조명 영향을 많이 받으므로 RGB의 R-B 차이도 보조 지표로 사용합니다.
    yellow_bias = lab_b - 128
    red_bias = lab_a - 128
    rgb_warm_bias = (r - b) / 4
    warmth_score = yellow_bias * 1.25 + rgb_warm_bias - max(red_bias - yellow_bias, 0) * 0.7

    if warmth_score >= 7:
        return "warm", "웜톤", warmth_score

    return "cool", "쿨톤", warmth_score


def analyze_personal_color(avg_r, avg_g, avg_b):
    lab = rgb_to_lab(avg_r, avg_g, avg_b)
    hsv = rgb_to_hsv(avg_r, avg_g, avg_b)
    tone, tone_name, warmth_score = analyze_tone(lab, avg_r, avg_g, avg_b)

    brightness = hsv["v"]
    saturation = hsv["s"]
    lightness = lab["l"]
    chroma = abs(lab["a"] - 128) + abs(lab["b"] - 128)

    # 권장 threshold 제안(OpenCV 기준):
    # brightness(HSV V): 205 이상 밝음, 165~204 중간, 164 이하 어두움
    # saturation(HSV S): 70 이상 선명, 40~69 보통, 39 이하 부드러움/뮤트
    # lightness(LAB L): 170 이상 고명도, 140~169 중명도, 139 이하 저명도
    # chroma(LAB a/b 중립점 128과의 거리): 28 이상 고채도/고대비, 16~27 중간, 15 이하 저채도
    # warmth_score: 7 이상 웜, 7 미만 쿨로 판단합니다.
    #
    # 최종 타입은 위 지표를 딱 잘라 if로만 나누지 않고,
    # 9개 타입별 목표 프로필과 현재 피부색의 거리를 비교해 가장 가까운 타입을 선택합니다.
    # 이렇게 하면 경계값 근처의 사용자가 갑자기 전혀 다른 결과로 튀는 현상을 줄일 수 있습니다.
    candidates = {
        season: profile
        for season, profile in PERSONAL_COLOR_PROFILES.items()
        if PERSONAL_COLOR_TYPES[season]["tone"] == tone
    }

    best_season = None
    best_score = None

    for season, profile in candidates.items():
        score = (
            abs(brightness - profile["value"]) * 0.22
            + abs(saturation - profile["saturation"]) * 0.28
            + abs(lightness - profile["lightness"]) * 0.28
            + abs(chroma - profile["chroma"]) * 0.18
            + abs(warmth_score - profile["warmth"]) * 0.14
        )

        if best_score is None or score < best_score:
            best_score = score
            best_season = season

    result = PERSONAL_COLOR_TYPES[best_season].copy()
    result.update({
        "tone": tone,
        "toneName": tone_name,
        "season": best_season
    })

    return result


@app.get("/")
def root():
    return {"message": "Tone-Z Backend"}


@app.post("/diagnosis")
async def diagnosis(file: UploadFile = File(...)):
    contents = await file.read()

    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    mp_face_mesh = mp.solutions.face_mesh
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return {"error": "얼굴 랜드마크 검출 실패"}

        h, w, _ = image.shape
        face_landmarks = results.multi_face_landmarks[0]

        left_cheek = face_landmarks.landmark[123]
        right_cheek = face_landmarks.landmark[352]
        forehead = face_landmarks.landmark[10]

        left_x = int(left_cheek.x * w)
        left_y = int(left_cheek.y * h)
        right_x = int(right_cheek.x * w)
        right_y = int(right_cheek.y * h)
        forehead_x = int(forehead.x * w)
        forehead_y = int(forehead.y * h)

        def get_avg_color(cx, cy, size=30):
            x1 = max(cx - size, 0)
            y1 = max(cy - size, 0)
            x2 = min(cx + size, w)
            y2 = min(cy + size, h)

            roi = image[y1:y2, x1:x2]
            b, g, r, _ = cv2.mean(roi)

            return {
                "r": int(r),
                "g": int(g),
                "b": int(b),
                "hex": "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))
            }

        left_color = get_avg_color(left_x, left_y)
        right_color = get_avg_color(right_x, right_y)
        forehead_color = get_avg_color(forehead_x, forehead_y, size=25)

        avg_r = int((left_color["r"] + right_color["r"] + forehead_color["r"]) / 3)
        avg_g = int((left_color["g"] + right_color["g"] + forehead_color["g"]) / 3)
        avg_b = int((left_color["b"] + right_color["b"] + forehead_color["b"]) / 3)
        avg_hex = "#{:02X}{:02X}{:02X}".format(avg_r, avg_g, avg_b)
        avg_lab = rgb_to_lab(avg_r, avg_g, avg_b)
        avg_hsv = rgb_to_hsv(avg_r, avg_g, avg_b)
        personal_color = analyze_personal_color(avg_r, avg_g, avg_b)

        return {
            "message": "피부색 추출 성공",
            "landmarkCount": len(face_landmarks.landmark),
            "leftCheek": {
                "x": left_x,
                "y": left_y
            },
            "rightCheek": {
                "x": right_x,
                "y": right_y
            },
            "forehead": {
                "x": forehead_x,
                "y": forehead_y
            },
            "averageSkinColor": {
                "r": avg_r,
                "g": avg_g,
                "b": avg_b,
                "hex": avg_hex,
                "lab": avg_lab,
                "hsv": avg_hsv
            },
            "leftCheekColor": left_color,
            "rightCheekColor": right_color,
            "foreheadColor": forehead_color,
            **personal_color
        }
