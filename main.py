from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

FACE_LANDMARKER_MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
_face_landmarker = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(contents: bytes):
    np_array = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)


def get_face_landmarks(rgb_image):
    global _face_landmarker

    if FACE_LANDMARKER_MODEL_PATH.exists():
        if _face_landmarker is None:
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)
                ),
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
            _face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = _face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(rgb_image)

        if not result.multi_face_landmarks:
            return None

        return result.multi_face_landmarks[0].landmark


PERSONAL_COLOR_TYPES = {
    "spring-light": {
        "tone": "warm",
        "toneName": "웜톤",
        "seasonName": "봄 웜 라이트",
        "description": "맑고 밝은 따뜻한 파스텔 색상이 잘 어울리는 타입입니다.",
        "bestColors": ["#FFD6A5", "#FFB5C2", "#FFF1A8", "#B8E6C1"],
        "worstColors": ["#2E2E2E", "#4B3F72", "#6B4E3D"],
    },
    "spring-bright": {
        "tone": "warm",
        "toneName": "웜톤",
        "seasonName": "봄 웜 브라이트",
        "description": "생기 있고 선명한 따뜻한 색상이 얼굴을 또렷하게 살려주는 타입입니다.",
        "bestColors": ["#FF6F61", "#FFB000", "#7ED957", "#00C2A8"],
        "worstColors": ["#6D6875", "#8D99AE", "#4A4A4A"],
    },
    "spring-soft": {
        "tone": "warm",
        "toneName": "웜톤",
        "seasonName": "봄 웜 소프트",
        "description": "부드럽고 산뜻한 따뜻한 색상이 자연스럽게 어울리는 타입입니다.",
        "bestColors": ["#F6C8A8", "#E8B7A2", "#D9C589", "#A8CFA0"],
        "worstColors": ["#111827", "#6B21A8", "#0F4C5C"],
    },
    "autumn-mute": {
        "tone": "warm",
        "toneName": "웜톤",
        "seasonName": "가을 웜 뮤트",
        "description": "차분하고 탁도가 있는 따뜻한 색상이 고급스럽게 어울리는 타입입니다.",
        "bestColors": ["#C58B5C", "#A98467", "#8FA86E", "#B08D57"],
        "worstColors": ["#BDEBFF", "#D8C7FF", "#FF4FA3"],
    },
    "autumn-deep": {
        "tone": "warm",
        "toneName": "웜톤",
        "seasonName": "가을 웜 딥",
        "description": "깊고 진한 따뜻한 색상이 분위기와 대비감을 살려주는 타입입니다.",
        "bestColors": ["#6B3F2A", "#8B5E34", "#556B2F", "#7A4E2D"],
        "worstColors": ["#FADADD", "#B7D7F0", "#EAF4FF"],
    },
    "summer-light": {
        "tone": "cool",
        "toneName": "쿨톤",
        "seasonName": "여름 쿨 라이트",
        "description": "밝고 부드러운 차가운 파스텔 색상이 맑게 어울리는 타입입니다.",
        "bestColors": ["#B7D7F0", "#D8C7FF", "#F6CFE1", "#C9E4DE"],
        "worstColors": ["#7A3E1D", "#D96C2C", "#6B3F2A"],
    },
    "summer-mute": {
        "tone": "cool",
        "toneName": "쿨톤",
        "seasonName": "여름 쿨 뮤트",
        "description": "회색기가 섞인 차분한 차가운 색상이 세련되게 어울리는 타입입니다.",
        "bestColors": ["#9FB3C8", "#B8A9C9", "#D6AFC2", "#8FA3A3"],
        "worstColors": ["#FFB000", "#FF6F61", "#7A4E2D"],
    },
    "winter-bright": {
        "tone": "cool",
        "toneName": "쿨톤",
        "seasonName": "겨울 쿨 브라이트",
        "description": "차갑고 선명한 고채도 색상이 강한 생기를 주는 타입입니다.",
        "bestColors": ["#0057FF", "#E6007E", "#00A3FF", "#FFFFFF"],
        "worstColors": ["#C58B5C", "#A98467", "#D9C589"],
    },
    "winter-deep": {
        "tone": "cool",
        "toneName": "쿨톤",
        "seasonName": "겨울 쿨 딥",
        "description": "어둡고 깊은 차가운 색상과 강한 대비가 잘 어울리는 타입입니다.",
        "bestColors": ["#111827", "#1E1B4B", "#B0005A", "#EAF4FF"],
        "worstColors": ["#FFD6A5", "#A98467", "#F6C8A8"],
    },
}


PERSONAL_COLOR_PROFILES = {
    # value: 피부 HSV V, saturation: 피부 HSV S, lightness: 피부 LAB L
    # chroma: 피부 LAB a/b가 중립점 128에서 떨어진 정도
    # contrast: 피부와 눈/머리의 밝기 차이, hair_value/eye_value: 눈과 머리의 어두운 정도
    "spring-light": {"tone": "warm", "value": 214, "saturation": 52, "lightness": 178, "chroma": 22, "warmth": 22, "contrast": 28, "hair_value": 150, "eye_value": 120},
    "spring-bright": {"tone": "warm", "value": 220, "saturation": 78, "lightness": 174, "chroma": 35, "warmth": 28, "contrast": 42, "hair_value": 118, "eye_value": 94},
    "spring-soft": {"tone": "warm", "value": 202, "saturation": 42, "lightness": 164, "chroma": 18, "warmth": 16, "contrast": 22, "hair_value": 145, "eye_value": 120},
    "autumn-mute": {"tone": "warm", "value": 174, "saturation": 38, "lightness": 145, "chroma": 17, "warmth": 24, "contrast": 28, "hair_value": 105, "eye_value": 92},
    "autumn-deep": {"tone": "warm", "value": 144, "saturation": 62, "lightness": 122, "chroma": 30, "warmth": 30, "contrast": 48, "hair_value": 62, "eye_value": 55},
    "summer-light": {"tone": "cool", "value": 210, "saturation": 34, "lightness": 178, "chroma": 14, "warmth": -14, "contrast": 24, "hair_value": 140, "eye_value": 110},
    "summer-mute": {"tone": "cool", "value": 178, "saturation": 30, "lightness": 154, "chroma": 12, "warmth": -18, "contrast": 20, "hair_value": 122, "eye_value": 95},
    "winter-bright": {"tone": "cool", "value": 214, "saturation": 74, "lightness": 166, "chroma": 32, "warmth": -24, "contrast": 72, "hair_value": 42, "eye_value": 36},
    "winter-deep": {"tone": "cool", "value": 136, "saturation": 66, "lightness": 112, "chroma": 28, "warmth": -28, "contrast": 76, "hair_value": 34, "eye_value": 30},
}


def rgb_to_lab(r: int, g: int, b: int):
    rgb_pixel = np.uint8([[[r, g, b]]])
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)[0][0]

    return {
        "l": int(lab_pixel[0]),
        "a": int(lab_pixel[1]),
        "b": int(lab_pixel[2]),
    }


def rgb_to_hsv(r: int, g: int, b: int):
    rgb_pixel = np.uint8([[[r, g, b]]])
    hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0][0]

    return {
        "h": int(hsv_pixel[0]),
        "s": int(hsv_pixel[1]),
        "v": int(hsv_pixel[2]),
    }


def make_color(r: int, g: int, b: int):
    return {
        "r": int(r),
        "g": int(g),
        "b": int(b),
        "hex": "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b)),
        "hsv": rgb_to_hsv(int(r), int(g), int(b)),
    }


def mean_rgb_from_pixels(pixels):
    if pixels is None or len(pixels) == 0:
        return None

    avg_bgr = np.mean(pixels, axis=0)
    b, g, r = avg_bgr
    return make_color(r, g, b)


def analyze_tone(lab, r: int, g: int, b: int):
    lab_a = lab["a"]
    lab_b = lab["b"]

    # OpenCV LAB의 a/b는 128이 중립입니다.
    # warm/cool은 머리카락 색을 쓰지 않고 피부 LAB를 우선합니다.
    # b가 충분히 높고 a보다 확실히 높으면 노란기/웜으로 봅니다.
    # a와 b가 비슷하거나 b가 낮으면 조명에 의한 노란기일 수 있어 cool/neutral 가능성을 열어둡니다.
    # RGB R-B는 아주 약한 보조 지표로만 사용해 따뜻한 조명에 과하게 끌려가지 않게 합니다.
    yellow_bias = lab_b - 128
    red_bias = lab_a - 128
    yellow_over_red = lab_b - lab_a
    rgb_warm_bias = (r - b) * 0.08
    warmth_score = (
        yellow_bias * 1.35
        + yellow_over_red * 1.15
        + rgb_warm_bias
        - max(red_bias - yellow_bias, 0) * 0.85
    )

    if warmth_score > 8:
        return "warm", "웜톤", warmth_score
    if warmth_score < -6:
        return "cool", "쿨톤", warmth_score

    return "neutral", "뉴트럴", warmth_score


def get_contrast_level(score: int):
    if score >= 70:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def calculate_contrast(skin_color, eye_color=None, hair_color=None):
    skin_v = skin_color["hsv"]["v"]
    reference_values = []

    if eye_color:
        reference_values.append(eye_color["hsv"]["v"])
    if hair_color:
        reference_values.append(hair_color["hsv"]["v"])

    if not reference_values:
        return 0, "low"

    dark_feature_v = min(reference_values)
    avg_feature_v = sum(reference_values) / len(reference_values)
    value_gap = max(0, skin_v - dark_feature_v)
    balanced_gap = max(0, skin_v - avg_feature_v)
    black_feature_bonus = max(0, 90 - dark_feature_v)
    score = int(min(100, value_gap * 0.55 + balanced_gap * 0.25 + black_feature_bonus * 0.35))

    return score, get_contrast_level(score)


def build_analysis_reason(result, skin_hsv, eye_color, hair_color, contrast_score, contrast_level):
    season = result["season"]
    confidence = result.get("confidence", 0)

    if season.startswith("winter") and contrast_level == "high":
        if skin_hsv["v"] < 185:
            return f"피부 밝기가 낮고 눈동자/머리카락과의 대비가 강합니다. contrastScore가 {contrast_score}로 높아 겨울 쿨 딥 가능성을 높게 판단했습니다."
        return f"피부는 밝은 편이고 눈동자/머리카락이 어두워 대비가 강합니다. contrastScore가 {contrast_score}로 높아 겨울 쿨 계열을 우선 판단했습니다."
    if season.startswith("summer") and contrast_level == "low":
        return f"피부 LAB의 노란기가 강하지 않고 채도가 낮으며 대비가 낮습니다. 봄 웜보다 여름 쿨의 부드러운 후보가 더 가까워 {season}로 판단했습니다."
    if season == "spring-soft":
        return f"피부 LAB에서 따뜻한 노란기가 확인되지만 대비와 채도가 높지 않아 봄 웜 소프트로 판단했습니다. confidence는 {confidence}입니다."
    if season.startswith("spring"):
        return f"피부 LAB b값이 높고 명도가 밝아 웜 계열 점수가 높았습니다. 대비와 채도를 함께 비교해 {season}로 판단했습니다."
    if season == "autumn-mute":
        return "피부의 따뜻한 기운은 있으나 채도와 밝기가 차분하고 대비가 강하지 않아 가을 웜 뮤트로 판단했습니다."
    if season == "autumn-deep":
        return "피부 밝기가 낮고 전체 대비가 깊은 편이라 가을 웜 딥으로 판단했습니다."
    if result["tone"] == "cool":
        return f"피부 LAB b값이 강하지 않고 대비/채도 조건을 함께 비교한 결과 쿨 계열인 {season}가 가장 가까웠습니다."

    return "피부색, 눈동자색, 머리카락색, 대비감이 경계에 가까워 가장 가까운 퍼스널 컬러 타입으로 판단했습니다."


def analyze_personal_color(avg_r, avg_g, avg_b, eye_color=None, hair_color=None):
    lab = rgb_to_lab(avg_r, avg_g, avg_b)
    hsv = rgb_to_hsv(avg_r, avg_g, avg_b)
    skin_color = make_color(avg_r, avg_g, avg_b)
    tone, tone_name, warmth_score = analyze_tone(lab, avg_r, avg_g, avg_b)

    brightness = hsv["v"]
    saturation = hsv["s"]
    lightness = lab["l"]
    chroma = abs(lab["a"] - 128) + abs(lab["b"] - 128)
    contrast_score, contrast_level = calculate_contrast(skin_color, eye_color, hair_color)
    hair_value = hair_color["hsv"]["v"] if hair_color else brightness
    eye_value = eye_color["hsv"]["v"] if eye_color else brightness

    # 분류 기준 요약(OpenCV 기준)
    # 1. warm/cool: 피부 LAB b(노란기), LAB a(붉은기), RGB R-B 차이로 warmth_score를 계산합니다.
    #    -6~8은 neutral로 보고 spring 계열로 바로 보내지 않습니다.
    # 2. light/deep: 피부 HSV V/LAB L뿐 아니라 머리카락 HSV V를 함께 봅니다.
    #    머리카락은 warm/cool 판단에는 쓰지 않고 contrast/deep 판단에만 사용합니다.
    # 3. bright/mute: 피부 HSV S와 LAB chroma가 높으면 bright, 낮고 대비도 낮으면 mute/soft로 봅니다.
    # 4. contrast: 피부와 눈/머리의 밝기 차이가 크면 winter 계열 가능성을 강하게 보정합니다.
    # 권장 threshold: V 205+ 밝음, S 70+ 선명, LAB L 170+ 고명도,
    # contrastScore 70+ high, 35~69 medium, 34 이하 low입니다.

    is_dark_features = hair_value <= 78 and eye_value <= 78
    is_bright_skin = brightness >= 185 and lightness >= 155
    is_high_contrast = contrast_level == "high"
    is_low_saturation = saturation <= 45 and chroma <= 22
    is_weak_yellow = lab["b"] < 136 or (lab["b"] - lab["a"]) <= 2

    # 겨울 쿨 보정: 피부만 밝게 잡히면 봄/여름으로 흔들릴 수 있습니다.
    # 밝은 피부 + 검은 눈/머리 + 높은 대비는 실제 퍼스널 컬러 이론에서 겨울 계열의 핵심 신호입니다.
    if is_bright_skin and is_high_contrast and (is_dark_features or min(hair_value, eye_value) <= 86):
        if saturation >= 48 or brightness >= 205:
            best_season = "winter-bright"
        else:
            best_season = "winter-deep"

        result = PERSONAL_COLOR_TYPES[best_season].copy()
        result.update({
            "tone": "cool",
            "toneName": "쿨톤",
            "season": best_season,
            "contrastScore": contrast_score,
            "contrastLevel": contrast_level,
            "confidence": 0.9 if contrast_score >= 82 else 0.82,
        })
        result["analysisReason"] = build_analysis_reason(
            result, hsv, eye_color, hair_color, contrast_score, contrast_level
        )
        return result

    # 밝은 피부 + 낮은 대비 + 낮은/중간 채도는 라이트보다 소프트한 인상이 우선입니다.
    # LAB b가 강하지 않은 neutral/cool 피부라면 spring-soft보다 summer-light/summer-mute를 우선 비교합니다.
    if is_bright_skin and contrast_level == "low" and saturation <= 55 and tone != "warm":
        best_season = "summer-mute" if is_low_saturation and brightness < 205 else "summer-light"
        result = PERSONAL_COLOR_TYPES[best_season].copy()
        result.update({
            "tone": result["tone"],
            "toneName": result["toneName"],
            "season": best_season,
            "contrastScore": contrast_score,
            "contrastLevel": contrast_level,
            "confidence": 0.76 if tone == "neutral" else 0.82,
        })
        result["analysisReason"] = build_analysis_reason(
            result, hsv, eye_color, hair_color, contrast_score, contrast_level
        )
        return result

    # 웜톤이라도 피부 밝기/명도가 중간 이하라면 봄보다 가을 후보를 우선합니다.
    # 따뜻한 조명이나 갈색 염색머리 때문에 spring-light/spring-soft로 과분류되는 것을 막습니다.
    if tone == "warm" and brightness <= 185 and lightness <= 158:
        best_season = "autumn-deep" if contrast_level == "high" or brightness <= 145 else "autumn-mute"
        result = PERSONAL_COLOR_TYPES[best_season].copy()
        result.update({
            "tone": result["tone"],
            "toneName": result["toneName"],
            "season": best_season,
            "contrastScore": contrast_score,
            "contrastLevel": contrast_level,
            "confidence": 0.82 if best_season == "autumn-mute" else 0.78,
        })
        result["analysisReason"] = build_analysis_reason(
            result, hsv, eye_color, hair_color, contrast_score, contrast_level
        )
        return result

    if tone == "neutral":
        candidates = PERSONAL_COLOR_PROFILES.copy()
    else:
        candidates = {
            season: profile
            for season, profile in PERSONAL_COLOR_PROFILES.items()
            if profile["tone"] == tone
        }

    # 피부 톤은 웜으로 보이지만 대비가 아주 높고 눈/머리가 검다면 겨울 후보도 같이 비교합니다.
    if is_high_contrast and (is_dark_features or min(hair_value, eye_value) <= 86):
        candidates["winter-bright"] = PERSONAL_COLOR_PROFILES["winter-bright"]
        candidates["winter-deep"] = PERSONAL_COLOR_PROFILES["winter-deep"]

    # 피부가 밝고 LAB b가 강하지 않으며 채도가 낮으면 summer 후보를 열어둡니다.
    # 따뜻한 조명 때문에 spring-soft/spring-light로 과분류되는 것을 줄이기 위한 보정입니다.
    if is_bright_skin and is_low_saturation and is_weak_yellow:
        candidates["summer-light"] = PERSONAL_COLOR_PROFILES["summer-light"]
        candidates["summer-mute"] = PERSONAL_COLOR_PROFILES["summer-mute"]

    best_season = None
    best_score = None
    ranked_scores = []

    for season, profile in candidates.items():
        score = (
            abs(brightness - profile["value"]) * 0.18
            + abs(saturation - profile["saturation"]) * 0.20
            + abs(lightness - profile["lightness"]) * 0.20
            + abs(chroma - profile["chroma"]) * 0.14
            + abs(warmth_score - profile["warmth"]) * 0.10
            + abs(contrast_score - profile["contrast"]) * 0.22
            + abs(hair_value - profile["hair_value"]) * 0.13
            + abs(eye_value - profile["eye_value"]) * 0.13
        )

        # 높은 대비는 winter, 낮은 대비는 soft/mute 계열에 가산점을 줍니다.
        if contrast_level == "high" and season.startswith("winter"):
            score -= 18
        if contrast_level == "high" and season.startswith("spring"):
            score += 10
        if contrast_level == "low" and season in ["summer-mute", "spring-soft"]:
            score -= 12
        if contrast_level == "low" and season == "summer-light":
            score -= 8
        if contrast_level == "low" and season in ["spring-light", "spring-bright"]:
            score += 7
        if contrast_level == "low" and season.startswith("winter"):
            score += 18
        if tone == "neutral" and season.startswith("spring"):
            score += 9
        if tone == "neutral" and season.startswith("summer"):
            score -= 7
        if is_bright_skin and is_low_saturation and is_weak_yellow and season.startswith("summer"):
            score -= 12
        if is_bright_skin and is_low_saturation and is_weak_yellow and season.startswith("spring"):
            score += 12
        if lab["b"] >= 140 and (lab["b"] - lab["a"]) >= 4 and season.startswith("spring"):
            score -= 8

        ranked_scores.append((score, season))
        if best_score is None or score < best_score:
            best_score = score
            best_season = season

    ranked_scores.sort()
    second_score = ranked_scores[1][0] if len(ranked_scores) > 1 else best_score + 20
    score_gap = max(0, second_score - best_score)
    confidence = 0.55 + min(0.35, score_gap / 45)

    if tone == "neutral":
        confidence -= 0.08
    if contrast_level == "high" and best_season.startswith("winter"):
        confidence += 0.08
    if is_bright_skin and is_low_saturation and is_weak_yellow and best_season.startswith("summer"):
        confidence += 0.05

    confidence = round(max(0.35, min(0.95, confidence)), 2)

    result = PERSONAL_COLOR_TYPES[best_season].copy()
    result.update({
        "tone": result["tone"],
        "toneName": result["toneName"],
        "season": best_season,
        "contrastScore": contrast_score,
        "contrastLevel": contrast_level,
        "confidence": confidence,
    })
    result["analysisReason"] = build_analysis_reason(
        result, hsv, eye_color, hair_color, contrast_score, contrast_level
    )

    return result


def get_avg_color(image, cx, cy, size=30):
    h, w, _ = image.shape
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
        "hex": "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b)),
    }


def extract_iris_color(image, landmarks, indices, width, height):
    points = np.array(
        [[int(landmarks[i].x * width), int(landmarks[i].y * height)] for i in indices],
        dtype=np.int32,
    )
    cx, cy = np.mean(points, axis=0).astype(int)
    radius = max(3, int(max(np.ptp(points[:, 0]), np.ptp(points[:, 1])) / 2) + 2)

    x1 = max(cx - radius, 0)
    y1 = max(cy - radius, 0)
    x2 = min(cx + radius + 1, width)
    y2 = min(cy + radius + 1, height)
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    yy, xx = np.ogrid[y1:y2, x1:x2]
    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 홍채의 반사광/흰자 영역은 V가 매우 높습니다.
    # V 185 이상이면서 채도가 낮은 픽셀, RGB가 모두 밝은 픽셀은 제외합니다.
    reflection_mask = ((hsv_roi[:, :, 2] >= 185) & (hsv_roi[:, :, 1] <= 70)) | np.all(roi >= 175, axis=2)
    valid_mask = circle_mask & ~reflection_mask
    pixels = roi[valid_mask]

    if len(pixels) < 6:
        pixels = roi[circle_mask]

    return mean_rgb_from_pixels(pixels)


def extract_eye_color(image, landmarks, width, height):
    left_iris = extract_iris_color(image, landmarks, [468, 469, 470, 471, 472], width, height)
    right_iris = extract_iris_color(image, landmarks, [473, 474, 475, 476, 477], width, height)
    colors = [color for color in [left_iris, right_iris] if color]

    if not colors:
        return None

    avg_r = int(sum(color["r"] for color in colors) / len(colors))
    avg_g = int(sum(color["g"] for color in colors) / len(colors))
    avg_b = int(sum(color["b"] for color in colors) / len(colors))
    return make_color(avg_r, avg_g, avg_b)


def extract_hair_color(image, landmarks, width, height):
    xs = [int(point.x * width) for point in landmarks]
    ys = [int(point.y * height) for point in landmarks]
    face_x1 = max(min(xs), 0)
    face_x2 = min(max(xs), width)
    face_y1 = max(min(ys), 0)
    face_y2 = min(max(ys), height)
    face_height = max(1, face_y2 - face_y1)

    forehead_y = int(landmarks[10].y * height)
    x_padding = int((face_x2 - face_x1) * 0.12)
    x1 = max(face_x1 - x_padding, 0)
    x2 = min(face_x2 + x_padding, width)
    y1 = max(forehead_y - int(face_height * 0.38), 0)
    y2 = max(forehead_y - int(face_height * 0.06), y1 + 1)
    y2 = min(y2, height)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    value = hsv_roi[:, :, 2]
    saturation = hsv_roi[:, :, 1]

    # 머리카락 후보는 보통 피부/배경보다 어둡습니다.
    # 너무 밝은 피부, 조명, 배경을 제외하기 위해 V가 낮은 픽셀을 우선 사용합니다.
    dark_threshold = min(145, int(np.percentile(value, 35)) + 20)
    hair_mask = (value <= dark_threshold) & ~((value >= 170) & (saturation <= 45))
    pixels = roi[hair_mask]

    if len(pixels) < 30:
        flat_roi = roi.reshape(-1, 3)
        flat_value = value.reshape(-1)
        darkest_count = max(20, int(len(flat_value) * 0.18))
        darkest_indices = np.argsort(flat_value)[:darkest_count]
        pixels = flat_roi[darkest_indices]

    return mean_rgb_from_pixels(pixels)


def is_valid_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < 45:
        return False, {
            "brightness": round(brightness, 2),
            "blurScore": round(blur_score, 2),
            "reason": "too_dark",
        }
    if brightness > 235:
        return False, {
            "brightness": round(brightness, 2),
            "blurScore": round(blur_score, 2),
            "reason": "too_bright",
        }
    if blur_score < 20:
        return False, {
            "brightness": round(brightness, 2),
            "blurScore": round(blur_score, 2),
            "reason": "too_blurry",
        }

    return True, {
        "brightness": round(brightness, 2),
        "blurScore": round(blur_score, 2),
        "reason": "ok",
    }


def analyze_single_frame(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = get_face_landmarks(rgb_image)

    if not landmarks:
        return {"error": "얼굴 랜드마크 검출 실패"}

    h, w, _ = image.shape

    left_cheek = landmarks[123]
    right_cheek = landmarks[352]
    forehead = landmarks[10]

    left_x = int(left_cheek.x * w)
    left_y = int(left_cheek.y * h)
    right_x = int(right_cheek.x * w)
    right_y = int(right_cheek.y * h)
    forehead_x = int(forehead.x * w)
    forehead_y = int(forehead.y * h)

    left_color = get_avg_color(image, left_x, left_y)
    right_color = get_avg_color(image, right_x, right_y)
    forehead_color = get_avg_color(image, forehead_x, forehead_y, size=25)

    avg_r = int(left_color["r"] * 0.4 + right_color["r"] * 0.4 + forehead_color["r"] * 0.2)
    avg_g = int(left_color["g"] * 0.4 + right_color["g"] * 0.4 + forehead_color["g"] * 0.2)
    avg_b = int(left_color["b"] * 0.4 + right_color["b"] * 0.4 + forehead_color["b"] * 0.2)
    avg_hex = "#{:02X}{:02X}{:02X}".format(avg_r, avg_g, avg_b)
    avg_lab = rgb_to_lab(avg_r, avg_g, avg_b)
    avg_hsv = rgb_to_hsv(avg_r, avg_g, avg_b)

    eye_color = extract_eye_color(image, landmarks, w, h)
    hair_color = extract_hair_color(image, landmarks, w, h)
    skin_color = make_color(avg_r, avg_g, avg_b)
    contrast_score, contrast_level = calculate_contrast(skin_color, eye_color, hair_color)
    personal_color = analyze_personal_color(avg_r, avg_g, avg_b, eye_color, hair_color)

    return {
        "message": "피부색 추출 성공",
        "landmarkCount": len(landmarks),
        "leftCheek": {"x": left_x, "y": left_y},
        "rightCheek": {"x": right_x, "y": right_y},
        "forehead": {"x": forehead_x, "y": forehead_y},
        "averageSkinColor": {
            "r": avg_r,
            "g": avg_g,
            "b": avg_b,
            "hex": avg_hex,
            "lab": avg_lab,
            "hsv": avg_hsv,
        },
        "leftCheekColor": left_color,
        "rightCheekColor": right_color,
        "foreheadColor": forehead_color,
        "eyeColor": eye_color,
        "hairColor": hair_color,
        "contrastScore": contrast_score,
        "contrastLevel": contrast_level,
        **personal_color,
    }


@app.get("/")
def root():
    return {"message": "Tone-Z Backend"}


@app.post("/diagnosis")
async def diagnosis(file: UploadFile = File(...)):
    contents = await file.read()
    image = decode_image(contents)

    if image is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    return analyze_single_frame(image)


@app.post("/diagnosis/video")
async def diagnosis_video(files: list[UploadFile] = File(...)):
    frame_count = len(files)
    valid_results = []
    skipped_frames = []

    for index, file in enumerate(files):
        contents = await file.read()
        image = decode_image(contents)

        if image is None:
            skipped_frames.append({"index": index, "reason": "invalid_image"})
            continue

        is_valid, quality = is_valid_frame(image)
        if not is_valid:
            skipped_frames.append({"index": index, **quality})
            continue

        result = analyze_single_frame(image)
        if "error" in result:
            skipped_frames.append({"index": index, "reason": "face_not_detected"})
            continue

        valid_results.append(result)

    valid_frame_count = len(valid_results)

    if valid_frame_count < 2:
        return {
            "error": "분석 가능한 프레임이 부족합니다.",
            "frameCount": frame_count,
            "validFrameCount": valid_frame_count,
            "skippedFrames": skipped_frames,
        }

    season_votes = {}
    for result in valid_results:
        season = result["season"]
        confidence = result.get("confidence", 0.5)
        season_votes[season] = round(season_votes.get(season, 0) + confidence, 4)

    final_season = max(season_votes, key=season_votes.get)
    total_vote_score = sum(season_votes.values())
    final_confidence = round(season_votes[final_season] / total_vote_score, 2) if total_vote_score else 0
    selected_type = PERSONAL_COLOR_TYPES[final_season]

    avg_r = int(sum(result["averageSkinColor"]["r"] for result in valid_results) / valid_frame_count)
    avg_g = int(sum(result["averageSkinColor"]["g"] for result in valid_results) / valid_frame_count)
    avg_b = int(sum(result["averageSkinColor"]["b"] for result in valid_results) / valid_frame_count)
    avg_contrast_score = int(sum(result["contrastScore"] for result in valid_results) / valid_frame_count)

    return {
        "message": "영상 기반 퍼스널 컬러 분석 성공",
        "frameCount": frame_count,
        "validFrameCount": valid_frame_count,
        "season": final_season,
        "tone": selected_type["tone"],
        "toneName": selected_type["toneName"],
        "seasonName": selected_type["seasonName"],
        "description": selected_type["description"],
        "bestColors": selected_type["bestColors"],
        "worstColors": selected_type["worstColors"],
        "confidence": final_confidence,
        "seasonVotes": season_votes,
        "averageSkinColor": {
            "r": avg_r,
            "g": avg_g,
            "b": avg_b,
            "hex": "#{:02X}{:02X}{:02X}".format(avg_r, avg_g, avg_b),
            "lab": rgb_to_lab(avg_r, avg_g, avg_b),
            "hsv": rgb_to_hsv(avg_r, avg_g, avg_b),
        },
        "contrastScore": avg_contrast_score,
        "contrastLevel": get_contrast_level(avg_contrast_score),
        "analysisReason": (
            f"{valid_frame_count}개의 유효 프레임을 분석하고 confidence를 가중치로 season 투표를 진행했습니다. "
            f"가장 높은 누적 점수는 {final_season}입니다."
        ),
        "skippedFrameCount": len(skipped_frames),
        "skippedFrames": skipped_frames,
    }
