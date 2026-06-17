from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import csv
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

FACE_LANDMARKER_MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
SAMPLES_CSV_PATH = Path(__file__).with_name("samples.csv")
_face_landmarker = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://tone-z.mirim-it-show.site",
        "http://tone-z.mirim-it-show.site",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(contents: bytes):
    np_array = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)


def get_face_landmarks(rgb_image):
    global _face_landmarker

    if not FACE_LANDMARKER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"FaceLandmarker model not found: {FACE_LANDMARKER_MODEL_PATH}"
        )

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


DEFAULT_PERSONAL_COLOR_PROFILES = {
    # value: 피부 HSV V, saturation: 피부 HSV S, lightness: 피부 LAB L
    # chroma: 피부 LAB a/b가 중립점 128에서 떨어진 정도
    # contrast: 피부와 눈/머리의 밝기 차이, hair_value/eye_value: 눈과 머리의 어두운 정도
    # 한국인 기준으로 보정: 한국인은 머리/눈이 전반적으로 어두워서 서양 기준보다 hair_value/eye_value를 낮췄습니다.
    # autumn-deep만 유독 낮게 설정해 전형적인 흑발/흑안 한국인이 autumn-deep으로 과분류되는 것을 방지합니다.
    "spring-light": {"tone": "warm", "value": 218, "saturation": 38, "lightness": 184, "chroma": 16, "warmth": 14, "contrast": 24, "hair_value": 95, "eye_value": 78},
    "spring-bright": {"tone": "warm", "value": 218, "saturation": 62, "lightness": 178, "chroma": 28, "warmth": 18, "contrast": 42, "hair_value": 70, "eye_value": 55},
    "spring-soft": {"tone": "warm", "value": 198, "saturation": 34, "lightness": 166, "chroma": 14, "warmth": 12, "contrast": 22, "hair_value": 88, "eye_value": 70},
    "autumn-mute": {"tone": "warm", "value": 176, "saturation": 34, "lightness": 150, "chroma": 15, "warmth": 15, "contrast": 30, "hair_value": 72, "eye_value": 58},
    "autumn-deep": {"tone": "warm", "value": 145, "saturation": 48, "lightness": 126, "chroma": 24, "warmth": 20, "contrast": 58, "hair_value": 40, "eye_value": 32},
    "summer-light": {"tone": "cool", "value": 212, "saturation": 28, "lightness": 182, "chroma": 10, "warmth": 5, "contrast": 22, "hair_value": 92, "eye_value": 75},
    "summer-mute": {"tone": "cool", "value": 186, "saturation": 24, "lightness": 160, "chroma": 10, "warmth": 2, "contrast": 24, "hair_value": 78, "eye_value": 62},
    "winter-bright": {"tone": "cool", "value": 210, "saturation": 56, "lightness": 166, "chroma": 26, "warmth": -8, "contrast": 68, "hair_value": 32, "eye_value": 25},
    "winter-deep": {"tone": "cool", "value": 142, "saturation": 48, "lightness": 118, "chroma": 24, "warmth": -10, "contrast": 78, "hair_value": 22, "eye_value": 18},
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
    r = int(r)
    g = int(g)
    b = int(b)

    return {
        "r": r,
        "g": g,
        "b": b,
        "hex": "#{:02X}{:02X}{:02X}".format(r, g, b),
        "lab": rgb_to_lab(r, g, b),
        "hsv": rgb_to_hsv(r, g, b),
    }


def mean_rgb_from_pixels(pixels):
    if pixels is None or len(pixels) == 0:
        return None

    avg_bgr = np.mean(pixels, axis=0)
    b, g, r = avg_bgr
    return make_color(r, g, b)


def median_rgb_from_pixels(pixels):
    if pixels is None or len(pixels) == 0:
        return None

    median_bgr = np.median(pixels, axis=0)
    b, g, r = median_bgr
    return make_color(r, g, b)


def representative_skin_color_from_pixels(pixels):
    if pixels is None or len(pixels) == 0:
        return None

    hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    value = hsv_pixels[:, 2]
    saturation = hsv_pixels[:, 1]

    lower_v = np.percentile(value, 15)
    upper_v = np.percentile(value, 85)
    lower_s = np.percentile(saturation, 5)
    upper_s = np.percentile(saturation, 95)
    mask = (
        (value >= lower_v)
        & (value <= upper_v)
        & (saturation >= lower_s)
        & (saturation <= upper_s)
        & ~((value >= 215) & (saturation <= 35))
    )
    filtered = pixels[mask]

    if len(filtered) < 12:
        filtered = pixels

    return median_rgb_from_pixels(filtered)


def average_color(colors):
    valid_colors = [color for color in colors if color]

    if not valid_colors:
        return None

    avg_r = int(sum(color["r"] for color in valid_colors) / len(valid_colors))
    avg_g = int(sum(color["g"] for color in valid_colors) / len(valid_colors))
    avg_b = int(sum(color["b"] for color in valid_colors) / len(valid_colors))
    return make_color(avg_r, avg_g, avg_b)


def average_debug_metrics(results):
    numeric_keys = [
        "brightness",
        "saturation",
        "lightness",
        "chroma",
        "warmthScore",
        "contrastScore",
        "weightedLabB",
        "weightedSaturation",
        "cheekSaturation",
        "cheekLabB",
        "hairValue",
        "eyeValue",
    ]
    averaged = {}
    metrics_list = [result.get("debugMetrics", {}) for result in results]

    for key in numeric_keys:
        values = [metrics[key] for metrics in metrics_list if key in metrics and isinstance(metrics[key], (int, float))]
        if values:
            averaged[key] = round(sum(values) / len(values), 2)

    tone_counts = {}
    season_group_counts = {}
    contrast_level_counts = {}

    for metrics in metrics_list:
        tone = metrics.get("tone")
        season_group = metrics.get("seasonGroup")
        contrast_level = metrics.get("contrastLevel")

        if tone:
            tone_counts[tone] = tone_counts.get(tone, 0) + 1
        if season_group:
            season_group_counts[season_group] = season_group_counts.get(season_group, 0) + 1
        if contrast_level:
            contrast_level_counts[contrast_level] = contrast_level_counts.get(contrast_level, 0) + 1

    if tone_counts:
        averaged["tone"] = max(tone_counts, key=tone_counts.get)
    if season_group_counts:
        averaged["seasonGroup"] = max(season_group_counts, key=season_group_counts.get)
    if contrast_level_counts:
        averaged["contrastLevel"] = max(contrast_level_counts, key=contrast_level_counts.get)

    return averaged


AGGREGATE_METRIC_FIELDS = [
    "brightness",
    "saturation",
    "lightness",
    "warmthScore",
    "weightedLabB",
    "weightedSaturation",
    "cheekLabB",
    "contrastScore",
    "featureDarkness",
    "skinFeatureGap",
]


STABLE_MEDIAN_FIELDS = [
    "brightness",
    "saturation",
    "lightness",
    "warmthScore",
    "weightedLabB",
    "weightedSaturation",
    "cheekLabB",
    "contrastScore",
    "featureDarkness",
    "skinFeatureGap",
]


def most_common_value(values, default=None):
    counts = {}

    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1

    if not counts:
        return default

    return max(counts, key=counts.get)


def remove_metric_outliers(results):
    if len(results) < 4:
        return results

    bounds = {}

    for field in AGGREGATE_METRIC_FIELDS:
        values = [
            float(result["debugMetrics"][field])
            for result in results
            if field in result.get("debugMetrics", {})
        ]

        if len(values) < 4:
            continue

        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1

        if iqr == 0:
            continue

        bounds[field] = (q1 - (iqr * 1.5), q3 + (iqr * 1.5))

    if not bounds:
        return results

    filtered_results = []

    for result in results:
        metrics = result.get("debugMetrics", {})
        is_outlier = False

        for field, (lower_bound, upper_bound) in bounds.items():
            if field not in metrics:
                continue

            value = float(metrics[field])
            if value < lower_bound or value > upper_bound:
                is_outlier = True
                break

        if not is_outlier:
            filtered_results.append(result)

    return filtered_results if len(filtered_results) >= 2 else results


def median_debug_metrics(results):
    metrics_list = [result.get("debugMetrics", {}) for result in results]
    averaged = {}

    for field in CSV_METRIC_FIELDS:
        values = [
            float(metrics[field])
            for metrics in metrics_list
            if field in metrics and isinstance(metrics[field], (int, float))
        ]

        if values:
            averaged[field] = round(float(np.median(values)), 2)

    averaged["tone"] = most_common_value([metrics.get("tone") for metrics in metrics_list], "neutral")
    averaged["seasonGroup"] = most_common_value([metrics.get("seasonGroup") for metrics in metrics_list])
    averaged["contrastLevel"] = get_contrast_level(int(averaged.get("contrastScore", 0)))

    return averaged


def stable_median_metrics(results):
    metrics_list = [result.get("debugMetrics", {}) for result in results]
    medians = {}

    for field in STABLE_MEDIAN_FIELDS:
        values = [
            float(metrics[field])
            for metrics in metrics_list
            if field in metrics and isinstance(metrics[field], (int, float))
        ]

        if values:
            medians[field] = round(float(np.median(values)), 2)

    medians["contrastLevel"] = get_contrast_level(int(medians.get("contrastScore", 0)))
    return medians


def select_video_frame_files(files, max_frames=60):
    if len(files) <= max_frames:
        return list(enumerate(files))

    indices = np.linspace(0, len(files) - 1, max_frames, dtype=int)
    return [(int(index), files[int(index)]) for index in indices]


def summarize_skipped_frames(skipped_frames):
    reason_counts = {}

    for frame in skipped_frames:
        reason = frame.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return reason_counts


CSV_METRIC_FIELDS = [
    "brightness",
    "saturation",
    "lightness",
    "chroma",
    "warmthScore",
    "contrastScore",
    "weightedLabB",
    "weightedSaturation",
    "cheekSaturation",
    "cheekLabB",
    "hairValue",
    "eyeValue",
]


def build_csv_metrics(debug_metrics):
    return {
        field: round(float(debug_metrics.get(field, 0)), 4)
        for field in CSV_METRIC_FIELDS
    }


def append_sample_to_csv(label, debug_metrics):
    row = {"label": label, **build_csv_metrics(debug_metrics)}
    file_exists = SAMPLES_CSV_PATH.exists()

    with SAMPLES_CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["label", *CSV_METRIC_FIELDS])

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def calculate_profiles_from_samples():
    if not SAMPLES_CSV_PATH.exists():
        return {}

    sums = {}
    field_counts = {}

    with SAMPLES_CSV_PATH.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            label = row.get("label")

            if label not in PERSONAL_COLOR_TYPES:
                continue

            sums.setdefault(label, {field: 0.0 for field in CSV_METRIC_FIELDS})
            field_counts.setdefault(label, {field: 0 for field in CSV_METRIC_FIELDS})

            for field in CSV_METRIC_FIELDS:
                raw_value = row.get(field)

                if raw_value in [None, ""]:
                    continue

                try:
                    sums[label][field] += float(raw_value)
                    field_counts[label][field] += 1
                except ValueError:
                    pass

    profiles = {}

    for label, totals in sums.items():
        profiles[label] = {
            field: totals[field] / field_counts[label][field]
            for field in CSV_METRIC_FIELDS
            if field_counts[label][field] > 0
        }
        profiles[label]["tone"] = PERSONAL_COLOR_TYPES[label]["tone"]

    return profiles


def get_active_color_profiles():
    default_profiles = {
        season: {
            "brightness": profile["value"],
            "saturation": profile["saturation"],
            "lightness": profile["lightness"],
            "chroma": profile["chroma"],
            "warmthScore": profile["warmth"],
            "contrastScore": profile["contrast"],
            "weightedLabB": 133.5 + (profile["warmth"] / 3.2),
            "weightedSaturation": profile["saturation"],
            "cheekSaturation": profile["saturation"],
            "cheekLabB": 133.5 + (profile["warmth"] / 3.2),
            "hairValue": profile["hair_value"],
            "eyeValue": profile["eye_value"],
            "tone": profile["tone"],
        }
        for season, profile in DEFAULT_PERSONAL_COLOR_PROFILES.items()
    }

    return default_profiles


def profile_distance(metrics, profile):
    # 모든 분류를 같은 metric vector 거리 비교로 처리합니다.
    # brightness/saturation/contrast가 과하게 지배하지 않도록 낮추고,
    # 한국인 피부에서 중요한 LAB b 기반 warm/cool 축과 눈/눈썹 특징을 더 크게 반영합니다.
    weights = {
        "brightness": 0.10,
        "saturation": 0.08,
        "lightness": 0.12,
        "chroma": 0.08,
        "warmthScore": 0.24,
        "contrastScore": 0.05,
        "weightedLabB": 0.30,
        "weightedSaturation": 0.10,
        "cheekSaturation": 0.10,
        "cheekLabB": 0.22,
        "hairValue": 0.02,
        "eyeValue": 0.03,
    }

    score = 0.0

    for field, weight in weights.items():
        if field not in profile:
            continue

        score += abs(float(metrics.get(field, 0)) - float(profile[field])) * weight

    metric_tone = metrics.get("tone")
    profile_tone = profile.get("tone")
    weighted_lab_b = float(metrics.get("weightedLabB", 133.5))

    if metric_tone == "warm" and profile_tone == "cool":
        score += 12
    elif metric_tone == "cool" and profile_tone == "warm":
        score += 10
    elif metric_tone == "neutral":
        if weighted_lab_b >= 136.5 and profile_tone == "cool":
            score += 2
        elif weighted_lab_b < 133.5 and profile_tone == "warm":
            score += 2

    return round(score, 2)


def filter_pixels_for_region(pixels, region="default"):
    if pixels is None or len(pixels) == 0:
        return pixels

    hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    saturation = hsv_pixels[:, 1]
    value = hsv_pixels[:, 2]

    if region == "skin":
        ycrcb_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        cr = ycrcb_pixels[:, 1]
        cb = ycrcb_pixels[:, 2]
        mask = (
            (value >= 55)
            & (value <= 240)
            & (saturation <= 155)
            & (cr >= 125)
            & (cr <= 185)
            & (cb >= 70)
            & (cb <= 150)
            & ~((value >= 215) & (saturation <= 35))
        )
    elif region == "iris":
        mask = ~(((value >= 185) & (saturation <= 70)) | np.all(pixels >= 175, axis=1))
    elif region == "eyebrow":
        threshold = min(150, int(np.percentile(value, 45)) + 20)
        mask = (value <= threshold) & ~((value >= 175) & (saturation <= 45))
    else:
        mask = (value >= 10) & (value <= 250)

    filtered = pixels[mask]
    return filtered if len(filtered) > 0 else pixels


def get_dominant_color_by_kmeans(image, points=None, roi=None, k=3, region="default"):
    h, w, _ = image.shape

    if points is not None:
        points = np.array(points, dtype=np.int32)
        x, y, box_w, box_h = cv2.boundingRect(points)
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + box_w, w)
        y2 = min(y + box_h, h)
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            return None

        shifted_points = points - np.array([x1, y1])
        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, shifted_points, 255)
        pixels = cropped[mask == 255]
    elif roi is not None:
        x1, y1, x2, y2 = roi
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w)
        y2 = min(int(y2), h)
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            return None

        pixels = cropped.reshape(-1, 3)
    else:
        pixels = image.reshape(-1, 3)

    pixels = filter_pixels_for_region(pixels, region)

    if len(pixels) == 0:
        return None

    if region == "skin":
        return representative_skin_color_from_pixels(pixels)

    if len(pixels) < k:
        return mean_rgb_from_pixels(pixels)

    samples = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)

    # 피부/눈썹/홍채 모두 조명 반사나 배경이 한 클러스터를 차지할 수 있어서
    # 가장 큰 클러스터 중 극단적으로 밝은 후보를 피하고 대표성이 높은 클러스터를 선택합니다.
    ranked_indices = np.argsort(counts)[::-1]
    selected_center = centers[ranked_indices[0]]

    for index in ranked_indices:
        center = centers[index]
        center_hsv = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_BGR2HSV)[0][0]
        if region == "skin" and center_hsv[2] >= 245 and center_hsv[1] <= 35:
            continue
        if region in ["iris", "eyebrow"] and center_hsv[2] >= 190 and center_hsv[1] <= 50:
            continue
        selected_center = center
        break

    b, g, r = selected_center
    return make_color(r, g, b)


def analyze_tone(lab, r: int, g: int, b: int):
    lab_a = lab["a"]
    lab_b = lab["b"]

    # 한국인 피부는 LAB b가 133~138 근처에 많이 몰립니다.
    # 그래서 128 중립점에서 멀리 떨어진 값만 warm으로 보던 기준을 완화하고,
    # LAB b와 b-a 차이를 중심으로 하되 RGB 편향은 약하게만 사용합니다.
    yellow_bias = lab_b - 128
    yellow_over_red = lab_b - lab_a
    rgb_warm_bias = (r - b) * 0.04
    warmth_score = (
        (lab_b - 133.5) * 2.2
        + yellow_bias * 0.25
        + yellow_over_red * 0.55
        + rgb_warm_bias
    )

    if lab_b >= 136 or warmth_score >= 4:
        return "warm", "웜톤", warmth_score
    if lab_b <= 132 and warmth_score <= -3:
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


def normalize_weights(weighted_colors):
    valid_items = [
        (name, color, weight)
        for name, color, weight in weighted_colors
        if color and weight > 0
    ]
    total_weight = sum(weight for _, _, weight in valid_items)

    if total_weight <= 0:
        return []

    return [
        (name, color, weight / total_weight)
        for name, color, weight in valid_items
    ]


def weighted_color_from_regions(weighted_colors):
    normalized = normalize_weights(weighted_colors)

    if not normalized:
        return None, {}

    avg_r = sum(color["r"] * weight for _, color, weight in normalized)
    avg_g = sum(color["g"] * weight for _, color, weight in normalized)
    avg_b = sum(color["b"] * weight for _, color, weight in normalized)
    weights = {
        name: round(weight, 4)
        for name, _, weight in normalized
    }

    return make_color(avg_r, avg_g, avg_b), weights


def calculate_adaptive_skin_color(region_colors):
    left_cheek = region_colors.get("leftCheekColor")
    right_cheek = region_colors.get("rightCheekColor")
    forehead = region_colors.get("foreheadColor")
    chin = region_colors.get("chinColor")
    nose = region_colors.get("noseColor")

    default_weights = {
        "leftCheekColor": 0.25,
        "rightCheekColor": 0.25,
        "foreheadColor": 0.20,
        "chinColor": 0.15,
        "noseColor": 0.15,
    }
    base_regions = [color for color in [forehead, nose, chin] if color]
    cheek_regions = [color for color in [left_cheek, right_cheek] if color]
    base_saturations = [color["hsv"]["s"] for color in base_regions]
    cheek_saturations = [color["hsv"]["s"] for color in cheek_regions]
    base_skin_saturation = (
        float(np.median(base_saturations))
        if base_saturations
        else float(np.median(cheek_saturations)) if cheek_saturations else 0.0
    )
    cheek_saturation = (
        sum(cheek_saturations) / len(cheek_saturations)
        if cheek_saturations
        else base_skin_saturation
    )
    weights = default_weights.copy()

    if cheek_saturation > base_skin_saturation + 25:
        original_cheek_total = weights["leftCheekColor"] + weights["rightCheekColor"]
        weights["leftCheekColor"] *= 0.58
        weights["rightCheekColor"] *= 0.58
        released_weight = original_cheek_total - weights["leftCheekColor"] - weights["rightCheekColor"]
        base_weight_total = weights["foreheadColor"] + weights["chinColor"] + weights["noseColor"]

        if base_weight_total > 0:
            for key in ["foreheadColor", "chinColor", "noseColor"]:
                weights[key] += released_weight * (weights[key] / base_weight_total)

    skin_color, final_weights = weighted_color_from_regions([
        ("leftCheekColor", left_cheek, weights["leftCheekColor"]),
        ("rightCheekColor", right_cheek, weights["rightCheekColor"]),
        ("foreheadColor", forehead, weights["foreheadColor"]),
        ("chinColor", chin, weights["chinColor"]),
        ("noseColor", nose, weights["noseColor"]),
    ])

    return skin_color, {
        "baseSkinSaturation": round(base_skin_saturation, 2),
        "adjustedCheekWeight": round(
            final_weights.get("leftCheekColor", 0) + final_weights.get("rightCheekColor", 0),
            4,
        ),
        "finalSkinWeights": final_weights,
    }


def calculate_feature_contrast(skin_color, eye_color=None, hair_color=None):
    brightness = skin_color["hsv"]["v"]
    eye_value = eye_color["hsv"]["v"] if eye_color else brightness
    hair_value = hair_color["hsv"]["v"] if hair_color else brightness
    feature_darkness = min(eye_value, hair_value)
    skin_feature_gap = max(0, brightness - feature_darkness)

    return {
        "eyeValue": eye_value,
        "hairValue": hair_value,
        "featureDarkness": feature_darkness,
        "skinFeatureGap": skin_feature_gap,
    }


def get_weighted_region_analysis(skin_color, eye_color=None, eyebrow_color=None, dominant_colors=None, contrast_level="low"):
    dominant_colors = dominant_colors or {}
    left_cheek = dominant_colors.get("leftCheekColor") or skin_color
    right_cheek = dominant_colors.get("rightCheekColor") or skin_color
    forehead = dominant_colors.get("foreheadColor") or skin_color
    chin = dominant_colors.get("chinColor") or skin_color
    nose = dominant_colors.get("noseColor") or skin_color
    region_colors = {
        "leftCheekColor": left_cheek,
        "rightCheekColor": right_cheek,
        "foreheadColor": forehead,
        "chinColor": chin,
        "noseColor": nose,
    }
    adaptive_skin_color, adaptive_info = calculate_adaptive_skin_color(region_colors)
    weighted_skin_color = adaptive_skin_color or skin_color

    weighted_lab_b = weighted_skin_color["lab"]["b"]
    weighted_saturation = weighted_skin_color["hsv"]["s"]
    weighted_lightness = weighted_skin_color["lab"]["l"]
    cheek_saturation = (left_cheek["hsv"]["s"] + right_cheek["hsv"]["s"]) / 2
    cheek_lab_b = (left_cheek["lab"]["b"] + right_cheek["lab"]["b"]) / 2
    forehead_saturation = forehead["hsv"]["s"]
    nose_saturation = nose["hsv"]["s"]
    chin_saturation = chin["hsv"]["s"]

    warmth_score = (
        (weighted_lab_b - 133.5) * 2.2
        + (cheek_lab_b - 135) * 1.0
        + (cheek_saturation - 35) * 0.05
    )

    if weighted_lab_b >= 138.5 and cheek_lab_b >= 138.0:
        tone = "warm"
        tone_name = "웜톤"
    elif weighted_lab_b <= 133.0 and cheek_lab_b <= 134.5:
        tone = "cool"
        tone_name = "쿨톤"
    else:
        tone = "neutral"
        tone_name = "뉴트럴"

    if tone == "warm":
        season_group = "spring" if weighted_lightness >= 160 and weighted_saturation >= 30 else "autumn"
    elif tone == "cool":
        season_group = "winter" if weighted_saturation >= 50 and contrast_level in ["medium", "high"] else "summer"
    else:
        if weighted_lab_b >= 134.8 and weighted_lightness >= 160:
            season_group = "spring"
        elif contrast_level == "high" and weighted_saturation >= 48:
            season_group = "winter"
        elif weighted_saturation <= 42:
            season_group = "summer"
        else:
            season_group = "spring" if weighted_lab_b >= 134 else "summer"

    return {
        "tone": tone,
        "toneName": tone_name,
        "warmthScore": warmth_score,
        "weightedLabB": round(weighted_lab_b, 2),
        "weightedSaturation": round(weighted_saturation, 2),
        "weightedLightness": round(weighted_lightness, 2),
        "cheekSaturation": round(cheek_saturation, 2),
        "cheekLabB": round(cheek_lab_b, 2),
        "foreheadSaturation": round(forehead_saturation, 2),
        "noseSaturation": round(nose_saturation, 2),
        "chinSaturation": round(chin_saturation, 2),
        "adjustedCheekWeight": adaptive_info["adjustedCheekWeight"],
        "finalSkinWeights": adaptive_info["finalSkinWeights"],
        "adaptiveSkinColor": weighted_skin_color,
        "seasonGroup": season_group,
    }


def build_analysis_reason(result, skin_hsv, eye_color, hair_color, contrast_score, contrast_level):
    season = result["season"]
    confidence = result.get("confidence", 0)
    metrics = result.get("colorMetrics", {})
    weighted_lab_b = metrics.get("weightedLabB")
    weighted_saturation = metrics.get("weightedSaturation")
    season_group = metrics.get("seasonGroup")
    metric_summary = ""

    if weighted_lab_b is not None and weighted_saturation is not None:
        metric_summary = (
            f" k-means로 추출한 뺨/눈동자/눈썹 대표색의 LAB b 가중값은 {weighted_lab_b}, "
            f"HSV S 가중값은 {weighted_saturation}이며 2차 후보는 {season_group} 계열입니다."
        )

    if season.startswith("winter") and contrast_level == "high":
        if skin_hsv["v"] < 185:
            return f"피부 밝기가 낮고 눈동자/머리카락과의 대비가 강합니다. contrastScore가 {contrast_score}로 높아 겨울 쿨 딥 가능성을 높게 판단했습니다.{metric_summary}"
        return f"피부는 밝은 편이고 눈동자/머리카락이 어두워 대비가 강합니다. contrastScore가 {contrast_score}로 높아 겨울 쿨 계열을 우선 판단했습니다.{metric_summary}"
    if season.startswith("summer") and contrast_level == "low":
        return f"피부 LAB의 노란기가 강하지 않고 채도가 낮으며 대비가 낮습니다. 봄 웜보다 여름 쿨의 부드러운 후보가 더 가까워 {season}로 판단했습니다.{metric_summary}"
    if season == "spring-soft":
        return f"피부 LAB에서 따뜻한 노란기가 확인되지만 대비와 채도가 높지 않아 봄 웜 소프트로 판단했습니다. confidence는 {confidence}입니다.{metric_summary}"
    if season.startswith("spring"):
        return f"대표색의 LAB b값이 높고 명도가 밝아 웜 계열 점수가 높았습니다. 대비와 채도를 함께 비교해 {season}로 판단했습니다.{metric_summary}"
    if season == "autumn-mute":
        return f"대표색의 따뜻한 기운은 있으나 채도와 밝기가 차분하고 대비가 강하지 않아 가을 웜 뮤트로 판단했습니다.{metric_summary}"
    if season == "autumn-deep":
        return f"피부 밝기가 낮고 전체 대비가 깊은 편이라 가을 웜 딥으로 판단했습니다.{metric_summary}"
    if result["tone"] == "cool":
        return f"대표색의 LAB b값이 강하지 않고 대비/채도 조건을 함께 비교한 결과 쿨 계열인 {season}가 가장 가까웠습니다.{metric_summary}"

    return f"뺨, 눈동자, 눈썹 대표색과 대비감이 경계에 가까워 가장 가까운 퍼스널 컬러 타입으로 판단했습니다.{metric_summary}"


def build_debug_metrics_from_colors(avg_r, avg_g, avg_b, eye_color=None, hair_color=None, eyebrow_color=None, dominant_colors=None):
    lab = rgb_to_lab(avg_r, avg_g, avg_b)
    hsv = rgb_to_hsv(avg_r, avg_g, avg_b)
    skin_color = make_color(avg_r, avg_g, avg_b)

    brightness = hsv["v"]
    saturation = hsv["s"]
    lightness = lab["l"]
    chroma = abs(lab["a"] - 128) + abs(lab["b"] - 128)
    contrast_score, contrast_level = calculate_contrast(skin_color, eye_color, hair_color)
    region_analysis = get_weighted_region_analysis(
        skin_color,
        eye_color=eye_color,
        eyebrow_color=eyebrow_color,
        dominant_colors=dominant_colors,
        contrast_level=contrast_level,
    )
    tone = region_analysis["tone"]
    tone_name = region_analysis["toneName"]
    warmth_score = region_analysis["warmthScore"]
    season_group = region_analysis["seasonGroup"]
    feature_contrast = calculate_feature_contrast(skin_color, eye_color, hair_color)
    hair_value = feature_contrast["hairValue"]
    eye_value = feature_contrast["eyeValue"]

    debug_metrics = {
        "brightness": brightness,
        "saturation": saturation,
        "lightness": lightness,
        "chroma": chroma,
        "warmthScore": round(float(warmth_score), 2),
        "tone": tone,
        "toneName": tone_name,
        "seasonGroup": season_group,
        "contrastScore": contrast_score,
        "contrastLevel": contrast_level,
        "weightedLabB": region_analysis["weightedLabB"],
        "weightedSaturation": region_analysis["weightedSaturation"],
        "cheekSaturation": region_analysis["cheekSaturation"],
        "cheekLabB": region_analysis["cheekLabB"],
        "foreheadSaturation": region_analysis["foreheadSaturation"],
        "noseSaturation": region_analysis["noseSaturation"],
        "chinSaturation": region_analysis["chinSaturation"],
        "hairValue": hair_value,
        "eyeValue": eye_value,
        "featureDarkness": feature_contrast["featureDarkness"],
        "skinFeatureGap": feature_contrast["skinFeatureGap"],
        "adjustedCheekWeight": region_analysis["adjustedCheekWeight"],
        "finalSkinWeights": region_analysis["finalSkinWeights"],
    }
    debug_metrics["stableSeason"] = stable_classify_from_metrics(debug_metrics)["season"]
    debug_metrics["voteSeason"] = None
    debug_metrics["voteRatio"] = 0.0

    return debug_metrics, region_analysis, hsv, contrast_score, contrast_level


def classify_personal_color_from_metrics(debug_metrics, skin_hsv=None, eye_color=None, hair_color=None, include_debug=False):
    profiles = get_active_color_profiles()
    ranked_scores = sorted(
        [
            (profile_distance(debug_metrics, profile), season)
            for season, profile in profiles.items()
        ]
    )

    best_score, best_season = ranked_scores[0]
    second_score = ranked_scores[1][0] if len(ranked_scores) > 1 else best_score + 20
    score_gap = max(0, second_score - best_score)
    confidence = 0.55 + min(0.35, score_gap / 45)
    tone = debug_metrics.get("tone")

    if tone == "neutral":
        confidence -= 0.08

    confidence = round(max(0.35, min(0.95, confidence)), 2)
    top_candidates = [
        {"season": season, "score": score}
        for score, season in ranked_scores[:3]
    ]

    result = PERSONAL_COLOR_TYPES[best_season].copy()
    result.update({
        "tone": result["tone"],
        "toneName": result["toneName"],
        "season": best_season,
        "confidence": confidence,
    })

    if include_debug:
        result.update({
            "contrastScore": debug_metrics.get("contrastScore"),
            "contrastLevel": debug_metrics.get("contrastLevel"),
            "colorMetrics": {
                "tone": debug_metrics.get("tone"),
                "toneName": debug_metrics.get("toneName"),
                "warmthScore": debug_metrics.get("warmthScore"),
                "weightedLabB": debug_metrics.get("weightedLabB"),
                "weightedSaturation": debug_metrics.get("weightedSaturation"),
                "cheekSaturation": debug_metrics.get("cheekSaturation"),
                "cheekLabB": debug_metrics.get("cheekLabB"),
                "foreheadSaturation": debug_metrics.get("foreheadSaturation"),
                "noseSaturation": debug_metrics.get("noseSaturation"),
                "chinSaturation": debug_metrics.get("chinSaturation"),
                "eyeValue": debug_metrics.get("eyeValue"),
                "hairValue": debug_metrics.get("hairValue"),
                "featureDarkness": debug_metrics.get("featureDarkness"),
                "skinFeatureGap": debug_metrics.get("skinFeatureGap"),
                "adjustedCheekWeight": debug_metrics.get("adjustedCheekWeight"),
                "finalSkinWeights": debug_metrics.get("finalSkinWeights"),
                "seasonGroup": debug_metrics.get("seasonGroup"),
            },
            "topCandidates": top_candidates,
            "debugMetrics": debug_metrics,
            "csvMetrics": build_csv_metrics(debug_metrics),
        })

        if skin_hsv:
            result["analysisReason"] = build_analysis_reason(
                result,
                skin_hsv,
                eye_color,
                hair_color,
                debug_metrics.get("contrastScore", 0),
                debug_metrics.get("contrastLevel", "low"),
            )

    return result


def build_result_for_season(season, confidence):
    result = PERSONAL_COLOR_TYPES[season].copy()
    result.update({
        "season": season,
        "tone": result["tone"],
        "toneName": result["toneName"],
        "confidence": round(max(0.55, min(0.9, confidence)), 2),
    })
    return result


def stable_classify_from_metrics(metrics):
    brightness = float(metrics.get("brightness", 0))
    saturation = float(metrics.get("saturation", 0))
    lightness = float(metrics.get("lightness", 0))
    weighted_lab_b = float(metrics.get("weightedLabB", 133.5))
    weighted_saturation = float(metrics.get("weightedSaturation", saturation))
    cheek_lab_b = float(metrics.get("cheekLabB", weighted_lab_b))
    contrast_score = float(metrics.get("contrastScore", 0))
    eye_value = float(metrics.get("eyeValue", brightness))
    hair_value = float(metrics.get("hairValue", brightness))
    feature_darkness = float(metrics.get("featureDarkness", min(eye_value, hair_value)))
    skin_feature_gap = float(metrics.get("skinFeatureGap", max(0, brightness - feature_darkness)))

    if weighted_lab_b >= 138.5 and cheek_lab_b >= 138.0:
        tone = "warm"
    elif weighted_lab_b <= 133.0 and cheek_lab_b <= 134.5:
        tone = "cool"
    else:
        tone = "neutral"

    is_deep = (
        brightness < 160
        and lightness < 145
        and feature_darkness <= 70
        and skin_feature_gap >= 75
    )
    is_bright = weighted_saturation >= 65 and skin_feature_gap >= 70
    is_light = brightness >= 205 and lightness >= 180 and skin_feature_gap < 90
    is_low_contrast = skin_feature_gap < 55
    is_mid_contrast = skin_feature_gap < 90
    is_muted = weighted_saturation < 42
    is_high_contrast = skin_feature_gap >= 100
    winter_feature = feature_darkness <= 55 and is_high_contrast

    if tone == "neutral":
        if is_light:
            season = "spring-soft" if weighted_lab_b >= 135 else "summer-light"
        elif is_deep:
            season = "winter-deep" if winter_feature and weighted_lab_b < 133 else "autumn-mute"
        elif brightness < 170 and lightness < 150 and weighted_lab_b >= 135:
            season = "autumn-mute"
        elif brightness >= 190 and is_mid_contrast:
            season = "spring-soft" if weighted_lab_b >= 135.5 else "summer-mute"
        else:
            season = "summer-mute"
    elif tone == "warm":
        if is_deep:
            season = "autumn-deep"
        elif is_bright and is_high_contrast:
            season = "spring-bright"
        elif is_light or (brightness >= 198 and lightness >= 172 and is_low_contrast):
            season = "spring-light"
        elif brightness < 178 or lightness < 155 or is_muted:
            season = "autumn-mute"
        else:
            season = "spring-soft"
    else:
        if is_deep and winter_feature:
            season = "winter-deep"
        elif is_bright and (winter_feature or is_high_contrast):
            season = "winter-bright"
        elif is_light:
            season = "summer-light"
        else:
            season = "summer-mute"

    confidence = 0.78

    if tone != "neutral":
        confidence += 0.04
    if is_light or is_bright or is_deep:
        confidence += 0.03
    if is_low_contrast and season.endswith(("bright", "deep")):
        confidence -= 0.08
    if 132.5 < weighted_lab_b < 137.5:
        confidence -= 0.04

    return build_result_for_season(season, confidence)


def majority_vote_season(results, tie_metrics):
    season_counts = {}

    for result in results:
        metrics = result.get("debugMetrics", {})
        season = stable_classify_from_metrics(metrics)["season"]
        season_counts[season] = season_counts.get(season, 0) + 1

    if not season_counts:
        return stable_classify_from_metrics(tie_metrics)["season"], 0

    max_count = max(season_counts.values())
    tied_seasons = [
        season
        for season, count in season_counts.items()
        if count == max_count
    ]

    if len(tied_seasons) == 1:
        return tied_seasons[0], max_count

    profiles = get_active_color_profiles()
    selected_season = min(
        tied_seasons,
        key=lambda season: profile_distance(tie_metrics, profiles[season]),
    )

    return selected_season, max_count


def choose_final_video_result(filtered_results):
    final_metrics = stable_median_metrics(filtered_results)
    stable_result = stable_classify_from_metrics(final_metrics)
    tie_metrics = median_debug_metrics(filtered_results)
    vote_season, vote_count = majority_vote_season(filtered_results, tie_metrics)
    vote_ratio = vote_count / max(1, len(filtered_results))

    if vote_season == stable_result["season"] or vote_ratio >= 0.65:
        final_season = vote_season
    else:
        final_season = stable_result["season"]

    final_metrics["stableSeason"] = stable_result["season"]
    final_metrics["voteSeason"] = vote_season
    final_metrics["voteRatio"] = round(vote_ratio, 4)
    final_confidence = max(stable_result["confidence"] - 0.04, 0.72) + min(0.14, vote_ratio * 0.14)
    final_result = build_result_for_season(final_season, final_confidence)

    return final_result, final_metrics


def public_personal_color_result(result):
    return {
        "season": result["season"],
        "tone": result["tone"],
        "toneName": result["toneName"],
        "seasonName": result["seasonName"],
        "description": result["description"],
        "bestColors": result["bestColors"],
        "worstColors": result["worstColors"],
        "confidence": result["confidence"],
    }


def analyze_personal_color(avg_r, avg_g, avg_b, eye_color=None, hair_color=None, eyebrow_color=None, dominant_colors=None, include_debug=False):
    debug_metrics, _, hsv, _, _ = build_debug_metrics_from_colors(
        avg_r,
        avg_g,
        avg_b,
        eye_color=eye_color,
        hair_color=hair_color,
        eyebrow_color=eyebrow_color,
        dominant_colors=dominant_colors,
    )
    result = classify_personal_color_from_metrics(
        debug_metrics,
        skin_hsv=hsv,
        eye_color=eye_color,
        hair_color=hair_color,
        include_debug=include_debug,
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

    return make_color(r, g, b)


def landmark_points(landmarks, indices, width, height):
    return [
        [int(landmarks[index].x * width), int(landmarks[index].y * height)]
        for index in indices
    ]


def square_roi(cx, cy, size, width, height):
    return (
        max(cx - size, 0),
        max(cy - size, 0),
        min(cx + size, width),
        min(cy + size, height),
    )


def scaled_roi_size(face_width, ratio, min_size=18, max_size=60):
    return max(min_size, min(max_size, int(face_width * ratio)))


def extract_iris_color(image, landmarks, indices, width, height):
    if len(landmarks) <= max(indices):
        return None

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

    if len(pixels) == 0:
        return None

    return get_dominant_color_by_kmeans(
        pixels.reshape(-1, 1, 3),
        roi=(0, 0, 1, len(pixels)),
        k=3,
        region="iris",
    )


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


def extract_eyebrow_color(image, landmarks, width, height):
    left_indices = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
    right_indices = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]

    left_color = get_dominant_color_by_kmeans(
        image,
        points=landmark_points(landmarks, left_indices, width, height),
        k=3,
        region="eyebrow",
    )
    right_color = get_dominant_color_by_kmeans(
        image,
        points=landmark_points(landmarks, right_indices, width, height),
        k=3,
        region="eyebrow",
    )
    colors = [color for color in [left_color, right_color] if color]

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

    if len(pixels) == 0:
        return None

    return get_dominant_color_by_kmeans(
        pixels.reshape(-1, 1, 3),
        roi=(0, 0, 1, len(pixels)),
        k=3,
        region="eyebrow",
    )


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
    if brightness > 245:
        return False, {
            "brightness": round(brightness, 2),
            "blurScore": round(blur_score, 2),
            "reason": "too_bright",
        }
    if blur_score < 10:
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


def extract_skin_regions(image, landmarks, width, height):
    left_cheek = landmarks[123]
    right_cheek = landmarks[352]
    forehead = landmarks[10]
    chin = landmarks[152]
    nose = landmarks[6]

    left_x = int(left_cheek.x * width)
    left_y = int(left_cheek.y * height)
    right_x = int(right_cheek.x * width)
    right_y = int(right_cheek.y * height)
    forehead_x = int(forehead.x * width)
    forehead_y = int(forehead.y * height)
    chin_x = int(chin.x * width)
    chin_y = int(chin.y * height)
    nose_x = int(nose.x * width)
    nose_y = int(nose.y * height)
    xs = [int(point.x * width) for point in landmarks]
    face_width = max(1, max(xs) - min(xs))
    cheek_roi_size = scaled_roi_size(face_width, 0.055, min_size=22, max_size=55)
    forehead_roi_size = scaled_roi_size(face_width, 0.045, min_size=18, max_size=45)
    chin_roi_size = scaled_roi_size(face_width, 0.045, min_size=18, max_size=45)
    nose_roi_size = scaled_roi_size(face_width, 0.035, min_size=16, max_size=36)

    left_color = get_dominant_color_by_kmeans(
        image,
        roi=square_roi(left_x, left_y, cheek_roi_size, width, height),
        k=3,
        region="skin",
    ) or get_avg_color(image, left_x, left_y, size=cheek_roi_size)
    right_color = get_dominant_color_by_kmeans(
        image,
        roi=square_roi(right_x, right_y, cheek_roi_size, width, height),
        k=3,
        region="skin",
    ) or get_avg_color(image, right_x, right_y, size=cheek_roi_size)
    forehead_color = get_dominant_color_by_kmeans(
        image,
        roi=square_roi(forehead_x, forehead_y, forehead_roi_size, width, height),
        k=3,
        region="skin",
    ) or get_avg_color(image, forehead_x, forehead_y, size=forehead_roi_size)
    chin_color = get_dominant_color_by_kmeans(
        image,
        roi=square_roi(chin_x, chin_y, chin_roi_size, width, height),
        k=3,
        region="skin",
    ) or get_avg_color(image, chin_x, chin_y, size=chin_roi_size)
    nose_color = get_dominant_color_by_kmeans(
        image,
        roi=square_roi(nose_x, nose_y, nose_roi_size, width, height),
        k=3,
        region="skin",
    ) or get_avg_color(image, nose_x, nose_y, size=nose_roi_size)

    region_colors = {
        "leftCheekColor": left_color,
        "rightCheekColor": right_color,
        "foreheadColor": forehead_color,
        "chinColor": chin_color,
        "noseColor": nose_color,
    }
    skin_color, _ = calculate_adaptive_skin_color(region_colors)

    return {
        "points": {
            "leftCheek": {"x": left_x, "y": left_y},
            "rightCheek": {"x": right_x, "y": right_y},
            "forehead": {"x": forehead_x, "y": forehead_y},
            "chin": {"x": chin_x, "y": chin_y},
            "nose": {"x": nose_x, "y": nose_y},
        },
        "colors": region_colors,
        "averageSkinColor": skin_color or average_color(region_colors.values()),
    }


def analyze_single_frame(image, include_debug=False, classify=True, fast_video=False):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = get_face_landmarks(rgb_image)

    if not landmarks:
        return {"error": "얼굴 랜드마크 검출 실패"}

    h, w, _ = image.shape
    skin_regions = extract_skin_regions(image, landmarks, w, h)
    points = skin_regions["points"]
    dominant_colors = skin_regions["colors"]
    average_skin_color = skin_regions["averageSkinColor"]

    left_color = dominant_colors["leftCheekColor"]
    right_color = dominant_colors["rightCheekColor"]
    forehead_color = dominant_colors["foreheadColor"]
    chin_color = dominant_colors["chinColor"]
    nose_color = dominant_colors["noseColor"]
    avg_r = average_skin_color["r"]
    avg_g = average_skin_color["g"]
    avg_b = average_skin_color["b"]
    avg_hex = average_skin_color["hex"]
    avg_lab = average_skin_color["lab"]
    avg_hsv = average_skin_color["hsv"]

    eye_color = None if fast_video else extract_eye_color(image, landmarks, w, h)
    eyebrow_color = None if fast_video else extract_eyebrow_color(image, landmarks, w, h)
    hair_color = extract_hair_color(image, landmarks, w, h)
    dominant_colors.update({
        "eyeColor": eye_color,
        "eyebrowColor": eyebrow_color,
    })
    debug_metrics, _, avg_hsv, contrast_score, contrast_level = build_debug_metrics_from_colors(
        avg_r,
        avg_g,
        avg_b,
        eye_color=eye_color,
        hair_color=hair_color,
        eyebrow_color=eyebrow_color,
        dominant_colors=dominant_colors,
    )
    personal_color = {}

    if classify:
        personal_color = classify_personal_color_from_metrics(
            debug_metrics,
            skin_hsv=avg_hsv,
            eye_color=eye_color,
            hair_color=hair_color,
            include_debug=include_debug,
        )

    frame_result = {
        "message": "피부색 추출 성공",
        "landmarkCount": len(landmarks),
        "leftCheek": points["leftCheek"],
        "rightCheek": points["rightCheek"],
        "forehead": points["forehead"],
        "chin": points["chin"],
        "nose": points["nose"],
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
        "chinColor": chin_color,
        "noseColor": nose_color,
        "eyeColor": eye_color,
        "eyebrowColor": eyebrow_color,
        "hairColor": hair_color,
        "dominantColors": dominant_colors,
        "contrastScore": contrast_score,
        "contrastLevel": contrast_level,
        **personal_color,
    }

    if include_debug or not classify:
        frame_result.update({
            "debugMetrics": debug_metrics,
            "csvMetrics": build_csv_metrics(debug_metrics),
        })

    return frame_result


@app.get("/")
def root():
    return {"message": "Tone-Z Backend"}


@app.post("/diagnosis")
async def diagnosis(file: UploadFile = File(...)):
    contents = await file.read()
    image = decode_image(contents)

    if image is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    result = analyze_single_frame(image)

    if "error" in result:
        return result

    return public_personal_color_result(result)


@app.post("/collect-sample")
async def collect_sample(label: str = Form(...), file: UploadFile = File(...)):
    if label not in PERSONAL_COLOR_TYPES:
        return {
            "error": "지원하지 않는 라벨입니다.",
            "allowedLabels": list(PERSONAL_COLOR_TYPES.keys()),
        }

    contents = await file.read()
    image = decode_image(contents)

    if image is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    result = analyze_single_frame(image, include_debug=True)

    if "error" in result:
        return result

    append_sample_to_csv(label, result["debugMetrics"])
    sample_profiles = calculate_profiles_from_samples()

    return {
        "message": "샘플 저장 성공",
        "label": label,
        "csvPath": str(SAMPLES_CSV_PATH),
        "savedMetrics": result["csvMetrics"],
        "sampleProfileCount": len(sample_profiles),
        "season": result["season"],
    }


@app.post("/diagnosis/video")
async def diagnosis_video(files: list[UploadFile] = File(...)):
    frame_count = len(files)
    valid_results = []
    skipped_frames = []

    selected_files = select_video_frame_files(files, max_frames=8)

    for index, file in selected_files:
        contents = await file.read()
        image = decode_image(contents)

        if image is None:
            skipped_frames.append({"index": index, "reason": "invalid_image"})
            continue

        is_valid, quality = is_valid_frame(image)
        if not is_valid:
            skipped_frames.append({"index": index, **quality})
            continue

        try:
            result = analyze_single_frame(image, classify=False, fast_video=True)
        except Exception as exc:
            skipped_frames.append({
                "index": index,
                "reason": f"analysis_failed:{exc.__class__.__name__}",
            })
            continue

        if "error" in result:
            skipped_frames.append({
                "index": index,
                "reason": result.get("error", "face_not_detected"),
            })
            continue

        valid_results.append(result)

    valid_frame_count = len(valid_results)
    skipped_summary = summarize_skipped_frames(skipped_frames)
    print(
        "[diagnosis/video]",
        {
            "frameCount": frame_count,
            "validFrameCount": valid_frame_count,
            "skippedFrameCount": len(skipped_frames),
            "skippedReasons": skipped_summary,
        },
        flush=True,
    )

    if valid_frame_count == 0:
        return {
            "error": "분석 가능한 프레임이 부족합니다.",
            "frameCount": frame_count,
            "validFrameCount": valid_frame_count,
            "skippedFrames": skipped_frames,
        }

    filtered_results = remove_metric_outliers(valid_results)
    final_result, final_metrics = choose_final_video_result(filtered_results)
    print(
        "[diagnosis/video/final]",
        {
            "filteredFrameCount": len(filtered_results),
            "stableSeason": final_metrics.get("stableSeason"),
            "voteSeason": final_metrics.get("voteSeason"),
            "voteRatio": final_metrics.get("voteRatio"),
            "finalSeason": final_result["season"],
        },
        flush=True,
    )

    return public_personal_color_result(final_result)
