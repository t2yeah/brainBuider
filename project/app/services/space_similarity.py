#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import yaml

try:
    import torch
    from transformers import ClapModel, ClapProcessor
    _CLAP_AVAILABLE = True
except Exception:
    torch = None
    ClapModel = None
    ClapProcessor = None
    _CLAP_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "data" / "results"
PROJECT_ROOT = Path("/home/team-009/project")
CONFIG_ROOT = PROJECT_ROOT / "config"
DATA_ROOT = PROJECT_ROOT / "data"
TMP_ROOT = PROJECT_ROOT / "tmp"

DEFAULT_TAXONOMY_PATH = CONFIG_ROOT / "space_taxonomy.yaml"

TARGET_SR = 48000
SEGMENT_SEC = 5.0
HOP_SEC = 2.5

CLAP_MODEL_NAME = "laion/clap-htsat-unfused"

WEIGHT_CLAP = 0.70
WEIGHT_RULE = 0.30
EPS = 1e-9

# ===== 追加: 曖昧度判定パラメータ =====
AMBIGUITY_HIGH_GAP = 0.030
AMBIGUITY_MEDIUM_GAP = 0.060
DOMINANCE_STRONG_RATIO = 0.70
DOMINANCE_MEDIUM_RATIO = 0.55

SPACE_FAMILY_MAP: Dict[str, str] = {
    "dense_forest": "outdoor_nature",
    "open_field": "outdoor_nature",
    "mountain_slope": "outdoor_nature",
    "river_side": "outdoor_nature",
    "urban_street": "urban_outdoor",
    "indoor_room": "indoor",
}

VISUAL_BIAS_GROUPS: Dict[str, str] = {
    "dense_forest": "forest_bias",
    "open_field": "open_bias",
    "mountain_slope": "mountain_bias",
    "river_side": "water_bias",
    "urban_street": "urban_bias",
    "indoor_room": "indoor_bias",
}


DEFAULT_SPACE_TAXONOMY: Dict[str, Any] = {
    "version": "v3_imagable_general",
    "spaces": [
        {
            "id": "dense_forest",
            "label": "Dense Forest",
            "description": "A dense forest with nearby insects and birds, enclosed natural ambience, leaves, trees, humid air, and little human presence.",
            "feature_hints": {
                "spectral_centroid_mean": [1500, 5500],
                "spectral_bandwidth_mean": [1200, 6500],
                "zero_crossing_rate_mean": [0.03, 0.18],
                "spectral_flatness_mean": [0.003, 0.08],
                "silence_ratio": [0.10, 0.85],
            },
        },
        {
            "id": "open_field",
            "label": "Open Field",
            "description": "A wide open field with airy ambience, open sky, exposed wind, sparse natural sound sources, and a broad outdoor feeling.",
            "feature_hints": {
                "spectral_centroid_mean": [1200, 4500],
                "spectral_bandwidth_mean": [1000, 5200],
                "zero_crossing_rate_mean": [0.02, 0.16],
                "silence_ratio": [0.20, 0.95],
            },
        },
        {
            "id": "mountain_slope",
            "label": "Mountain Slope",
            "description": "A mountain slope or hillside with distant birds, wind, open natural reverberation, and expansive outdoor space.",
            "feature_hints": {
                "spectral_centroid_mean": [1300, 5000],
                "dynamic_range": [0.01, 0.22],
                "spectral_rolloff_mean": [2500, 12000],
                "silence_ratio": [0.20, 0.95],
            },
        },
        {
            "id": "river_side",
            "label": "River Side",
            "description": "A river side or stream environment with flowing water texture, continuous outdoor ambience, and natural water motion.",
            "feature_hints": {
                "spectral_bandwidth_mean": [1500, 7000],
                "spectral_rolloff_mean": [3000, 15000],
                "spectral_flatness_mean": [0.005, 0.15],
                "onset_strength_mean": [0.0, 3.0],
            },
        },
        {
            "id": "urban_street",
            "label": "Urban Street",
            "description": "An urban street with traffic-like noise, harder reflections, human or mechanical activity, and less purely natural ambience.",
            "feature_hints": {
                "spectral_centroid_mean": [800, 6500],
                "zero_crossing_rate_mean": [0.02, 0.22],
                "spectral_flatness_mean": [0.01, 0.25],
                "silence_ratio": [0.00, 0.60],
            },
        },
        {
            "id": "indoor_room",
            "label": "Indoor Room",
            "description": "An indoor room with contained reflections, controlled acoustic field, nearby sound sources, and limited openness.",
            "feature_hints": {
                "spectral_centroid_mean": [500, 4500],
                "dynamic_range": [0.00, 0.15],
                "silence_ratio": [0.20, 0.98],
            },
        },
    ],
}


@dataclass
class SpaceDefinition:
    id: str
    label: str
    description: str
    feature_hints: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class SegmentResult:
    segment_id: int
    start_sec: float
    end_sec: float
    features: Dict[str, float]
    scores: List[Dict[str, Any]]
    top_space: str
    top_score: float


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    if denom <= EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_to_score(cos: float) -> float:
    return clamp01((cos + 1.0) / 2.0)


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak < EPS:
        return audio.astype(np.float32)
    return (audio / peak).astype(np.float32)


def load_space_taxonomy(taxonomy_path: Optional[Path]) -> Tuple[str, List[SpaceDefinition]]:
    data: Dict[str, Any]

    if taxonomy_path and taxonomy_path.exists():
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        data = DEFAULT_SPACE_TAXONOMY

    version = str(data.get("version", "unknown"))
    raw_spaces = data.get("spaces", [])

    spaces: List[SpaceDefinition] = []
    for item in raw_spaces:
        spaces.append(
            SpaceDefinition(
                id=str(item["id"]),
                label=str(item.get("label", item["id"])),
                description=str(item.get("description", item["id"])),
                feature_hints=dict(item.get("feature_hints", {})),
            )
        )

    if not spaces:
        raise ValueError("space taxonomy に有効な spaces がありません。")

    return version, spaces


def extract_audio_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    if audio.size == 0:
        return {
            "duration_sec": 0.0,
            "sample_rate": float(sr),
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "peak_amplitude": 0.0,
            "dynamic_range": 0.0,
            "spectral_centroid_mean": 0.0,
            "spectral_bandwidth_mean": 0.0,
            "spectral_rolloff_mean": 0.0,
            "zero_crossing_rate_mean": 0.0,
            "spectral_flatness_mean": 0.0,
            "onset_strength_mean": 0.0,
            "silence_ratio": 1.0,
        }

    y = audio.astype(np.float32)
    frame_length = 2048
    hop_length = 512

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
    )[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
    )[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )[0]
    spectral_flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=frame_length, hop_length=hop_length
    )[0]
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    peak_amplitude = float(np.max(np.abs(y)))
    dynamic_range = float(np.percentile(np.abs(y), 95) - np.percentile(np.abs(y), 5))

    if rms.size > 0:
        threshold = max(float(np.max(rms)) * 0.10, 1e-6)
        silence_ratio = float(np.mean(rms < threshold))
    else:
        silence_ratio = 1.0

    return {
        "duration_sec": float(len(y) / sr),
        "sample_rate": float(sr),
        "rms_mean": safe_float(np.mean(rms)),
        "rms_std": safe_float(np.std(rms)),
        "peak_amplitude": peak_amplitude,
        "dynamic_range": dynamic_range,
        "spectral_centroid_mean": safe_float(np.mean(spectral_centroid)),
        "spectral_bandwidth_mean": safe_float(np.mean(spectral_bandwidth)),
        "spectral_rolloff_mean": safe_float(np.mean(spectral_rolloff)),
        "zero_crossing_rate_mean": safe_float(np.mean(zcr)),
        "spectral_flatness_mean": safe_float(np.mean(spectral_flatness)),
        "onset_strength_mean": safe_float(np.mean(onset_strength)),
        "silence_ratio": silence_ratio,
    }


def score_feature_hint(value: float, low: float, high: float) -> float:
    value = safe_float(value)
    low = safe_float(low)
    high = safe_float(high)

    if high <= low:
        return 0.5

    if low <= value <= high:
        center = (low + high) / 2.0
        half = (high - low) / 2.0
        if half <= EPS:
            return 1.0
        dist = abs(value - center) / half
        return clamp01(1.0 - 0.35 * dist)

    if value < low:
        gap = (low - value) / (high - low + EPS)
        return clamp01(0.7 - 0.7 * gap)

    gap = (value - high) / (high - low + EPS)
    return clamp01(0.7 - 0.7 * gap)


def compute_rule_score(features: Dict[str, float], space_def: SpaceDefinition) -> float:
    hints = space_def.feature_hints or {}
    if not hints:
        return 0.5

    scores = []
    for feat_name, bounds in hints.items():
        if feat_name not in features:
            continue
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        scores.append(score_feature_hint(features[feat_name], bounds[0], bounds[1]))

    if not scores:
        return 0.5

    return float(np.mean(scores))


# ===== 追加: 後続互換を崩さない補助分析 =====
def classify_ambiguity(score_gap: float) -> str:
    if score_gap < AMBIGUITY_HIGH_GAP:
        return "high"
    if score_gap < AMBIGUITY_MEDIUM_GAP:
        return "medium"
    return "low"


def classify_dominance(top1_mean: float, top3_mean_sum: float) -> Tuple[str, float]:
    if top3_mean_sum <= EPS:
        return "weak", 0.0
    ratio = top1_mean / top3_mean_sum
    if ratio >= DOMINANCE_STRONG_RATIO:
        return "strong", ratio
    if ratio >= DOMINANCE_MEDIUM_RATIO:
        return "medium", ratio
    return "weak", ratio


def build_family_scores(space_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    family_agg: Dict[str, List[float]] = {}
    for item in space_scores:
        family = SPACE_FAMILY_MAP.get(item["space_id"], "other")
        family_agg.setdefault(family, []).append(float(item["mean_score"]))

    out = []
    for family, values in family_agg.items():
        out.append(
            {
                "family": family,
                "mean_score_sum": round(float(np.sum(values)), 6),
                "mean_score_avg": round(float(np.mean(values)), 6),
                "member_count": len(values),
            }
        )
    out.sort(key=lambda x: x["mean_score_sum"], reverse=True)
    return out


def build_visual_bias(top_candidates: List[Dict[str, Any]]) -> str:
    if not top_candidates:
        return "unknown"

    ids = [c["space_id"] for c in top_candidates[:3]]
    unique_biases = [VISUAL_BIAS_GROUPS.get(x, "unknown") for x in ids]

    if "forest_bias" in unique_biases and "open_bias" in unique_biases:
        return "forest_open_mix"
    if "forest_bias" in unique_biases and "water_bias" in unique_biases:
        return "forest_water_mix"
    if "open_bias" in unique_biases and "water_bias" in unique_biases:
        return "open_water_mix"

    return unique_biases[0] if unique_biases else "unknown"


def count_transitions(segment_results: List[SegmentResult]) -> int:
    if not segment_results:
        return 0
    transitions = 0
    prev = segment_results[0].top_space
    for seg in segment_results[1:]:
        if seg.top_space != prev:
            transitions += 1
        prev = seg.top_space
    return transitions


def build_meta_analysis(
    space_scores: List[Dict[str, Any]],
    segment_results: List[SegmentResult],
) -> Dict[str, Any]:
    top1 = space_scores[0] if len(space_scores) > 0 else None
    top2 = space_scores[1] if len(space_scores) > 1 else None
    top3 = space_scores[2] if len(space_scores) > 2 else None

    top1_mean = float(top1["mean_score"]) if top1 else 0.0
    top2_mean = float(top2["mean_score"]) if top2 else 0.0
    top3_mean = float(top3["mean_score"]) if top3 else 0.0

    score_gap = max(0.0, top1_mean - top2_mean)
    ambiguity_level = classify_ambiguity(score_gap)
    dominance, dominance_ratio = classify_dominance(top1_mean, top1_mean + top2_mean + top3_mean)

    family_scores = build_family_scores(space_scores)
    best_family = family_scores[0]["family"] if family_scores else "other"
    transitions = count_transitions(segment_results)

    # segment多数決
    segment_top_counts: Dict[str, int] = {}
    for seg in segment_results:
        segment_top_counts[seg.top_space] = segment_top_counts.get(seg.top_space, 0) + 1

    segment_top_counts_sorted = sorted(
        [{"space_id": k, "count": v} for k, v in segment_top_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )

    return {
        "top1_label": top1["space_id"] if top1 else None,
        "top2_label": top2["space_id"] if top2 else None,
        "top3_label": top3["space_id"] if top3 else None,
        "top1_score": round(top1_mean, 6),
        "top2_score": round(top2_mean, 6),
        "top3_score": round(top3_mean, 6),
        "score_gap": round(score_gap, 6),
        "ambiguity_level": ambiguity_level,
        "confidence_rank": (
            "ambiguous" if ambiguity_level == "high"
            else "moderate" if ambiguity_level == "medium"
            else "clear"
        ),
        "environment_family": best_family,
        "dominance": dominance,
        "dominance_ratio_top3": round(dominance_ratio, 6),
        "temporal_variation": "present" if transitions > 0 else "stable",
        "transition_count": transitions,
        "segment_vote_ranking": segment_top_counts_sorted[:5],
        "family_scores": family_scores,
        "visual_bias": build_visual_bias(space_scores[:3]),
    }


def build_environment_structure(
    space_scores: List[Dict[str, Any]],
    meta_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    top3 = space_scores[:3]
    subtype_candidates = [x["space_id"] for x in top3]

    return {
        "family": meta_analysis.get("environment_family", "other"),
        "subtype_candidates": subtype_candidates,
        "dominance": meta_analysis.get("dominance", "weak"),
        "ambiguity_level": meta_analysis.get("ambiguity_level", "high"),
        "certainty": meta_analysis.get("confidence_rank", "ambiguous"),
        "temporal_variation": meta_analysis.get("temporal_variation", "present"),
        "visual_bias": meta_analysis.get("visual_bias", "unknown"),
    }


class ClapEmbedder:
    def __init__(self, model_name: str = CLAP_MODEL_NAME, device: Optional[str] = None) -> None:
        self.available = False
        self.device = "cpu"
        self.processor = None
        self.model = None
        self.embedding_dim = 512

        if not _CLAP_AVAILABLE:
            print("[WARN] transformers/torch がないため CLAP 無効")
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.processor = ClapProcessor.from_pretrained(model_name)
            self.model = ClapModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
        except Exception as e:
            print(f"[WARN] CLAP init failed: {e}")
            self.available = False

    def _zero_batch(self, n: int) -> np.ndarray:
        return np.zeros((n, self.embedding_dim), dtype=np.float32)

    def _zero_vec(self) -> np.ndarray:
        return np.zeros((self.embedding_dim,), dtype=np.float32)

    def _to_embedding_tensor(self, output):
        if output is None:
            raise ValueError("CLAP output is None")

        if torch.is_tensor(output):
            return output

        for attr in ["text_embeds", "audio_embeds", "pooler_output", "last_hidden_state"]:
            if hasattr(output, attr):
                value = getattr(output, attr)
                if torch.is_tensor(value):
                    if value.ndim == 3:
                        return value[:, 0, :]
                    return value

        if isinstance(output, (tuple, list)) and len(output) > 0:
            first = output[0]
            if torch.is_tensor(first):
                if first.ndim == 3:
                    return first[:, 0, :]
                return first

        raise TypeError(f"Unsupported CLAP output type: {type(output)}")

    @torch.no_grad() if _CLAP_AVAILABLE else (lambda f: f)
    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.available or not texts:
            return self._zero_batch(len(texts))

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self.model.get_text_features(**inputs)
        emb = self._to_embedding_tensor(output)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        if emb.ndim != 2:
            raise ValueError(f"Unexpected text embedding shape: {tuple(emb.shape)}")

        return emb.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad() if _CLAP_AVAILABLE else (lambda f: f)
    def get_audio_embedding(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        if not self.available:
            return self._zero_vec()

        audio_array = np.asarray(audio_array, dtype=np.float32)

        last_error = None
        for key in ("audio", "audios"):
            try:
                inputs = self.processor(
                    **{key: [audio_array]},
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output = self.model.get_audio_features(**inputs)
                emb = self._to_embedding_tensor(output)
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

                if emb.ndim == 2:
                    emb = emb[0]

                if emb.ndim != 1:
                    raise ValueError(f"Unexpected audio embedding shape: {tuple(emb.shape)}")

                return emb.detach().cpu().numpy().astype(np.float32)
            except Exception as e:
                last_error = e

        raise RuntimeError(f"CLAP audio embedding failed for both audio/audios: {last_error}")


class SpaceSimilarityMatcher:
    def __init__(
        self,
        taxonomy_path: Optional[Path] = None,
        use_clap: bool = True,
        device: Optional[str] = None,
        segment_sec: float = SEGMENT_SEC,
        hop_sec: float = HOP_SEC,
        target_sr: int = TARGET_SR,
    ) -> None:
        self.taxonomy_version, self.spaces = load_space_taxonomy(taxonomy_path)
        self.segment_sec = segment_sec
        self.hop_sec = hop_sec
        self.target_sr = target_sr

        self.clap = ClapEmbedder(device=device) if use_clap else None
        self.use_clap = bool(use_clap and self.clap and self.clap.available)
        self.text_embeddings: Dict[str, np.ndarray] = {}

        if self.use_clap:
            try:
                texts = [space.description for space in self.spaces]
                text_embs = self.clap.get_text_embeddings(texts)
                for space, emb in zip(self.spaces, text_embs):
                    self.text_embeddings[space.id] = emb
            except Exception as e:
                print(f"[WARN] text embedding init failed, fallback to rule-only: {e}")
                self.use_clap = False
                self.text_embeddings = {}

    def _load_audio(self, wav_path: str | Path) -> Tuple[np.ndarray, int]:
        wav_path = str(wav_path)
        audio, sr = librosa.load(wav_path, sr=self.target_sr, mono=True)
        audio = peak_normalize(audio)
        return audio, sr

    def _segment_audio(self, audio: np.ndarray, sr: int) -> List[Tuple[int, float, float, np.ndarray]]:
        seg_len = max(1, int(self.segment_sec * sr))
        hop_len = max(1, int(self.hop_sec * sr))

        if len(audio) <= seg_len:
            return [(0, 0.0, float(len(audio) / sr), audio)]

        segments = []
        seg_id = 0

        for start in range(0, len(audio) - seg_len + 1, hop_len):
            end = start + seg_len
            seg_audio = audio[start:end]
            segments.append((seg_id, start / sr, end / sr, seg_audio))
            seg_id += 1

        last_start = len(audio) - seg_len
        if last_start > 0:
            if not segments or int(segments[-1][1] * sr) != last_start:
                seg_audio = audio[last_start:last_start + seg_len]
                segments.append((seg_id, last_start / sr, (last_start + seg_len) / sr, seg_audio))

        return segments

    def _score_one_segment(
        self,
        segment_id: int,
        start_sec: float,
        end_sec: float,
        seg_audio: np.ndarray,
        sr: int,
    ) -> SegmentResult:
        features = extract_audio_features(seg_audio, sr)

        audio_emb = None
        if self.use_clap:
            try:
                audio_emb = self.clap.get_audio_embedding(seg_audio, sr)
                print(f"[DEBUG] segment={segment_id} clap_audio_emb_shape={audio_emb.shape}")
            except Exception as e:
                print(f"[WARN] CLAP audio embedding failed on segment {segment_id}: {e}")
                audio_emb = None

        scored = []
        for space in self.spaces:
            rule_score = compute_rule_score(features, space)

            clap_score = 0.5
            if audio_emb is not None and space.id in self.text_embeddings:
                cos = cosine_similarity(audio_emb, self.text_embeddings[space.id])
                clap_score = cosine_to_score(cos)

            if audio_emb is not None:
                final_score = clamp01((WEIGHT_CLAP * clap_score) + (WEIGHT_RULE * rule_score))
            else:
                final_score = clamp01(rule_score)

            scored.append(
                {
                    "space_id": space.id,
                    "label": space.label,
                    "clap_score": round(float(clap_score), 6),
                    "rule_score": round(float(rule_score), 6),
                    "final_score": round(float(final_score), 6),
                }
            )

        scored.sort(key=lambda x: x["final_score"], reverse=True)

        return SegmentResult(
            segment_id=segment_id,
            start_sec=round(start_sec, 3),
            end_sec=round(end_sec, 3),
            features={k: round(float(v), 6) for k, v in features.items()},
            scores=scored,
            top_space=scored[0]["space_id"],
            top_score=float(scored[0]["final_score"]),
        )

    def score_audio(self, wav_path: str | Path) -> Dict[str, Any]:
        audio, sr = self._load_audio(wav_path)
        full_features = extract_audio_features(audio, sr)
        segments = self._segment_audio(audio, sr)

        segment_results: List[SegmentResult] = []
        for seg_id, start_sec, end_sec, seg_audio in segments:
            seg_result = self._score_one_segment(seg_id, start_sec, end_sec, seg_audio, sr)
            segment_results.append(seg_result)

        aggregate: Dict[str, List[float]] = {space.id: [] for space in self.spaces}
        for seg_result in segment_results:
            for score in seg_result.scores:
                aggregate[score["space_id"]].append(float(score["final_score"]))

        space_scores = []
        for space in self.spaces:
            values = aggregate.get(space.id, [])
            mean_score = float(np.mean(values)) if values else 0.0
            max_score = float(np.max(values)) if values else 0.0
            space_scores.append(
                {
                    "space_id": space.id,
                    "label": space.label,
                    "mean_score": round(mean_score, 6),
                    "max_score": round(max_score, 6),
                }
            )

        space_scores.sort(key=lambda x: x["mean_score"], reverse=True)

        meta_analysis = build_meta_analysis(space_scores, segment_results)
        environment_structure = build_environment_structure(space_scores, meta_analysis)

        return {
            "input_path": str(wav_path),
            "space_taxonomy_version": self.taxonomy_version,
            "clap_enabled": bool(self.use_clap),
            "features": {k: round(float(v), 6) for k, v in full_features.items()},
            "top_space": space_scores[0]["space_id"] if space_scores else None,
            "top_space_label": space_scores[0]["label"] if space_scores else None,
            "space_scores": space_scores,
            "segment_space_scores": [
                {
                    "segment_id": seg.segment_id,
                    "start_sec": seg.start_sec,
                    "end_sec": seg.end_sec,
                    "top_space": seg.top_space,
                    "top_score": round(seg.top_score, 6),
                    "scores": seg.scores[:5],
                    "features": seg.features,
                }
                for seg in segment_results
            ],
            # ===== 追加キー（後続互換を崩さない）=====
            "meta_analysis": meta_analysis,
            "environment_structure": environment_structure,
        }


def resolve_audio_path(audio_id: str) -> Path:
    candidate_dirs = [
        DATA_ROOT / "uploads",
        DATA_ROOT / "audio",
        DATA_ROOT / "audios",
        DATA_ROOT / "input",
        TMP_ROOT,
        PROJECT_ROOT,
    ]
    candidate_exts = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

    candidates: List[Path] = []

    for base_dir in candidate_dirs:
        if not base_dir.exists():
            continue

        for ext in candidate_exts:
            candidates.append(base_dir / f"{audio_id}{ext}")

        try:
            for p in base_dir.rglob(f"{audio_id}*"):
                if p.is_file():
                    candidates.append(p)
        except Exception:
            pass

    seen = set()
    uniq = []
    for p in candidates:
        sp = str(p)
        if sp not in seen:
            uniq.append(p)
            seen.add(sp)

    for p in uniq:
        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError(
        f"audio_id={audio_id} に対応する音声ファイルが見つかりません。"
        f" 探索先: {[str(d) for d in candidate_dirs]}"
    )


def run_space_similarity(
    audio_id: Optional[str] = None,
    audio_path: Optional[str] = None,
    taxonomy_path: Optional[str] = None,
    use_clap: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    if not audio_id and not audio_path:
        raise ValueError("audio_id または audio_path のどちらかが必要です。")

    if audio_path:
        wav_path = Path(audio_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"audio_path が存在しません: {wav_path}")
    else:
        wav_path = resolve_audio_path(str(audio_id))

    taxonomy = Path(taxonomy_path) if taxonomy_path else DEFAULT_TAXONOMY_PATH

    matcher = SpaceSimilarityMatcher(
        taxonomy_path=taxonomy,
        use_clap=use_clap,
        device=device,
    )

    result = matcher.score_audio(wav_path)

    if audio_id:
        result["audio_id"] = audio_id

        result_dir = RESULT_ROOT / str(audio_id)
        result_dir.mkdir(parents=True, exist_ok=True)

        output_path = result_dir / "05_space_similarity.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[space_similarity] saved -> {output_path}")

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Space similarity scorer")
    parser.add_argument("--audio-id", type=str, default=None)
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--taxonomy-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-clap", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        result = run_space_similarity(
            audio_id=args.audio_id,
            audio_path=args.audio_path,
            taxonomy_path=args.taxonomy_path,
            use_clap=not args.no_clap,
            device=args.device,
        )
        if args.pretty:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        error_payload = {
            "error": str(e),
            "audio_id": args.audio_id,
            "audio_path": args.audio_path,
        }
        print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        raise


if __name__ == "__main__":
    main()