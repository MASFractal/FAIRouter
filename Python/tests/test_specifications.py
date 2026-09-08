import numpy as np

from fai_router.enums import Style
from fai_router.settings import Settings
from fai_router.specifications import Specifications
from fai_router.services import InputFeaturesService


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def requested():
    return Specifications(style_type=Style.SCIENTIFIC, symbol_length=3000, word_length=450,
                          paragraph_count=6, section_count=3, table_count=1, heading_depth=2,
                          avg_sentence_length=18, readability_score=30, term_density=0.6, formality_score=0.9)


def test_dimension_matches_settings():
    assert len(requested().feature_vector()) == Settings.features_spec_dim() == 23


def test_scaling_lets_style_matter():
    """Те же три случая, что в измерении на C#: до нормировки детский стиль получал 0,9998."""
    r = requested().feature_vector()
    childish = requested()
    childish.style_type = Style.CHILDREN
    childish.avg_sentence_length, childish.readability_score = 6, 95
    childish.term_density, childish.formality_score = 0.05, 0.1
    half = requested()
    half.symbol_length, half.word_length, half.paragraph_count = 1500, 225, 3

    assert abs(cosine(requested().feature_vector(), r) - 1.0) < 1e-9
    assert abs(cosine(half.feature_vector(), r) - 0.9963) < 2e-3
    assert abs(cosine(childish.feature_vector(), r) - 0.6408) < 2e-3


def test_no_single_coordinate_dominates():
    r = requested().feature_vector()
    shares = r * r / (r @ r)
    assert shares.max() < 0.25


def test_bounded_fields_are_clamped():
    spec = Specifications(term_density=80, formality_score=-3, readability_score=900)
    assert (spec.term_density, spec.formality_score, spec.readability_score) == (1.0, 0.0, 100.0)


def test_negative_count_does_not_poison_vector():
    spec = Specifications(symbol_length=-500)
    assert np.isfinite(spec.feature_vector()).all()


def test_roundtrip_dict():
    spec = requested()
    spec.language, spec.has_references = "ru", True
    restored = Specifications.from_dict(spec.to_dict())
    assert np.allclose(restored.feature_vector(), spec.feature_vector())
    assert restored.language == "ru" and restored.has_references


def test_real_prompt_keeps_specification_visible():
    """В сырых токенах счетчики давали 100% длины вектора, и роутер не видел типа задачи."""
    prompt = ("Напиши научный обзор методов регуляризации нейросетей на 1500 знаков, "
              "раздели на 3 раздела, добавь таблицу сравнения и ссылки на источники.")
    features = InputFeaturesService.get_features(prompt)
    features.input_specifications = requested()
    vector = features.feature_vector()
    head_share = float(vector[:Settings.FEATURES_DIM] @ vector[:Settings.FEATURES_DIM])
    assert len(vector) == Settings.full_dim() == 25
    assert head_share < 0.2
