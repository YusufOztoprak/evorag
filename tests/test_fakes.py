import pytest

from tests.fakes import cosine_similarity


def test_identical_vectors_have_similarity_one():
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

def test_orthogonal_vectors_have_similarity_zero():
    assert cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)