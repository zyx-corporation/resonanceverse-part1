"""BoolQ スタンドアロン（answer）と GLUE 形式（label）のラベル抽出。"""

from datasets import Dataset

from experiments.slm_downstream import _texts_and_labels_from_glue


def test_boolq_labels_from_answer_column() -> None:
    ds = Dataset.from_dict(
        {
            "question": ["q1", "q2"],
            "passage": ["p1", "p2"],
            "answer": [True, False],
        }
    )
    texts, labels = _texts_and_labels_from_glue(ds, "boolq")
    assert texts == ["q1 p1", "q2 p2"]
    assert labels == [1, 0]


def test_boolq_labels_from_label_column() -> None:
    ds = Dataset.from_dict(
        {
            "question": ["q1"],
            "passage": ["p1"],
            "label": [1],
        }
    )
    texts, labels = _texts_and_labels_from_glue(ds, "boolq")
    assert labels == [1]


def test_sst2_labels() -> None:
    ds = Dataset.from_dict({"sentence": ["x"], "label": [1]})
    texts, labels = _texts_and_labels_from_glue(ds, "sst2")
    assert texts == ["x"] and labels == [1]
