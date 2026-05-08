from pathlib import Path

from spatialcore.enrichment import clean_term_label, gmt_to_decoupler


def test_clean_term_label():
    assert clean_term_label("HALLMARK_INTERFERON_GAMMA_RESPONSE") == "Interferon Gamma Response"


def test_gmt_to_decoupler(tmp_path: Path):
    gmt = tmp_path / "toy.gmt"
    gmt.write_text("PATHWAY\tdescription\tGeneA\tGeneB\n", encoding="utf-8")
    df = gmt_to_decoupler(gmt)
    assert list(df.columns) == ["source", "target"]
    assert df.shape[0] == 2
