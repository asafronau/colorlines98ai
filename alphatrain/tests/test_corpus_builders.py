"""Regression tests for corpus-composition choices."""

from alphatrain.scripts.build_hardce_corpora import rows_of_game


def _game(n=10_100, seed=17):
    return {'seed': seed, 'moves': list(range(n))}


def test_uncapped_rows_historical_default(monkeypatch):
    monkeypatch.delenv('SELFPLAY_CAP', raising=False)
    rows = rows_of_game(_game(), 'uncapped')
    assert len(rows) == 10_000


def test_uncapped_rows_can_be_explicitly_unlimited(monkeypatch):
    monkeypatch.setenv('SELFPLAY_CAP', '7')
    rows = rows_of_game(_game(), 'uncapped', selfplay_cap=0)
    assert rows == list(range(30, 10_100))
