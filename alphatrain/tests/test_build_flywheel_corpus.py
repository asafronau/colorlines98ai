import json

import torch

from alphatrain.scripts.build_flywheel_corpus import build


def _board():
    b = [[0] * 9 for _ in range(9)]
    b[0][0] = 1
    return b


def _move(chosen=81, visits=(9, 3)):
    return {
        'board': _board(), 'next_balls': [], 'num_next': 0,
        'chosen_move': {'sr': chosen // 81 // 9,
                        'sc': chosen // 81 % 9,
                        'tr': chosen % 81 // 9,
                        'tc': chosen % 9},
        'cand_moves': [162, 81], 'cand_visits': list(visits),
    }


def test_target_builder_preserves_raw_visits_and_behavior(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    game = {'seed': 40, 'capped': True,
            'moves': [_move(chosen=81), _move(chosen=81, visits=(7, 5))]}
    (raw / 'game_seed40.json').write_text(json.dumps(game))
    base = tmp_path / 'base.pt'
    torch.save({'model': {}}, base)
    out = tmp_path / 'target.pt'
    manifest = {
        'schema_version': 1, 'iteration': 'test', 'lineage': 'small128',
        'base_checkpoint': str(base), 'label_protocol': 'test',
        'target_output': str(out), 'anchor_output': str(tmp_path / 'a.pt'),
        'target_sources': [{
            'name': 'current', 'path': str(raw), 'format': 'moves',
            'role': 'target', 'generator': 'small128_test',
            'temperature_moves': 1,
        }],
        'anchor_sources': [],
    }
    mp = tmp_path / 'manifest.json'
    mp.write_text(json.dumps(manifest))

    built = build(mp, 'target', top_k=15)
    # Raw visit winner remains 162; chosen behavior 81 is not forced to top-1.
    assert built['pol_indices'][0, 0].item() == 162
    torch.testing.assert_close(built['pol_values'][0, :2],
                               torch.tensor([0.75, 0.25], dtype=torch.float16))
    assert built['cand_visit'][0, :2].tolist() == [9, 3]
    assert built['full_search_record'].tolist() == [0, 0]
    assert built['behavior_move'].tolist() == [81, 81]
    assert built['teacher_move'].tolist() == [162, 162]
    assert built['base_move'].tolist() == [-1, -1]
    assert built['target_weight'].tolist() == [0.0, 1.0]
    assert built['split'].tolist() == [1, 1]
    assert built['game_capped'].tolist() == [1, 1]
    assert built['search_sims'].tolist() == [0, 0]
    assert built['trajectory_kind'].tolist() == [1, 1]


def test_anchor_builder_keeps_recorded_move_and_group_split(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    state = {'board': _board(), 'next_balls': [], 'num_next': 0,
             'move': 81, 'turn': 77}
    (raw / 'game_seed41.json').write_text(json.dumps({
        'seed': 41, 'final_score': 1234, 'final_turns': 1000,
        'died': False, 'states': [state]}))
    base = tmp_path / 'base.pt'
    torch.save({'model': {}}, base)
    out = tmp_path / 'anchor.pt'
    manifest = {
        'schema_version': 1, 'iteration': 'test',
        'lineage': 'scratch18b96_test',
        'lineage_family': 'small_policy',
        'source_generator_prefix': 'scratch18b96_test_',
        'base_checkpoint': str(base),
        'target_output': str(tmp_path / 't.pt'), 'anchor_output': str(out),
        'target_sources': [],
        'anchor_sources': [{
            'name': 'current_greedy', 'path': str(raw), 'format': 'states',
            'role': 'anchor', 'generator': 'scratch18b96_test_greedy'}],
    }
    mp = tmp_path / 'manifest.json'
    mp.write_text(json.dumps(manifest))

    built = build(mp, 'anchor')
    assert built['behavior_move'].tolist() == [81]
    assert built['base_move'].tolist() == [-1]
    assert built['turn'].tolist() == [77]
    assert built['target_weight'].tolist() == [0.0]
    assert built['split'].tolist() == [0]
    assert built['trajectory_kind'].tolist() == [4]
    assert built['game_score'].tolist() == [1234]
    assert built['game_turns'].tolist() == [1000]
    assert built['game_capped'].tolist() == [1]


def test_compact_lineage_rejects_foreign_generator(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'game_seed41.json').write_text(json.dumps({
        'seed': 41, 'states': [{
            'board': _board(), 'next_balls': [], 'num_next': 0,
            'move': 81, 'turn': 0,
        }],
    }))
    base = tmp_path / 'base.pt'
    torch.save({'model': {}}, base)
    manifest = {
        'schema_version': 2, 'iteration': 'test',
        'lineage': 'scratch18b96_test',
        'lineage_family': 'small_policy',
        'source_generator_prefix': 'scratch18b96_test_',
        'base_checkpoint': str(base),
        'target_output': str(tmp_path / 'target.pt'),
        'anchor_output': str(tmp_path / 'anchor.pt'),
        'target_sources': [],
        'anchor_sources': [{
            'name': 'foreign', 'path': str(raw), 'format': 'states',
            'role': 'anchor', 'generator': 'pillar3k_teacher',
        }],
    }
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))

    try:
        build(manifest_path, 'anchor')
    except ValueError as exc:
        assert 'does not match lineage prefix' in str(exc)
    else:
        raise AssertionError('foreign generator was accepted')


def test_clean_label_budget_and_teacher_are_distinct_from_behavior(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    row = _move(chosen=81)
    row['teacher_move'] = 162
    row.update({
        'cand_prior': [-1.7, -0.2], 'cand_q': [0.4, 0.1],
        'root_value': 0.3, 'q_min': 0.1, 'q_max': 0.4,
    })
    (raw / 'game_seed44.json').write_text(json.dumps({
        'seed': 44, 'capped': True, 'behavior_sims': 8,
        'clean_label_sims': 32, 'temperature_moves': 15, 'moves': [row],
    }))
    base = tmp_path / 'base.pt'
    torch.save({'model': {}}, base)
    manifest = {
        'schema_version': 1, 'iteration': 'test', 'lineage': 'small128',
        'base_checkpoint': str(base),
        'target_output': str(tmp_path / 'target.pt'),
        'anchor_output': str(tmp_path / 'anchor.pt'),
        'target_sources': [{
            'name': 'clean', 'path': str(raw), 'format': 'moves',
            'role': 'target', 'generator': 'small128_test',
        }],
        'anchor_sources': [],
    }
    mp = tmp_path / 'manifest.json'
    mp.write_text(json.dumps(manifest))

    built = build(mp, 'target')
    assert built['behavior_move'].tolist() == [81]
    assert built['teacher_move'].tolist() == [162]
    # Candidate order follows search visits; the actor's exact legal prior
    # prefers action 81 and is preserved independently of the clean teacher.
    assert built['base_move'].tolist() == [81]
    assert built['search_sims'].tolist() == [32]
    assert built['clean_label'].tolist() == [1]
    # Temperature belongs to the behavior policy; the independent clean tree
    # still supplies a valid label on this exploratory early-turn state.
    assert built['target_weight'].tolist() == [1.0]


def test_noise_free_greedy_search_is_a_clean_label_stream(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    row = _move(chosen=162)
    row.update({
        'cand_prior': [-0.2, -1.7], 'cand_q': [0.4, 0.1],
        'root_value': 0.3, 'q_min': 0.1, 'q_max': 0.4,
    })
    (raw / 'game_seed45.json').write_text(json.dumps({
        'seed': 45, 'capped': True, 'behavior_sims': 400,
        'clean_label_sims': 0, 'temperature_moves': 0,
        'behavior_dirichlet_weight': 0.0, 'moves': [row],
    }))
    base = tmp_path / 'base.pt'
    torch.save({'model': {}}, base)
    manifest = {
        'schema_version': 1, 'iteration': 'test', 'lineage': 'small128',
        'base_checkpoint': str(base),
        'target_output': str(tmp_path / 'target.pt'),
        'anchor_output': str(tmp_path / 'anchor.pt'),
        'target_sources': [{
            'name': 'clean_exploit', 'path': str(raw), 'format': 'moves',
            'role': 'target', 'generator': 'small128_test',
        }],
        'anchor_sources': [],
    }
    mp = tmp_path / 'manifest.json'
    mp.write_text(json.dumps(manifest))

    built = build(mp, 'target')
    assert built['search_sims'].tolist() == [400]
    assert built['clean_label'].tolist() == [1]
    assert built['target_weight'].tolist() == [1.0]
    assert built['full_search_record'].tolist() == [1]
    assert built['base_move'].tolist() == [162]
    assert built['cand_visit'][0, :2].tolist() == [9, 3]
    torch.testing.assert_close(
        built['cand_prior'][0, :2].float(), torch.tensor([-0.2, -1.7]),
        atol=1e-3, rtol=0)
    torch.testing.assert_close(
        built['cand_q'][0, :2].float(), torch.tensor([0.4, 0.1]),
        atol=1e-3, rtol=0)


def test_failed_tail_weight_uses_replay_local_not_absolute_turn(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    moves = [_move() for _ in range(25)]
    (raw / 'game_seed43.json').write_text(json.dumps({
        'seed': 9001, 'original_seed': 43, 'replay_from_turn': 10_000,
        'capped': False, 'moves': moves,
    }))
    base = tmp_path / 'base.pt'
    torch.save({'model': {}}, base)
    out = tmp_path / 'target.pt'
    manifest = {
        'schema_version': 1, 'iteration': 'test', 'lineage': 'small128',
        'base_checkpoint': str(base), 'target_output': str(out),
        'anchor_output': str(tmp_path / 'a.pt'),
        'target_sources': [{
            'name': 'crisis', 'path': str(raw), 'format': 'moves',
            'role': 'target', 'generator': 'small128_test',
            'failed_tail_weight': 0.25,
        }],
        'anchor_sources': [],
    }
    mp = tmp_path / 'manifest.json'
    mp.write_text(json.dumps(manifest))

    built = build(mp, 'target')
    assert built['turn'][0].item() == 10_000
    assert built['target_weight'][:5].tolist() == [1.0] * 5
    assert built['target_weight'][5:].tolist() == [0.25] * 20


def test_resumable_memmap_build_continues_after_completed_file(
        tmp_path, monkeypatch):
    import alphatrain.scripts.build_flywheel_corpus as module

    raw = tmp_path / 'raw'
    raw.mkdir()
    for seed in (41, 42):
        (raw / f'game_seed{seed}.json').write_text(json.dumps({
            'seed': seed, 'capped': True, 'moves': [_move()],
        }))
    base = tmp_path / 'base.pt'
    torch.save({'model': {}}, base)
    out = tmp_path / 'target.pt'
    manifest = {
        'schema_version': 1, 'iteration': 'test', 'lineage': 'small128',
        'base_checkpoint': str(base), 'target_output': str(out),
        'anchor_output': str(tmp_path / 'anchor.pt'),
        'target_sources': [{
            'name': 'current', 'path': str(raw), 'format': 'moves',
            'role': 'target', 'generator': 'small128_test',
        }],
        'anchor_sources': [],
    }
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    state = tmp_path / 'build_state'
    original = module.fill_position
    calls = {'n': 0}

    def interrupt_on_second(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 2:
            raise RuntimeError('simulated interruption')
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'fill_position', interrupt_on_second)
    try:
        module.build(manifest_path, 'target', resume_dir=state,
                     checkpoint_every_files=1)
    except RuntimeError as exc:
        assert 'simulated interruption' in str(exc)
    else:
        raise AssertionError('interruption did not fire')
    progress = json.loads((state / 'progress.json').read_text())
    assert progress['next_file'] == 1
    assert progress['rows_written'] == 1

    monkeypatch.setattr(module, 'fill_position', original)
    built = module.build(manifest_path, 'target', resume_dir=state,
                         checkpoint_every_files=1)
    assert len(built['boards']) == 2
    assert built['group_seed'].tolist() == [41, 42]
    assert json.loads((state / 'progress.json').read_text())['complete']
