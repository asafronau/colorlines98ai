/// Cross-language RNG parity test.
/// Verifies Rust SimpleRng produces identical output to Python SimpleRng.
use colorlines98::rng::SimpleRng;
use serde_json::Value;
use std::fs;

fn load_reference() -> Value {
    let data = fs::read_to_string("../game/tests/rng_reference.json")
        .expect("Failed to load rng_reference.json");
    serde_json::from_str(&data).expect("Failed to parse JSON")
}

#[test]
fn test_u64_sequence_matches_python() {
    let refs = load_reference();

    for seed_str in ["0", "42", "100", "777", "999999"] {
        let seed: u64 = seed_str.parse().unwrap();
        let mut rng = SimpleRng::new(seed);

        let expected = refs[seed_str]["first_10_u64"]
            .as_array()
            .unwrap();

        for (i, exp) in expected.iter().enumerate() {
            let exp_val = exp.as_u64().unwrap();
            let got = rng.next_u64();
            assert_eq!(
                got, exp_val,
                "seed={seed}, index={i}: Rust={got}, Python={exp_val}"
            );
        }
    }
}

#[test]
fn test_choice_no_replace_matches_python() {
    let refs = load_reference();

    for seed_str in ["0", "42", "100", "777", "999999"] {
        let seed: u64 = seed_str.parse().unwrap();
        let mut rng = SimpleRng::new(seed);

        let expected: Vec<usize> = refs[seed_str]["choice_81_3"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let got = rng.choice_no_replace(81, 3);
        assert_eq!(
            got, expected,
            "seed={seed}: choice_no_replace mismatch"
        );
    }
}

#[test]
fn test_integers_matches_python() {
    let refs = load_reference();

    for seed_str in ["0", "42", "100", "777", "999999"] {
        let seed: u64 = seed_str.parse().unwrap();
        let mut rng = SimpleRng::new(seed);

        // choice_no_replace consumes 3 u64 values first
        let _ = rng.choice_no_replace(81, 3);

        let expected: Vec<i64> = refs[seed_str]["integers_1_8_3"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();

        let got = rng.integers(1, 8, 3);
        assert_eq!(
            got, expected,
            "seed={seed}: integers mismatch"
        );
    }
}
