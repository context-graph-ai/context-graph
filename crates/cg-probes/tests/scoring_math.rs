//! RED tests — frozen sweep/threshold scoring math, and the anti-self-grading operating-point
//! rule (calibration picks, verdict evaluates).

use cg_probes::scoring::{PrPoint, pick_operating_point, point_at, sweep};

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

const SCORED: &[(f32, bool)] = &[
    (0.9, true),
    (0.8, false),
    (0.7, true),
    (0.6, true),
    (0.4, false),
];

#[test]
fn point_at_counts_exactly() {
    let p = point_at(SCORED, 0.65);
    assert_eq!((p.tp, p.fp, p.fn_count), (2, 1, 1));
    assert!(close(p.precision, 2.0 / 3.0));
    assert!(close(p.recall, 2.0 / 3.0));
    assert!(close(p.f1, 2.0 / 3.0));

    let strict = point_at(SCORED, 0.95);
    assert_eq!((strict.tp, strict.fp, strict.fn_count), (0, 0, 3));
    assert!(close(strict.recall, 0.0));
    assert!(close(strict.f1, 0.0));
}

#[test]
fn sweep_has_one_point_per_unique_score_descending() {
    let points = sweep(SCORED);
    let thresholds: Vec<f32> = points.iter().map(|p| p.threshold).collect();
    assert_eq!(thresholds, vec![0.9, 0.8, 0.7, 0.6, 0.4]);
    let at_09: &PrPoint = &points[0];
    assert_eq!((at_09.tp, at_09.fp, at_09.fn_count), (1, 0, 2));
    assert!(close(at_09.precision, 1.0));
    assert!(close(at_09.recall, 1.0 / 3.0));
}

#[test]
fn operating_point_respects_the_precision_floor() {
    // Only t=0.9 reaches precision 1.0; t=0.6 has precision 3/4.
    assert_eq!(pick_operating_point(SCORED, 0.9), Some(0.9));
    // Floor 0.75 admits t=0.6 (p=0.75, recall 1.0) and t=0.9 (p=1.0, recall 1/3):
    // recall wins → 0.6.
    assert_eq!(pick_operating_point(SCORED, 0.75), Some(0.6));
    // Unreachable floor → None (never "best effort").
    assert_eq!(
        pick_operating_point(&[(0.9, false), (0.5, true)], 0.9),
        None
    );
}

#[test]
fn operating_point_tiebreak_prefers_precision_then_higher_threshold() {
    // Two thresholds with identical recall: 0.8 (tp=1 fp=0) and 0.6 (tp=1 fp=1).
    let scored = &[(0.8, true), (0.6, false), (0.3, false)];
    assert_eq!(pick_operating_point(scored, 0.4), Some(0.8));
    // Identical recall AND precision at two thresholds → the higher (more conservative) one.
    let flat = &[(0.9, true), (0.7, true), (0.2, false)];
    // t=0.9: tp1 fn1 p=1.0 r=0.5 ; t=0.7: tp2 p=1.0 r=1.0 → r wins, 0.7. Then a duplicate-recall
    // shape: single positive scored twice above every negative.
    assert_eq!(pick_operating_point(flat, 0.9), Some(0.7));
    let dup = &[(0.9, true), (0.7, false)];
    assert_eq!(pick_operating_point(dup, 0.5), Some(0.9));
}

#[test]
fn degenerate_inputs_do_not_panic_or_divide_by_zero() {
    let empty: &[(f32, bool)] = &[];
    let p = point_at(empty, 0.5);
    assert_eq!((p.tp, p.fp, p.fn_count), (0, 0, 0));
    assert!(close(p.precision, 1.0)); // no predictions → vacuous precision
    assert!(close(p.recall, 0.0));
    assert!(sweep(empty).is_empty());
    assert_eq!(pick_operating_point(empty, 0.5), None);
}
