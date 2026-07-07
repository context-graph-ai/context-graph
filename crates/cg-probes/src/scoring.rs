//! Frozen threshold-sweep scoring math. Operating points are selected ONLY on the calibration
//! split (`pick_operating_point`), verdicts are evaluated ONLY on the disjoint verdict split at
//! that frozen threshold (`point_at`) — the anti-self-grading protocol registered in the
//! criteria. Full sweeps are always reported alongside.

/// Precision/recall at one threshold. Predictions are positive iff score >= threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct PrPoint {
    pub threshold: f32,
    pub tp: usize,
    pub fp: usize,
    pub fn_count: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

/// Evaluate `(score, is_gold_positive)` items at one threshold.
pub fn point_at(scored: &[(f32, bool)], threshold: f32) -> PrPoint {
    let (mut tp, mut fp, mut fn_count) = (0usize, 0usize, 0usize);
    for &(score, positive) in scored {
        match (score >= threshold, positive) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_count += 1,
            (false, false) => {}
        }
    }
    // No predictions → vacuous precision; no gold positives → zero recall.
    let precision = if tp + fp == 0 {
        1.0
    } else {
        tp as f32 / (tp + fp) as f32
    };
    let recall = if tp + fn_count == 0 {
        0.0
    } else {
        tp as f32 / (tp + fn_count) as f32
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    PrPoint {
        threshold,
        tp,
        fp,
        fn_count,
        precision,
        recall,
        f1,
    }
}

/// The full sweep: one point per unique score value (descending), i.e. every distinct
/// operating point the score set affords.
pub fn sweep(scored: &[(f32, bool)]) -> Vec<PrPoint> {
    let mut thresholds: Vec<f32> = scored.iter().map(|&(s, _)| s).collect();
    thresholds.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup();
    thresholds
        .into_iter()
        .map(|t| point_at(scored, t))
        .collect()
}

/// Select the frozen operating point on the CALIBRATION split: among sweep points with
/// precision >= `precision_floor` and tp > 0, maximize recall; ties break to higher precision,
/// then to the higher (more conservative) threshold. None when no point clears the floor.
pub fn pick_operating_point(calibration: &[(f32, bool)], precision_floor: f32) -> Option<f32> {
    sweep(calibration)
        .into_iter()
        .filter(|p| p.precision >= precision_floor && p.tp > 0)
        .max_by(|a, b| {
            a.recall
                .partial_cmp(&b.recall)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    a.precision
                        .partial_cmp(&b.precision)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(
                    a.threshold
                        .partial_cmp(&b.threshold)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        })
        .map(|p| p.threshold)
}
