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
    let _ = (scored, threshold);
    todo!("probe impl commit")
}

/// The full sweep: one point per unique score value (descending), i.e. every distinct
/// operating point the score set affords.
pub fn sweep(scored: &[(f32, bool)]) -> Vec<PrPoint> {
    let _ = scored;
    todo!("probe impl commit")
}

/// Select the frozen operating point on the CALIBRATION split: among sweep points with
/// precision >= `precision_floor` and tp > 0, maximize recall; ties break to higher precision,
/// then to the higher (more conservative) threshold. None when no point clears the floor.
pub fn pick_operating_point(calibration: &[(f32, bool)], precision_floor: f32) -> Option<f32> {
    let _ = (calibration, precision_floor);
    todo!("probe impl commit")
}
