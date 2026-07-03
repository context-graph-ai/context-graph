//! The bench report (criterion C7): per-category numbers vs the FROZEN bars (criteria commit
//! 99d99ee), the hosted-ceiling row next to every local row (the model-gap vs pipeline-gap
//! separator), per-lever history, and a printed verdict — in whichever direction it falls.

/// The owner-ratified bars (docs/cg-correctness-criteria.md). Rendered into every report so
/// numbers are always read AGAINST the registration.
pub const BAR_INTENTION_FIDELITY: f32 = 0.75;
pub const BAR_ATTACHMENT: f32 = 0.75;
pub const BAR_DECISION_RECALL: f32 = 0.60;
pub const BAR_DECISION_PRECISION: f32 = 0.75;
pub const BAR_ENTITY_RECALL: f32 = 0.70;
pub const BAR_ENTITY_TYPE_ACCURACY: f32 = 0.80;

#[derive(Debug, Clone, PartialEq)]
pub struct CategoryNumbers {
    pub intention_fidelity: f32,
    pub attachment: f32,
    pub decision_recall: f32,
    pub decision_precision: f32,
    pub entity_recall: f32,
    pub entity_type_accuracy: f32,
    /// Reported, not gated (frozen criteria).
    pub observation_recall: f32,
    pub duplication_rate: f32,
    pub sessions: usize,
}

impl CategoryNumbers {
    /// The gated categories against the frozen bars.
    pub fn passes_bars(&self) -> bool {
        self.intention_fidelity >= BAR_INTENTION_FIDELITY
            && self.attachment >= BAR_ATTACHMENT
            && self.decision_recall >= BAR_DECISION_RECALL
            && self.decision_precision >= BAR_DECISION_PRECISION
            && self.entity_recall >= BAR_ENTITY_RECALL
            && self.entity_type_accuracy >= BAR_ENTITY_TYPE_ACCURACY
    }
}

/// One measured variant: a lever configuration on a model tier (or the hosted ceiling).
#[derive(Debug, Clone, PartialEq)]
pub struct VariantRow {
    pub label: String,
    pub model: String,
    pub numbers: CategoryNumbers,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchReport {
    pub title: String,
    pub rows: Vec<VariantRow>,
}

impl BenchReport {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            rows: Vec::new(),
        }
    }

    pub fn push_row(&mut self, row: VariantRow) {
        self.rows.push(row);
    }

    /// Render the report: bars, every variant row (per-lever history), verdict per row.
    pub fn render_markdown(&self) -> String {
        todo!("cg-correctness C7: bench report rendering not implemented yet")
    }
}
