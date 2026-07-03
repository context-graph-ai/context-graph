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
        let mut out = format!(
            "# cg-correctness bench — {}\n\nBars (criteria commit 99d99ee, owner-ratified): \
             intention ≥ {BAR_INTENTION_FIDELITY:.2}, attachment ≥ {BAR_ATTACHMENT:.2}, \
             decision recall ≥ {BAR_DECISION_RECALL:.2}, decision precision ≥ \
             {BAR_DECISION_PRECISION:.2}, entity recall ≥ {BAR_ENTITY_RECALL:.2}, entity type \
             ≥ {BAR_ENTITY_TYPE_ACCURACY:.2}; observations reported, not gated.\n\n\
             | variant | model | sessions | intention | attach | dec-recall | dec-prec | \
             ent-recall | ent-type | obs | dup | verdict |\n\
             |---|---|---|---|---|---|---|---|---|---|---|---|\n",
            self.title
        );
        for row in &self.rows {
            let n = &row.numbers;
            let verdict = if n.passes_bars() { "PASS" } else { "FAIL" };
            out.push_str(&format!(
                "| {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {} |\n",
                row.label,
                row.model,
                n.sessions,
                n.intention_fidelity,
                n.attachment,
                n.decision_recall,
                n.decision_precision,
                n.entity_recall,
                n.entity_type_accuracy,
                n.observation_recall,
                n.duplication_rate,
                verdict,
            ));
        }
        out
    }
}
