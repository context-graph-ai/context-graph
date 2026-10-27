//! Staged merging deduplicates repeated items within a focused pass as well as across passes.

use std::cell::RefCell;
use std::collections::BTreeMap;

use context_graph::harness::{CaptureHarnessConfig, ResolutionKind, run_capture};
use context_graph::ingest::{
    CaptureRequest, ConversationExtractor, EntitySeed, ExtractionPass, ExtractionRequest,
    StructuredHarvest,
};
use context_graph::{Result as IngestResult, StoreConfig};

use super::fake_extractor::open_store_with_lexical_embedder;

struct DupeEntityExtractor {
    calls: RefCell<Vec<ExtractionPass>>,
    harvests: BTreeMap<&'static str, StructuredHarvest>,
}

impl ConversationExtractor for DupeEntityExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> IngestResult<StructuredHarvest> {
        self.extract_pass(request, ExtractionPass::Full)
    }

    fn extract_pass(
        &self,
        _request: ExtractionRequest<'_>,
        pass: ExtractionPass,
    ) -> IngestResult<StructuredHarvest> {
        self.calls.borrow_mut().push(pass);
        let key = match pass {
            ExtractionPass::Full => "full",
            ExtractionPass::EntitiesOnly => "entities",
            ExtractionPass::DecisionsOnly => "decisions",
            ExtractionPass::SessionIntention => "session",
        };
        Ok(self.harvests.get(key).cloned().unwrap_or_default())
    }
}

#[test]
fn staged_merge_dedupes_within_the_focused_pass() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    // The focused entities pass returns the SAME entity twice (a realistic model wobble: the
    // model lists an entity once per mention) plus one genuinely new one.
    let full = StructuredHarvest {
        entities: vec![EntitySeed {
            key: "e1".to_string(),
            name: "cg-bench".to_string(),
            ..EntitySeed::default()
        }],
        ..StructuredHarvest::default()
    };
    let entities_pass = StructuredHarvest {
        entities: vec![
            EntitySeed {
                key: "e2".to_string(),
                name: "scoring.rs".to_string(),
                ..EntitySeed::default()
            },
            EntitySeed {
                key: "e3".to_string(),
                name: "Scoring.rs".to_string(), // same entity, case wobble
                ..EntitySeed::default()
            },
        ],
        ..StructuredHarvest::default()
    };
    let extractor = DupeEntityExtractor {
        calls: RefCell::new(Vec::new()),
        harvests: BTreeMap::from([("full", full), ("entities", entities_pass)]),
    };
    let config = CaptureHarnessConfig {
        staged_extraction: true,
        ..CaptureHarnessConfig::default()
    };
    let (_report, harness_report) = run_capture(
        &store,
        CaptureRequest {
            context_label: "merge-dedupe-test".to_string(),
            transcript: "The cg-bench crate and scoring.rs are discussed here.".to_string(),
            model_available: true,
        },
        &extractor,
        &config,
    )
    .expect("capture succeeds");

    let entity_resolutions = harness_report
        .stage3_resolution_verdicts
        .iter()
        .filter(|v| v.kind == ResolutionKind::Entity)
        .count();
    assert_eq!(
        entity_resolutions, 2,
        "cg-bench + scoring.rs once each — the intra-pass duplicate collapses in the merge"
    );
}
