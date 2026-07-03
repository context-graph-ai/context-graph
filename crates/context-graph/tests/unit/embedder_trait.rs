use context_graph::embedder::mock::MockEmbedder;
use context_graph::{
    CG_EDGE_SYSTEM_TAG, CgError, DistanceMetric, Embedder, EmbeddingInput, EmbeddingOutput,
    EmbeddingSpace, EvidenceKind, EvidenceProducer,
};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};

fn expected_mock_vector(kind: &str, value: &str, dimension: usize) -> Vec<f32> {
    let mut vector = Vec::with_capacity(dimension);
    let mut block = 0_u64;
    while vector.len() < dimension {
        let mut hasher = Sha256::new();
        hasher.update(kind.as_bytes());
        hasher.update([0]);
        hasher.update(value.as_bytes());
        hasher.update(block.to_le_bytes());
        for chunk in hasher.finalize().chunks_exact(4) {
            if vector.len() == dimension {
                break;
            }
            let bytes = <[u8; 4]>::try_from(chunk).expect("sha256 chunk");
            vector.push(u32::from_le_bytes(bytes) as f32 / u32::MAX as f32);
        }
        block += 1;
    }
    let norm = vector_l2_norm(&vector);
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn vector_l2_norm(vector: &[f32]) -> f32 {
    vector
        .iter()
        .map(|component| component.powi(2))
        .sum::<f32>()
        .sqrt()
}

#[derive(Debug)]
struct RoutingProbeEmbedder {
    text_calls: AtomicUsize,
    image_calls: AtomicUsize,
}

impl RoutingProbeEmbedder {
    fn new() -> Self {
        Self {
            text_calls: AtomicUsize::new(0),
            image_calls: AtomicUsize::new(0),
        }
    }
}

impl Embedder for RoutingProbeEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        let (kind, value, dimension) = match input {
            EmbeddingInput::Text(text) => {
                self.text_calls.fetch_add(1, Ordering::SeqCst);
                ("text", text, 4)
            }
            EmbeddingInput::ImageFrame { uri } => {
                self.image_calls.fetch_add(1, Ordering::SeqCst);
                ("image", uri, 512)
            }
            EmbeddingInput::ImageBytes(bytes) => {
                self.image_calls.fetch_add(1, Ordering::SeqCst);
                ("image", format!("bytes:{}", bytes.len()), 512)
            }
            EmbeddingInput::AudioSegment { .. } => {
                return Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::AudioSegment,
                });
            }
            EmbeddingInput::VideoSegment { .. } => {
                return Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::VideoSegment,
                });
            }
            EmbeddingInput::StructuredSignal(_) => {
                return Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::StructuredSignal,
                });
            }
        };
        Ok(EmbeddingOutput {
            embedding_space_id: format!("{kind}_probe_space"),
            dimension,
            producer: EvidenceProducer {
                system: CG_EDGE_SYSTEM_TAG.to_string(),
                model_name: format!("{kind}-probe"),
                model_version: "1".to_string(),
                pipeline_version: "routing".to_string(),
            },
            vector: expected_mock_vector(kind, &value, dimension),
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text, EvidenceKind::ImageFrame]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: "routing_probe".to_string(),
            dimension: 512,
            metric: DistanceMetric::Cosine,
            model_family: "probe".to_string(),
            model_name: "routing-probe".to_string(),
            model_version: "1".to_string(),
            supported_evidence_kinds: self.supported_inputs(),
            default_for_evidence_kind: None,
            bindings: Vec::new(),
        }
    }
}

#[test]
fn embedder_trait_implementation() {
    let embedder = MockEmbedder::new("unit_text_space", 4);
    let output = embedder
        .embed(EmbeddingInput::Text("remember the screenshot".to_string()))
        .expect("mock embedder implements Embedder");
    assert_eq!(
        output.vector,
        expected_mock_vector("text", "remember the screenshot", 4)
    );
    assert_eq!(output.dimension, 4);
    assert_eq!(output.producer.system, CG_EDGE_SYSTEM_TAG);
    assert_eq!(output.producer.model_name, "mock");
}

#[test]
fn embedder_trait_typed_input() {
    let embedder = MockEmbedder::new("unit_text_space", 4);
    let result = embedder.embed(EmbeddingInput::ImageFrame {
        uri: "cas://sha256:1111111111111111111111111111111111111111111111111111111111111111"
            .to_string(),
    });
    assert!(matches!(
        result,
        Err(CgError::UnsupportedEvidenceKind {
            kind: EvidenceKind::ImageFrame
        })
    ));
}

#[test]
fn embedding_input_routes_text_evidence_to_text_handler() {
    let embedder = MockEmbedder::new("unit_text_space", 4);
    let output = embedder
        .embed(EmbeddingInput::Text("route text".to_string()))
        .expect("text route");
    assert_eq!(output.vector, expected_mock_vector("text", "route text", 4));
}

#[test]
fn embedding_input_routes_image_frame_to_vision_handler_when_supported() {
    let embedder = RoutingProbeEmbedder::new();
    let uri = "cas://sha256:2222222222222222222222222222222222222222222222222222222222222222";
    assert!(
        embedder
            .supported_inputs()
            .contains(&EvidenceKind::ImageFrame)
    );
    let output = embedder
        .embed(EmbeddingInput::ImageFrame {
            uri: uri.to_string(),
        })
        .expect("image frame route");
    assert_eq!(output.embedding_space_id, "image_probe_space");
    assert_eq!(output.dimension, 512);
    assert_eq!(output.producer.system, CG_EDGE_SYSTEM_TAG);
    assert_eq!(output.producer.model_name, "image-probe");
    assert_eq!(output.vector, expected_mock_vector("image", uri, 512));
    assert_eq!(embedder.text_calls.load(Ordering::SeqCst), 0);
    assert_eq!(embedder.image_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn embedder_returns_embedding_space_id_not_model_name() {
    let embedder = MockEmbedder::new("unit_text_space", 4);
    let output = embedder
        .embed(EmbeddingInput::Text("space identity".to_string()))
        .expect("mock output");
    assert_eq!(output.embedding_space_id, "unit_text_space");
    assert_eq!(output.dimension, 4);
    assert_eq!(output.producer.model_name, "mock");
}

#[test]
fn box_dyn_embedder_can_be_constructed() {
    let embedder: Box<dyn Embedder> = Box::new(MockEmbedder::new("boxed_space", 3));
    let output = embedder
        .embed(EmbeddingInput::Text("boxed".to_string()))
        .expect("boxed embedder output");
    assert_eq!(output.vector, expected_mock_vector("text", "boxed", 3));
    assert_eq!(output.embedding_space_id, "boxed_space");
    assert_eq!(output.dimension, 3);
    assert_eq!(output.producer.system, CG_EDGE_SYSTEM_TAG);
}
