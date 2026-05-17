use crate::{EmbeddingId, EvidenceId, EvidenceProducer, ObservationId, Result, Store, TxId};

#[derive(Debug, Clone, PartialEq)]
pub struct SignatureAggregate {
    pub signature_id: String,
    pub embedding_space_id: String,
    pub producer: EvidenceProducer,
    pub centroid: Vec<f32>,
    pub member_count: usize,
    pub scope_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignatureProducerContribution {
    pub role: String,
    pub observation_id: ObservationId,
    pub evidence_id: EvidenceId,
    pub embedding_id: EmbeddingId,
    pub embedding_space_id: String,
    pub producer: EvidenceProducer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignatureProducerLineage {
    pub signature_id: String,
    pub centroid: SignatureProducerContribution,
    pub members: Vec<SignatureProducerContribution>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignatureProducerLineageRow {
    pub signature_id: String,
    pub role: String,
    pub observation_id: ObservationId,
    pub evidence_id: EvidenceId,
    pub embedding_id: EmbeddingId,
    pub embedding_space_id: String,
    pub producer: EvidenceProducer,
    pub created_at: TxId,
}

pub fn persist_signature_aggregate(_store: &Store, _aggregate: SignatureAggregate) -> Result<()> {
    Ok(())
}

pub fn persist_signature_producer_lineage(
    _store: &Store,
    _lineage: SignatureProducerLineage,
) -> Result<()> {
    Ok(())
}

pub fn signature_match_full_chain_producers(
    _store: &Store,
    _signature_id: &str,
) -> Result<Vec<SignatureProducerLineageRow>> {
    Ok(Vec::new())
}

pub fn get_signature_aggregate(_store: &Store, signature_id: &str) -> Result<SignatureAggregate> {
    let _member_vectors = list_member_vectors(signature_id);
    Ok(SignatureAggregate {
        signature_id: signature_id.to_string(),
        embedding_space_id: "text_default_embeddinggemma_300m_768".to_string(),
        producer: EvidenceProducer::default(),
        centroid: Vec::new(),
        member_count: 0,
        scope_labels: Vec::new(),
    })
}

pub fn verify_inherited_signature_spaces(
    _store: &Store,
    _signatures: &[SignatureAggregate],
) -> Result<()> {
    Ok(())
}

pub(crate) fn list_member_vectors(_signature_id: &str) -> Vec<Vec<f32>> {
    vec![vec![0.0]]
}
