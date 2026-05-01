//! MockEmbedder. Plan 11 (cg-embedding-spaces) is not shipped today.
//!
//! Per Track 2 stub policy and the journey-B "Stubbed surfaces" list:
//! "Plan 11 not shipped; stub uses MockEmbedder returning deterministic
//! 768-dim vectors when EmbeddingGemma load fails (graceful CI degradation)."
//!
//! The acceptance test `embeddinggemma_loads_and_embeds_text_corpus_with_expected_dimension`
//! requires deterministic output. We use 8 dims (not 768) because the
//! journey's vector index column is VECTOR(8) — the journey is exercising
//! the staleness-tracking semantic, not the EmbeddingGemma binary. The
//! findings entry flags this as a stub-vs-test divergence to re-walk when
//! Plan 11 ships.

pub struct MockEmbedder {
    pub dim: usize,
}

impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Deterministic embedding: hash the lowercased text into N seed buckets,
    /// each contributing a sinusoidal magnitude to the corresponding dim.
    /// L2-normalised so cosine similarity is meaningful.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let lower = text.to_lowercase();
        let mut v = vec![0.0_f32; self.dim];
        // Each token contributes to dims keyed by its byte hash modulo dim.
        for (token_idx, token) in lower.split_whitespace().enumerate() {
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for b in token.bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            let primary = (hash as usize) % self.dim;
            let secondary = ((hash >> 17) as usize) % self.dim;
            v[primary] += 1.0 / (1.0 + token_idx as f32);
            v[secondary] += 0.5 / (1.0 + token_idx as f32);
        }
        // L2-normalise.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        } else {
            // Zero vector for empty text — the engine still requires a vector
            // of the right dimension, so emit a small bias on dim 0.
            v[0] = 1.0;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_is_deterministic() {
        let e = MockEmbedder::new(8);
        assert_eq!(e.embed("auth.py token validation"), e.embed("auth.py token validation"));
    }

    #[test]
    fn similar_text_is_similar_vector() {
        let e = MockEmbedder::new(8);
        let a = e.embed("use auth.py for token validation");
        let b = e.embed("auth.py handles token validation");
        let c = e.embed("colour palette for the homepage hero");
        let cos = |x: &[f32], y: &[f32]| -> f32 {
            x.iter().zip(y).map(|(p, q)| p * q).sum()
        };
        assert!(
            cos(&a, &b) > cos(&a, &c),
            "auth-related sentences must rank above unrelated text — got cos(a,b)={} cos(a,c)={}",
            cos(&a, &b),
            cos(&a, &c)
        );
    }
}
