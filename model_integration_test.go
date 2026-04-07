//go:build integration

// Integration tests that download the live model from the HuggingFace Hub.
// Run with: go test -tags integration ./...
package model2vec_test

import (
	"testing"

	model2vec "github.com/ammar-ahmed22/model2vec-go"
)

const testModel = "minishlab/potion-base-8M"

// loadModel downloads and loads the test model from the HuggingFace Hub.
func loadModel(t *testing.T) *model2vec.StaticModel {
	t.Helper()
	m, err := model2vec.FromPretrained(testModel)
	if err != nil {
		t.Fatalf("FromPretrained(%q): %v", testModel, err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func TestFromPretrainedAndEncode(t *testing.T) {
	m := loadModel(t)

	sentences := []string{"Hello world", "Go is awesome"}
	embeddings := m.Encode(sentences)

	if len(embeddings) != len(sentences) {
		t.Fatalf("expected %d embeddings, got %d", len(sentences), len(embeddings))
	}

	dim := m.Dims()
	if dim == 0 {
		t.Fatal("Dims() returned 0")
	}

	for i, emb := range embeddings {
		if len(emb) != dim {
			t.Errorf("embeddings[%d]: expected length %d, got %d", i, dim, len(emb))
		}
	}
}

func TestEncodeWithArgs(t *testing.T) {
	m := loadModel(t)

	sentences := []string{"Hello world", "Go is awesome"}
	maxLen := 256
	embeddings := m.EncodeWithArgs(sentences, &maxLen, 512)

	if len(embeddings) != len(sentences) {
		t.Fatalf("expected %d embeddings, got %d", len(sentences), len(embeddings))
	}
	for i, emb := range embeddings {
		if len(emb) != m.Dims() {
			t.Errorf("embeddings[%d]: expected length %d, got %d", i, m.Dims(), len(emb))
		}
	}
}

func TestEncodeWithArgsNoLimit(t *testing.T) {
	m := loadModel(t)

	sentences := []string{"A short sentence."}
	embeddings := m.EncodeWithArgs(sentences, nil, 64)

	if len(embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(embeddings))
	}
	if len(embeddings[0]) != m.Dims() {
		t.Errorf("expected dim %d, got %d", m.Dims(), len(embeddings[0]))
	}
}

func TestEncodeConsistencyHub(t *testing.T) {
	m := loadModel(t)

	s := "The quick brown fox"
	a := m.EncodeSingle(s)
	b := m.EncodeSingle(s)

	if len(a) != len(b) {
		t.Fatalf("lengths differ: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("embedding[%d]: %v != %v", i, a[i], b[i])
		}
	}
}
