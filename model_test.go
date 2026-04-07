package model2vec_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	model2vec "github.com/ammar-ahmed22/model2vec-go"
)

const fixtureFloat32 = "testdata/test-model-float32"

// loadFixtureModel loads the small float32 fixture model with the given options.
func loadFixtureModel(t *testing.T, opts ...model2vec.Option) *model2vec.StaticModel {
	t.Helper()
	m, err := model2vec.FromPretrained(fixtureFloat32, opts...)
	if err != nil {
		t.Fatalf("FromPretrained(%q): %v", fixtureFloat32, err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// TestEncodeEmptyInput verifies that encoding an empty input slice produces no
// embeddings.
func TestEncodeEmptyInput(t *testing.T) {
	m := loadFixtureModel(t)
	embs := m.Encode(nil)
	if len(embs) != 0 {
		t.Fatalf("expected no embeddings for empty input, got %d", len(embs))
	}
}

// TestEncodeEmptySentence verifies that encoding a single empty sentence yields
// a zero vector of the model's dimensionality.
func TestEncodeEmptySentence(t *testing.T) {
	m := loadFixtureModel(t)
	embs := m.Encode([]string{""})
	if len(embs) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(embs))
	}
	if got, want := len(embs[0]), m.Dims(); got != want {
		t.Fatalf("expected dim %d, got %d", want, got)
	}
	for i, v := range embs[0] {
		if v != 0.0 {
			t.Errorf("embedding[%d] = %v, want 0.0", i, v)
		}
	}
}

// TestEncodeSingle verifies that EncodeSingle and Encode return shape-compatible
// results for the same input.
func TestEncodeSingle(t *testing.T) {
	m := loadFixtureModel(t)
	const sentence = "hello world"

	oneD := m.EncodeSingle(sentence)
	twoD := m.Encode([]string{sentence})

	if len(oneD) == 0 {
		t.Fatal("EncodeSingle returned an empty vector")
	}
	if len(twoD) != 1 {
		t.Fatalf("Encode([1]) returned %d vectors, want 1", len(twoD))
	}
	if len(twoD[0]) != len(oneD) {
		t.Errorf("dim mismatch: encode=%d encode_single=%d", len(twoD[0]), len(oneD))
	}
}

// TestNormalizationFlagOverride verifies that WithNormalize(false) overrides
// the model's default normalization setting.
func TestNormalizationFlagOverride(t *testing.T) {
	mNorm := loadFixtureModel(t)
	embNorm := mNorm.Encode([]string{"test sentence"})[0]
	normNorm := l2Norm(embNorm)

	mNoNorm := loadFixtureModel(t, model2vec.WithNormalize(false))
	embNoNorm := mNoNorm.Encode([]string{"test sentence"})[0]
	normNoNorm := l2Norm(embNoNorm)

	if math.Abs(float64(normNorm)-1.0) >= 1e-5 {
		t.Errorf("normalized vector should have unit norm, got %v", normNorm)
	}
	if normNoNorm <= normNorm {
		t.Errorf("expected un-normalized norm (%v) to exceed normalized norm (%v)",
			normNoNorm, normNorm)
	}
}

// TestFromBytesMatchesFromPretrained verifies that FromBytes produces a model
// identical to FromPretrained for the same fixture.
func TestFromBytesMatchesFromPretrained(t *testing.T) {
	fromPath, err := model2vec.FromPretrained(fixtureFloat32)
	if err != nil {
		t.Fatalf("FromPretrained: %v", err)
	}
	t.Cleanup(func() { fromPath.Close() })

	tokBytes := mustReadFile(t, filepath.Join(fixtureFloat32, "tokenizer.json"))
	mdlBytes := mustReadFile(t, filepath.Join(fixtureFloat32, "model.safetensors"))
	cfgBytes := mustReadFile(t, filepath.Join(fixtureFloat32, "config.json"))

	fromBytes, err := model2vec.FromBytes(tokBytes, mdlBytes, cfgBytes)
	if err != nil {
		t.Fatalf("FromBytes: %v", err)
	}
	t.Cleanup(func() { fromBytes.Close() })

	const query = "hello world"
	a := fromPath.EncodeSingle(query)
	b := fromBytes.EncodeSingle(query)

	if len(a) != len(b) {
		t.Fatalf("dim mismatch: from_path=%d from_bytes=%d", len(a), len(b))
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) >= 1e-6 {
			t.Errorf("element %d: from_path=%v from_bytes=%v", i, a[i], b[i])
		}
	}
}

// l2Norm returns the L2 norm of v.
func l2Norm(v []float32) float32 {
	var s float32
	for _, x := range v {
		s += x * x
	}
	return float32(math.Sqrt(float64(s)))
}

func mustReadFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return b
}
