package model2vec_test

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

// longHelloText is "hello" repeated 1000 times, space-joined — matching the
// Rust test's `vec!["hello"; 1000].join(" ")`.
func longHelloText() string {
	parts := make([]string, 1000)
	for i := range parts {
		parts[i] = "hello"
	}
	return strings.Join(parts, " ")
}

// loadExpected reads a Python-generated embedding fixture into [][]float32.
func loadExpected(t *testing.T, path string) [][]float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var expected [][]float32
	if err := json.Unmarshal(raw, &expected); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return expected
}

// TestEncodeMatchesPython compares Go-encoded embeddings against Python ground
// truth (model2vec library) for short and long inputs, with relative tolerance
// 1e-5.
func TestEncodeMatchesPython(t *testing.T) {
	m := loadFixture(t, "test-model-float32")

	cases := []struct {
		fixture string
		inputs  []string
	}{
		{"testdata/embeddings_short.json", []string{"hello world"}},
		{"testdata/embeddings_long.json", []string{longHelloText()}},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.fixture, func(t *testing.T) {
			expected := loadExpected(t, tc.fixture)
			output := m.Encode(tc.inputs)

			if len(output) != len(expected) {
				t.Fatalf("sentence count mismatch: got %d want %d", len(output), len(expected))
			}
			if len(output[0]) != len(expected[0]) {
				t.Fatalf("dim mismatch: got %d want %d", len(output[0]), len(expected[0]))
			}
			for i := range expected[0] {
				assertRelativeEq(t, output[0][i], expected[0][i], 1e-5,
					fmt.Sprintf("element %d", i))
			}
		})
	}
}

// TestEncodeMatchesPythonVocabQuantized validates the vocab-quantized variant
// against Python ground truth for the long input.
func TestEncodeMatchesPythonVocabQuantized(t *testing.T) {
	m := loadFixture(t, "test-model-vocab-quantized")

	expected := loadExpected(t, "testdata/embeddings_vocab_quantized.json")
	output := m.Encode([]string{longHelloText()})

	if len(output) != len(expected) {
		t.Fatalf("sentence count mismatch: got %d want %d", len(output), len(expected))
	}
	if len(output[0]) != len(expected[0]) {
		t.Fatalf("dim mismatch: got %d want %d", len(output[0]), len(expected[0]))
	}
	for i := range expected[0] {
		assertRelativeEq(t, output[0][i], expected[0][i], 1e-5,
			fmt.Sprintf("element %d", i))
	}
}
