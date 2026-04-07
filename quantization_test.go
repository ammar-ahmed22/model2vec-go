package model2vec_test

import (
	"fmt"
	"testing"
)

// TestQuantizedModelsMatchFloat32 verifies that quantized model variants
// produce embeddings element-wise close to the float32 reference, within a
// 10% relative tolerance.
func TestQuantizedModelsMatchFloat32(t *testing.T) {
	ref := loadFixture(t, "test-model-float32").Encode([]string{"hello world"})[0]

	for _, quant := range []string{"float16", "int8"} {
		quant := quant
		t.Run(quant, func(t *testing.T) {
			m := loadFixture(t, "test-model-"+quant)
			emb := m.Encode([]string{"hello world"})[0]

			if len(emb) != len(ref) {
				t.Fatalf("dim mismatch: %s=%d float32=%d", quant, len(emb), len(ref))
			}
			for i := range ref {
				assertRelativeEq(t, emb[i], ref[i], 1e-1,
					fmt.Sprintf("%s element %d", quant, i))
			}
		})
	}
}
