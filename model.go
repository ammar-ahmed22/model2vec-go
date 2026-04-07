// Package model2vec provides a Go implementation of Model2Vec static embedding models.
// It loads pre-trained models from the HuggingFace Hub or a local directory and
// encodes text into fixed-size float32 embedding vectors via mean-pooling over
// per-token static embeddings.
//
// Basic usage:
//
//	model, err := model2vec.FromPretrained("minishlab/potion-base-8M")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.Close()
//
//	embeddings := model.Encode([]string{"Hello world", "Go is great"})
//	fmt.Printf("dim=%d\n", len(embeddings[0]))
package model2vec

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	"github.com/x448/float16"
)

// StaticModel is a Model2Vec static embedding model.
// It maps input text to fixed-size float32 embedding vectors via mean-pooling
// of pre-computed per-token embeddings, with optional L2 normalization.
//
// StaticModel is not safe for concurrent use from multiple goroutines.
// Use separate instances or add external synchronization.
type StaticModel struct {
	tk             *tokenizer.Tokenizer
	embeddings     []float32 // flat, row-major: token i's embedding starts at i*cols
	rows, cols     int
	weights        []float32 // optional per-token weights (nil if absent)
	tokenMapping   []int     // optional token ID → row remapping (nil if absent)
	normalize      bool
	medianTokenLen int
	unkTokenID     *int // nil if the tokenizer declares no unk_token
}

// tokenizerJSON is used to parse tokenizer.json for metadata only.
// Supports two vocab schemas:
//   - WordPiece/BPE: model.vocab is an object {token: id}, model.unk_token is a string.
//   - Unigram: model.vocab is an array [[token, score], ...] where the index is
//     the token ID, and model.unk_id is the integer ID of the unk token.
type tokenizerJSON struct {
	Model struct {
		Vocab    json.RawMessage `json:"vocab"`
		UnkToken *string         `json:"unk_token"`
		UnkID    *int            `json:"unk_id"`
	} `json:"model"`
}

// FromPretrained loads a Model2Vec model from a HuggingFace Hub repository ID
// or a local directory path. The following files must be present:
// tokenizer.json, model.safetensors, config.json.
//
// When loading from the HuggingFace Hub, files are downloaded and cached in
// ~/.cache/huggingface/hub/ (or $HF_HOME/hub if set), matching the Python
// huggingface_hub cache layout so downloads are shared with Python tooling.
func FromPretrained(repoOrPath string, opts ...Option) (*StaticModel, error) {
	o := &Options{}
	for _, opt := range opts {
		opt(o)
	}

	tokPath, mdlPath, cfgPath, err := resolveModelFiles(repoOrPath, o.Token, o.Subfolder)
	if err != nil {
		return nil, fmt.Errorf("model2vec: resolving model files: %w", err)
	}

	cfgBytes, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, fmt.Errorf("model2vec: reading config.json: %w", err)
	}
	mdlBytes, err := os.ReadFile(mdlPath)
	if err != nil {
		return nil, fmt.Errorf("model2vec: reading model.safetensors: %w", err)
	}
	tokBytes, err := os.ReadFile(tokPath)
	if err != nil {
		return nil, fmt.Errorf("model2vec: reading tokenizer.json: %w", err)
	}

	return FromBytes(tokBytes, mdlBytes, cfgBytes, opts...)
}

// FromBytes loads a Model2Vec model from in-memory byte slices for the
// tokenizer.json, model.safetensors, and config.json files. It performs no
// filesystem or network access, mirroring the Rust library's from_bytes().
func FromBytes(tokenizerBytes, modelBytes, configBytes []byte, opts ...Option) (*StaticModel, error) {
	o := &Options{}
	for _, opt := range opts {
		opt(o)
	}

	// --- config.json: read the normalize flag ---
	var cfg struct {
		Normalize *bool `json:"normalize"`
	}
	if err := json.Unmarshal(configBytes, &cfg); err != nil {
		return nil, fmt.Errorf("model2vec: parsing config.json: %w", err)
	}
	normalize := true // default when absent
	if cfg.Normalize != nil {
		normalize = *cfg.Normalize
	}
	if o.Normalize != nil {
		normalize = *o.Normalize // explicit option wins
	}

	// --- model.safetensors: load embeddings (and optional weights / mapping) ---
	sf, err := parseSafetensors(modelBytes)
	if err != nil {
		return nil, fmt.Errorf("model2vec: parsing model.safetensors: %w", err)
	}

	// The embeddings tensor is named "embeddings"; fall back to "0" for older models.
	embName := "embeddings"
	if !sf.hasTensor(embName) {
		embName = "0"
	}
	embMeta, embRaw, err := sf.tensor(embName)
	if err != nil {
		return nil, fmt.Errorf("model2vec: extracting embeddings tensor: %w", err)
	}
	if len(embMeta.Shape) != 2 {
		return nil, fmt.Errorf("model2vec: expected 2-D embeddings tensor, got shape %v", embMeta.Shape)
	}
	rows, cols := embMeta.Shape[0], embMeta.Shape[1]

	embeddings, err := rawToFloat32(embRaw, embMeta.Dtype)
	if err != nil {
		return nil, fmt.Errorf("model2vec: converting embeddings: %w", err)
	}
	if len(embeddings) != rows*cols {
		return nil, fmt.Errorf("model2vec: embeddings length mismatch: got %d, want %d×%d=%d",
			len(embeddings), rows, cols, rows*cols)
	}

	// Optional per-token weights (present in quantized models).
	var weights []float32
	if sf.hasTensor("weights") {
		wMeta, wRaw, wErr := sf.tensor("weights")
		if wErr != nil {
			return nil, fmt.Errorf("model2vec: extracting weights tensor: %w", wErr)
		}
		weights, err = rawToFloat32(wRaw, wMeta.Dtype)
		if err != nil {
			return nil, fmt.Errorf("model2vec: converting weights: %w", err)
		}
	}

	// Optional token ID → row index mapping (present in quantized models).
	var tokenMapping []int
	if sf.hasTensor("mapping") {
		_, mRaw, mErr := sf.tensor("mapping")
		if mErr != nil {
			return nil, fmt.Errorf("model2vec: extracting mapping tensor: %w", mErr)
		}
		if len(mRaw)%4 != 0 {
			return nil, fmt.Errorf("model2vec: mapping tensor byte length (%d) not divisible by 4", len(mRaw))
		}
		tokenMapping = make([]int, len(mRaw)/4)
		for i := range tokenMapping {
			tokenMapping[i] = int(int32(binary.LittleEndian.Uint32(mRaw[i*4:])))
		}
	}

	// --- tokenizer.json: load tokenizer and extract metadata ---
	tk, err := pretrained.FromReader(bytes.NewReader(tokenizerBytes))
	if err != nil {
		return nil, fmt.Errorf("model2vec: loading tokenizer: %w", err)
	}

	medianLen, unkID, err := computeTokenizerMetadata(tokenizerBytes)
	if err != nil {
		return nil, fmt.Errorf("model2vec: computing tokenizer metadata: %w", err)
	}

	return &StaticModel{
		tk:             tk,
		embeddings:     embeddings,
		rows:           rows,
		cols:           cols,
		weights:        weights,
		tokenMapping:   tokenMapping,
		normalize:      normalize,
		medianTokenLen: medianLen,
		unkTokenID:     unkID,
	}, nil
}

// Encode generates embeddings for the given sentences using default parameters
// (maxLength=512, batchSize=1024).
func (m *StaticModel) Encode(sentences []string) [][]float32 {
	maxLen := 512
	return m.EncodeWithArgs(sentences, &maxLen, 1024)
}

// EncodeWithArgs generates embeddings with custom parameters.
//   - maxLength: maximum number of tokens per sentence (nil = no limit).
//   - batchSize: how many sentences to process at a time (controls memory use).
func (m *StaticModel) EncodeWithArgs(sentences []string, maxLength *int, batchSize int) [][]float32 {
	result := make([][]float32, 0, len(sentences))

	for i := 0; i < len(sentences); i += batchSize {
		end := i + batchSize
		if end > len(sentences) {
			end = len(sentences)
		}
		for _, text := range sentences[i:end] {
			// Char-level pre-truncation heuristic to avoid feeding excessively
			// long strings to the tokenizer (mirrors the Rust implementation).
			t := text
			if maxLength != nil {
				t = truncateString(t, *maxLength, m.medianTokenLen)
			}

			// Tokenize without special tokens, matching the Rust implementation.
			enc, err := m.tk.EncodeSingle(t, false)
			var ids []int
			if err == nil && enc != nil {
				ids = enc.Ids
			}

			// Strip unknown-token IDs.
			if m.unkTokenID != nil {
				unkID := *m.unkTokenID
				filtered := ids[:0]
				for _, id := range ids {
					if id != unkID {
						filtered = append(filtered, id)
					}
				}
				ids = filtered
			}

			// Token-level truncation.
			if maxLength != nil && len(ids) > *maxLength {
				ids = ids[:*maxLength]
			}

			result = append(result, m.poolIDs(ids))
		}
	}

	return result
}

// EncodeSingle encodes a single sentence and returns its embedding vector.
func (m *StaticModel) EncodeSingle(sentence string) []float32 {
	out := m.Encode([]string{sentence})
	if len(out) == 0 {
		return make([]float32, m.cols)
	}
	return out[0]
}

// Dims returns the dimensionality of the model's embedding vectors.
func (m *StaticModel) Dims() int {
	return m.cols
}

// Close is a no-op retained for API compatibility.
// The pure-Go tokenizer has no native resources to release.
func (m *StaticModel) Close() {}

// --- internal helpers ---

// computeTokenizerMetadata parses tokenizer.json to derive:
//   - medianLen: the median byte-length of vocabulary token strings, used for
//     char-level pre-truncation.
//   - unkID: the ID of the unk_token, or nil if the tokenizer has none.
func computeTokenizerMetadata(tokBytes []byte) (medianLen int, unkID *int, err error) {
	var tj tokenizerJSON
	if err = json.Unmarshal(tokBytes, &tj); err != nil {
		return 0, nil, fmt.Errorf("parsing tokenizer.json: %w", err)
	}

	// vocab may be either an object (WordPiece/BPE) or an array (Unigram).
	var vocabMap map[string]int
	var tokenLens []int
	if len(tj.Model.Vocab) > 0 {
		switch tj.Model.Vocab[0] {
		case '{':
			if err = json.Unmarshal(tj.Model.Vocab, &vocabMap); err != nil {
				return 0, nil, fmt.Errorf("parsing vocab object: %w", err)
			}
			tokenLens = make([]int, 0, len(vocabMap))
			for tok := range vocabMap {
				tokenLens = append(tokenLens, len(tok))
			}
		case '[':
			// Unigram: [[token, score], ...]
			var entries []json.RawMessage
			if err = json.Unmarshal(tj.Model.Vocab, &entries); err != nil {
				return 0, nil, fmt.Errorf("parsing vocab array: %w", err)
			}
			tokenLens = make([]int, 0, len(entries))
			for _, e := range entries {
				var pair []json.RawMessage
				if err = json.Unmarshal(e, &pair); err != nil || len(pair) == 0 {
					continue
				}
				var tok string
				if err = json.Unmarshal(pair[0], &tok); err != nil {
					continue
				}
				tokenLens = append(tokenLens, len(tok))
			}
		}
	}

	sort.Ints(tokenLens)
	if len(tokenLens) == 0 {
		medianLen = 1
	} else {
		medianLen = tokenLens[len(tokenLens)/2]
	}

	// Resolve unk ID from either unk_token (lookup in object vocab) or unk_id (Unigram).
	if tj.Model.UnkToken != nil && *tj.Model.UnkToken != "" && vocabMap != nil {
		if id, ok := vocabMap[*tj.Model.UnkToken]; ok {
			unkID = &id
		}
	} else if tj.Model.UnkID != nil {
		id := *tj.Model.UnkID
		unkID = &id
	}

	return medianLen, unkID, nil
}

// rawToFloat32 converts the raw byte slice from a safetensors tensor to []float32.
// Supported dtypes: F32, F16, I8, F64.
func rawToFloat32(data []byte, dtype string) ([]float32, error) {
	switch dtype {
	case "F32":
		if len(data)%4 != 0 {
			return nil, fmt.Errorf("F32 data length %d not divisible by 4", len(data))
		}
		out := make([]float32, len(data)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return out, nil

	case "F16":
		if len(data)%2 != 0 {
			return nil, fmt.Errorf("F16 data length %d not divisible by 2", len(data))
		}
		out := make([]float32, len(data)/2)
		for i := range out {
			out[i] = float16.Frombits(binary.LittleEndian.Uint16(data[i*2:])).Float32()
		}
		return out, nil

	case "I8":
		out := make([]float32, len(data))
		for i, b := range data {
			out[i] = float32(int8(b))
		}
		return out, nil

	case "F64":
		if len(data)%8 != 0 {
			return nil, fmt.Errorf("F64 data length %d not divisible by 8", len(data))
		}
		out := make([]float32, len(data)/8)
		for i := range out {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:])))
		}
		return out, nil

	default:
		return nil, fmt.Errorf("unsupported tensor dtype: %q", dtype)
	}
}

// truncateString truncates s to at most maxTokens*medianLen Unicode code points.
// This is a cheap heuristic to avoid tokenizing arbitrarily long strings.
func truncateString(s string, maxTokens, medianLen int) string {
	maxChars := maxTokens * medianLen
	count := 0
	for i := range s { // iterates over rune boundaries
		if count >= maxChars {
			return s[:i]
		}
		count++
	}
	return s
}

// poolIDs mean-pools the embedding rows for the given token IDs.
// Applies optional token mapping and per-token weights, then optionally
// L2-normalizes the result.
func (m *StaticModel) poolIDs(ids []int) []float32 {
	sum := make([]float32, m.cols)

	for _, id := range ids {
		tok := id

		// Apply token mapping if present.
		rowIdx := tok
		if m.tokenMapping != nil && tok < len(m.tokenMapping) {
			rowIdx = m.tokenMapping[tok]
		}

		// Per-token scale (default 1.0).
		scale := float32(1.0)
		if m.weights != nil && tok < len(m.weights) {
			scale = m.weights[tok]
		}

		// Accumulate the embedding row.
		base := rowIdx * m.cols
		if rowIdx >= 0 && base+m.cols <= len(m.embeddings) {
			for j := 0; j < m.cols; j++ {
				sum[j] += m.embeddings[base+j] * scale
			}
		}
	}

	// Mean pool.
	n := len(ids)
	if n == 0 {
		n = 1
	}
	denom := float32(n)
	for j := range sum {
		sum[j] /= denom
	}

	// Optional L2 normalization.
	if m.normalize {
		var sqSum float32
		for _, v := range sum {
			sqSum += v * v
		}
		norm := float32(math.Sqrt(float64(sqSum)))
		if norm < 1e-12 {
			norm = 1e-12
		}
		for j := range sum {
			sum[j] /= norm
		}
	}

	return sum
}
