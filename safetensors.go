package model2vec

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
)

// tensorMeta describes a single tensor's layout within a safetensors file.
type tensorMeta struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// safetensorsFile is a parsed safetensors file ready for tensor extraction.
type safetensorsFile struct {
	tensors    map[string]tensorMeta
	dataBuffer []byte // raw bytes after the JSON header
}

// parseSafetensors parses a safetensors file from raw bytes.
//
// Safetensors format:
//   - 8 bytes: little-endian uint64 header length N
//   - N bytes: JSON header (map of tensor name → metadata)
//   - remaining bytes: tensor data buffer (offsets are relative to this)
func parseSafetensors(data []byte) (*safetensorsFile, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("safetensors: data too short (%d bytes)", len(data))
	}

	headerLen := binary.LittleEndian.Uint64(data[:8])
	end := 8 + headerLen
	if uint64(len(data)) < end {
		return nil, fmt.Errorf("safetensors: file (%d bytes) shorter than declared header (8 + %d)", len(data), headerLen)
	}

	// Parse header JSON as a raw map so we can skip __metadata__.
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(data[8:end], &rawHeader); err != nil {
		return nil, fmt.Errorf("safetensors: failed to parse header JSON: %w", err)
	}

	tensors := make(map[string]tensorMeta, len(rawHeader))
	for name, raw := range rawHeader {
		if name == "__metadata__" {
			continue
		}
		var meta tensorMeta
		if err := json.Unmarshal(raw, &meta); err != nil {
			return nil, fmt.Errorf("safetensors: failed to parse metadata for tensor %q: %w", name, err)
		}
		tensors[name] = meta
	}

	return &safetensorsFile{
		tensors:    tensors,
		dataBuffer: data[end:],
	}, nil
}

// tensor returns the metadata and raw bytes for the named tensor.
func (sf *safetensorsFile) tensor(name string) (tensorMeta, []byte, error) {
	meta, ok := sf.tensors[name]
	if !ok {
		return tensorMeta{}, nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	start, end := meta.DataOffsets[0], meta.DataOffsets[1]
	if start < 0 || end > len(sf.dataBuffer) || start > end {
		return tensorMeta{}, nil, fmt.Errorf(
			"safetensors: invalid data offsets [%d, %d] for tensor %q (buffer size %d)",
			start, end, name, len(sf.dataBuffer),
		)
	}
	return meta, sf.dataBuffer[start:end], nil
}

// hasTensor reports whether the named tensor exists in the file.
func (sf *safetensorsFile) hasTensor(name string) bool {
	_, ok := sf.tensors[name]
	return ok
}
