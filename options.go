package model2vec

// Options holds configuration for loading a model.
type Options struct {
	// Token is an optional HuggingFace API token for accessing private models.
	Token string

	// Normalize overrides the model's default normalization setting from config.json.
	// If nil, the value from config.json is used (default: true for most models).
	Normalize *bool

	// Subfolder is an optional subdirectory within the HF repo or local path
	// that contains the model files.
	Subfolder string
}

// Option is a functional option for configuring model loading.
type Option func(*Options)

// WithToken sets the HuggingFace API token for private model access.
func WithToken(token string) Option {
	return func(o *Options) {
		o.Token = token
	}
}

// WithNormalize overrides whether output embeddings are L2-normalized.
// If not set, the value from the model's config.json is used.
func WithNormalize(normalize bool) Option {
	return func(o *Options) {
		o.Normalize = &normalize
	}
}

// WithSubfolder sets an optional subdirectory within the HF repo or local path
// where model files (tokenizer.json, model.safetensors, config.json) reside.
func WithSubfolder(subfolder string) Option {
	return func(o *Options) {
		o.Subfolder = subfolder
	}
}
