package model2vec

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const hfBaseURL = "https://huggingface.co"

// resolveModelFiles returns local filesystem paths for tokenizer.json,
// model.safetensors, and config.json.
//
// If repoOrPath is an existing local directory, files are read directly from it
// (with an optional subfolder). Otherwise, repoOrPath is treated as a
// HuggingFace Hub repository ID and the files are downloaded and cached.
func resolveModelFiles(repoOrPath, token, subfolder string) (tokenizerPath, modelPath, configPath string, err error) {
	if info, statErr := os.Stat(repoOrPath); statErr == nil && info.IsDir() {
		base := repoOrPath
		if subfolder != "" {
			base = filepath.Join(base, subfolder)
		}
		tokenizerPath = filepath.Join(base, "tokenizer.json")
		modelPath = filepath.Join(base, "model.safetensors")
		configPath = filepath.Join(base, "config.json")

		for _, p := range []string{tokenizerPath, modelPath, configPath} {
			if _, e := os.Stat(p); e != nil {
				return "", "", "", fmt.Errorf("model file not found: %s", p)
			}
		}
		return tokenizerPath, modelPath, configPath, nil
	}

	// Treat repoOrPath as a HuggingFace Hub repo ID.
	cacheDir := hfCacheDir()
	prefix := ""
	if subfolder != "" {
		prefix = subfolder + "/"
	}

	filenames := []string{
		prefix + "tokenizer.json",
		prefix + "model.safetensors",
		prefix + "config.json",
	}

	paths := make([]string, len(filenames))
	for i, filename := range filenames {
		p, dlErr := downloadHFFile(repoOrPath, filename, token, cacheDir)
		if dlErr != nil {
			return "", "", "", fmt.Errorf("downloading %s from %q: %w", filename, repoOrPath, dlErr)
		}
		paths[i] = p
	}

	return paths[0], paths[1], paths[2], nil
}

// hfCacheDir returns the HuggingFace hub cache directory, following the same
// convention as the Python huggingface_hub library ($HF_HOME/hub or
// $HUGGINGFACE_HUB_CACHE or ~/.cache/huggingface/hub).
func hfCacheDir() string {
	if d := os.Getenv("HF_HOME"); d != "" {
		return filepath.Join(d, "hub")
	}
	if d := os.Getenv("HUGGINGFACE_HUB_CACHE"); d != "" {
		return d
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "huggingface", "hub")
	}
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// downloadHFFile downloads a single file from a HuggingFace Hub repository,
// caching it under cacheDir. Returns the path to the local cached file.
// Subsequent calls with the same arguments return the cached path immediately.
func downloadHFFile(repoID, filename, token, cacheDir string) (string, error) {
	// Build a cache path: models--{org}--{name}/{filename}
	safeRepo := "models--" + strings.ReplaceAll(repoID, "/", "--")
	cachedPath := filepath.Join(cacheDir, safeRepo, filepath.FromSlash(filename))

	if _, err := os.Stat(cachedPath); err == nil {
		return cachedPath, nil // already cached
	}

	url := fmt.Sprintf("%s/%s/resolve/main/%s", hfBaseURL, repoID, filename)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("building request: %w", err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("User-Agent", "model2vec-go")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("HTTP request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected HTTP %d for %s", resp.StatusCode, url)
	}

	if err := os.MkdirAll(filepath.Dir(cachedPath), 0o755); err != nil {
		return "", fmt.Errorf("creating cache directory: %w", err)
	}

	// Write to a temp file then rename to avoid partial writes being cached.
	tmp, err := os.CreateTemp(filepath.Dir(cachedPath), ".download-*")
	if err != nil {
		return "", fmt.Errorf("creating temp file: %w", err)
	}
	tmpName := tmp.Name()

	if _, err = io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return "", fmt.Errorf("writing download: %w", err)
	}
	if err = tmp.Close(); err != nil {
		os.Remove(tmpName)
		return "", fmt.Errorf("closing temp file: %w", err)
	}
	if err = os.Rename(tmpName, cachedPath); err != nil {
		os.Remove(tmpName)
		return "", fmt.Errorf("moving to cache: %w", err)
	}

	return cachedPath, nil
}
