<div align="center">
    <h2>Fast State-of-the-Art Static Embeddings in Go</h2>
</div>

<div align="center">
    <p>
        <a href="#quickstart"><strong>Quickstart</strong></a> •
        <a href="#features"><strong>Features</strong></a> •
        <a href="#models"><strong>Models</strong></a> •
        <!-- <a href="#performance"><strong>Performance</strong></a> • -->
        <a href="#relation-to-python-model2vec"><strong>Relation to Python Model2Vec</strong></a>
    </p>
</div>

`model2vec-go` is a Go package providing an efficient implementation for inference with [Model2Vec](https://github.com/MinishLab/model2vec) static embedding models. It is a port of the official Rust implementation, [`model2vec-rs`](https://github.com/MinishLab/model2vec-rs). Model2Vec is a technique for creating compact and fast static embedding models from sentence transformers, achieving significant reductions in model size and inference speed.

## Quickstart
You can utilize `model2vec-rs` in two ways:

1.  **As a library** in your Go projects 
2.  **TODO: As a standalone Command-Line Interface (CLI) tool** for quick terminal-based inferencing

---

### 1. Using `model2vec-rs` as a Library

**a. Add `model2vec-go` as a dependency:**

```bash
go get github.com/ammar-ahmed22/model2vec-go
```

**b. Load a model and generate embeddings:**

```go
package main

import (
    "fmt"
    "log"

    model2vec "github.com/ammar-ahmed22/model2vec-go"
)

func main() {
    // Load a model from the Hugging Face Hub or a local path.
    // Arguments: (repo_or_path, ...options)
    model, err := model2vec.FromPretrained(
        "minishlab/potion-base-8M", // Model ID from Hugging Face or local path to model directory
        // model2vec.WithToken("hf_..."),      // Optional: Hugging Face API token for private models
        // model2vec.WithNormalize(true),       // Optional: override model's default normalization
        // model2vec.WithSubfolder("subdir"),   // Optional: subfolder if model files are not at the root
    )
    if err != nil {
        log.Fatal(err)
    }

    sentences := []string{
        "Hello world",
        "Go is awesome",
    }

    // Generate embeddings using default parameters
    // (Default max_length: 512, default batch_size: 1024)
    embeddings := model.Encode(sentences)
    // embeddings is [][]float32
    fmt.Printf("Generated %d embeddings.\n", len(embeddings))

    // To generate embeddings with custom arguments:
    maxLen := 256
    customEmbeddings := model.EncodeWithArgs(
        sentences,
        &maxLen, // Optional: custom max token length for truncation (nil = no limit)
        512,     // Custom batch size for processing
    )
    fmt.Printf("Generated %d custom embeddings.\n", len(customEmbeddings))
}
```

## Features

- **Pure Go Inference:** No CGO, no native dependencies — install with a single `go get`.
- **Hugging Face Hub Integration:** Load pre-trained Model2Vec models directly from the Hugging Face Hub using model IDs, or use models from local paths.
- **Model Formats:** Supports models with F32, F16, and I8 weight types stored in `safetensors` files.
- **Batch Processing:** Encodes multiple sentences in configurable batches.
- **Configurable Encoding:** Allows customization of maximum sequence length and batch size during encoding.

## What is Model2Vec?

Model2Vec is a technique to distill large sentence transformer models into highly efficient static embedding models. This process significantly reduces model size and computational requirements for inference. For a detailed understanding of how Model2Vec works, including the distillation process and model training, please refer to the main [Model2Vec Python repository](https://github.com/MinishLab/model2vec) and its documentation.

This `model2vec-go` package provides a Go engine specifically for **inference** using these Model2Vec models.

## Models

A variety of pre-trained Model2Vec models are available on the [HuggingFace Hub](https://huggingface.co/minishlab) (MinishLab collection). These can be loaded by `model2vec-go` using their Hugging Face model ID or by providing a local path to the model files.

| Model | Language | Distilled From | Params | Task |
|---|---|---|---|---|
| [potion-base-32M](https://huggingface.co/minishlab/potion-base-32M) | English | bge-base-en-v1.5 | 32.3M | General |
| [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M) | Multilingual | bge-m3 | 128M | General |
| [potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) | English | bge-base-en-v1.5 | 32.3M | Retrieval |
| [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) | English | bge-base-en-v1.5 | 7.5M | General |
| [potion-base-4M](https://huggingface.co/minishlab/potion-base-4M) | English | bge-base-en-v1.5 | 3.7M | General |
| [potion-base-2M](https://huggingface.co/minishlab/potion-base-2M) | English | bge-base-en-v1.5 | 1.8M | General |

## Relation to Python `model2vec`

- **`model2vec-go` (This Package):** Pure Go engine for fast **Model2Vec inference**. No CGO or native dependencies.
- **`model2vec` (Python-based):** Handles model **distillation, training, fine-tuning**, and Python-based inference. See the [Python repository](https://github.com/MinishLab/model2vec).

## License

MIT

## Citing Model2Vec

If you use the Model2Vec methodology or models in your research or work, please cite the original Model2Vec project:

```bibtex
@article{minishlab2024model2vec,
  author = {Tulkens, Stephan and {van Dongen}, Thomas},
  title = {Model2Vec: Fast State-of-the-Art Static Embeddings},
  year = {2024},
  url = {https://github.com/MinishLab/model2vec}
}
```
