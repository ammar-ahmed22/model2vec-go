package model2vec_test

import (
	"fmt"
	"log"

	model2vec "github.com/ammar-ahmed22/model2vec-go"
)

func ExampleFromPretrained() {
	// Load a model from the HuggingFace Hub.
	// Files are downloaded once and cached in ~/.cache/huggingface/hub/.
	model, err := model2vec.FromPretrained("minishlab/potion-base-8M")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	sentences := []string{
		"Hello world",
		"Go is awesome",
	}

	// Encode with default parameters (maxLength=512, batchSize=1024).
	embeddings := model.Encode(sentences)
	fmt.Printf("Generated %d embeddings of dimension %d\n", len(embeddings), model.Dims())
}

func ExampleStaticModel_EncodeWithArgs() {
	model, err := model2vec.FromPretrained("minishlab/potion-base-8M")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	sentences := []string{"Hello world", "Go is awesome"}

	// Encode with a custom max token length and batch size.
	maxLen := 256
	embeddings := model.EncodeWithArgs(sentences, &maxLen, 512)
	fmt.Printf("Generated %d embeddings\n", len(embeddings))
}

func ExampleStaticModel_EncodeSingle() {
	model, err := model2vec.FromPretrained("minishlab/potion-base-8M")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	emb := model.EncodeSingle("Hello world")
	fmt.Printf("Embedding dimension: %d\n", len(emb))
}
