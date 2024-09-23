import os
from dataset import NameDataset
from generators import DumbNameGenerator, NGramNameGenerator
from config import (
    DATA_DIR,
    NAMES_FILE,
    URL,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    MAX_K_FOR_NGRAM,
    NUM_GENERATED_NAMES,
    RANDOM_SEED,
)

import random

def main():
    # Set the random seed for reproducibility
    random.seed(RANDOM_SEED)

    # Get the absolute path to the current directory (where main.py is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the data file
    names_file: str = os.path.join(current_dir, DATA_DIR, NAMES_FILE)

    dataset: NameDataset = NameDataset(names_file, URL)

    dumb_generator: DumbNameGenerator = DumbNameGenerator(
        dataset.get_alphabet(), dataset.get_end_token()
    )

    # Train N-gram models with k from 1 to MAX_K_FOR_NGRAM
    ngram_generators: dict[int, NGramNameGenerator] = {}

    for k in range(1, MAX_K_FOR_NGRAM + 1):
        ngram_generator: NGramNameGenerator = NGramNameGenerator(
            dataset.get_alphabet(), dataset.get_end_token(), k=k
        )

        print(f"\nTraining {ngram_generator.get_generator_name()}...")

        # Train the model on batches of training data
        for batch in dataset.get_batch(is_train=True, batch_size=BATCH_SIZE):
            ngram_generator.train(batch)

        ngram_generators[k] = ngram_generator

    # Evaluate all generators on a random batch of test data
    eval_batch: list[list[str]] = [list("john>"), list("mary>"), list("robert>")]

    print("\nEvaluating generators on a sample batch:")

    for generator in [dumb_generator] + list(ngram_generators.values()):

        avg_log_likelihood: float = generator.evaluate_batch(eval_batch)

        print(
            f"Average log likelihood for {generator.get_generator_name()}: {avg_log_likelihood:.4f}"
        )

    # Additional evaluation on larger batches of test data
    print("\nEvaluating generators on larger batches:")

    for i, batch in enumerate(
        dataset.get_batch(is_train=False, batch_size=EVAL_BATCH_SIZE)
    ):

        print(f"Batch {i + 1}:")

        for generator in [dumb_generator] + list(ngram_generators.values()):

            avg_log_likelihood: float = generator.evaluate_batch(batch)

            print(
                f"{generator.get_generator_name()} - Average log likelihood: {avg_log_likelihood:.4f}"
            )

        if i == 2:
            break

    print(
        "\nGenerated names (DumbNameGenerator):",
        dumb_generator.generate_names(NUM_GENERATED_NAMES),
    )

    for k, ngram_generator in ngram_generators.items():

        print(
            f"\nGenerated names ({ngram_generator.get_generator_name()}):",
            ngram_generator.generate_names(NUM_GENERATED_NAMES),
        )

    batch: list[list[str]] = [list("joh"), list("mar")]

    print("\nProbabilities for next character in each sequence of the batch:")

    for generator in [dumb_generator] + list(ngram_generators.values()):

        print(f"\n{generator.get_generator_name()}:")

        probabilities: list[dict[str, float]] = generator.predict(batch)

        for i, probs in enumerate(probabilities):
            print(f"Sequence {i + 1}: {''.join(batch[i])}")

            top_5: list[tuple[str, float]] = sorted(
                probs.items(), key=lambda x: x[1], reverse=True
            )[:5]

            for char, prob in top_5:
                print(f"  {char}: {prob:.4f}")


if __name__ == "__main__":
    main()
