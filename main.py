import os
from dataset import NameDataset
from generators import DumbNameGenerator, NGramNameGenerator

def main():
    # Get the absolute path to the current directory (where main.py is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the data file
    names_file: str = os.path.join(current_dir, "data", "names.txt")
    
    url: str = (
        "https://raw.githubusercontent.com/dominictarr/random-name/master/first-names.txt"
    )

    dataset: NameDataset = NameDataset(names_file, url)

    dumb_generator: DumbNameGenerator = DumbNameGenerator(
        dataset.get_alphabet(), dataset.get_end_token()
    )

    # Train N-gram models with k from 1 to 4
    ngram_generators: dict[int, NGramNameGenerator] = {}

    for k in range(1, 5):
        ngram_generator: NGramNameGenerator = NGramNameGenerator(
            dataset.get_alphabet(), dataset.get_end_token(), k=k
        )

        print(f"\nTraining {ngram_generator.get_generator_name()}...")

        # Train the model on batches of data
        for batch in dataset.get_batch(batch_size=100):
            ngram_generator.train(batch)

        ngram_generators[k] = ngram_generator

    # Evaluate all generators on a random batch of data
    eval_batch: list[list[str]] = [list("john>"), list("mary>"), list("robert>")]

    print("\nEvaluating generators on a sample batch:")

    for generator in [dumb_generator] + list(ngram_generators.values()):

        avg_log_likelihood: float = generator.evaluate_batch(eval_batch)

        print(
            f"Average log likelihood for {generator.get_generator_name()}: {avg_log_likelihood:.4f}"
        )

    # Additional evaluation on larger batches of data
    print("\nEvaluating generators on larger batches:")

    for i, batch in enumerate(dataset.get_batch(batch_size=100)):

        print(f"Batch {i + 1}:")

        for generator in [dumb_generator] + list(ngram_generators.values()):

            avg_log_likelihood: float = generator.evaluate_batch(batch)

            print(
                f"{generator.get_generator_name()} - Average log likelihood: {avg_log_likelihood:.4f}"
            )

        if i == 2:
            break

    print("\nGenerated names (DumbNameGenerator):", dumb_generator.generate_names(5))

    for k, ngram_generator in ngram_generators.items():

        print(
            f"\nGenerated names ({ngram_generator.get_generator_name()}):",
            ngram_generator.generate_names(5),
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