"""Print multiple example triples from the triples TSV using StreamingTriplesDataset."""

from itertools import islice

from colbert.dataset.triples import StreamingTriplesDataset

NUM_EXAMPLES = 5
TRIPLES_PATH = "data/triples.train.small.tsv"

dataset = StreamingTriplesDataset(path=TRIPLES_PATH, shuffle_buffer_size=1000, seed=42)

for i, (query, positive, negative) in enumerate(islice(dataset, NUM_EXAMPLES)):
    print(f"--- Example {i + 1} ---")
    print(f"  Query:    {query[:120]}...")
    print(f"  Positive: {positive[:120]}...")
    print(f"  Negative: {negative[:120]}...")
    print()
