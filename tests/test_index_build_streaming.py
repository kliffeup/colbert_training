"""CPU integration test for the streaming (disk-backed) index build.

Exercises the full ``build_index`` orchestration — sampling, codec-param training
from the Stage-1 sample, the fused encode→compress streaming pass, ``.npy``
finalization, and Stage-4 inversion — with a tiny fake model (no GPU/HF model).
Then loads every artifact back via ``IndexSaver`` and decodes a document to prove
the doc-order layout the retriever depends on is intact.
"""

import numpy as np
import torch

from colbert.config import ColBERTConfig
from colbert.indexing.index_builder import build_index
from colbert.indexing.saver import IndexSaver
from colbert.dataset.collection import Collection


DIM = 16


class FakeModel:
    """Deterministic stand-in for ColBERT: maps each doc to N normalized vectors,
    where N = number of whitespace tokens (so doc lengths vary, incl. empty)."""

    def eval(self):
        return self

    def encode_docs(self, texts, maxlen=None):
        lengths = [min(len(t.split()), maxlen or 10_000) for t in texts]
        S = max(lengths + [1])
        B = len(texts)
        D = torch.zeros(B, S, DIM)
        mask = torch.zeros(B, S, dtype=torch.bool)
        for i, (t, n) in enumerate(zip(texts, lengths)):
            if n == 0:
                continue
            # Seed per (doc, token) for reproducibility independent of batch order.
            g = torch.Generator().manual_seed(abs(hash(t)) % (2**31))
            v = torch.randn(n, DIM, generator=g)
            D[i, :n] = torch.nn.functional.normalize(v, dim=1)
            mask[i, :n] = True
        return D, mask


def _write_collection(path, n_docs=64):
    lines = []
    rng = np.random.default_rng(0)
    for i in range(n_docs):
        n_tok = int(rng.integers(20, 45))
        words = " ".join(f"w{rng.integers(0, 50)}" for _ in range(n_tok))
        lines.append(f"D{i}\t{words}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_streaming_build_end_to_end(tmp_path):
    coll_path = tmp_path / "collection.tsv"
    _write_collection(coll_path, n_docs=64)
    collection = Collection(str(coll_path))

    index_path = tmp_path / "index"
    config = ColBERTConfig(
        dim=DIM, nbits=2, doc_maxlen=64, query_maxlen=8,
        kmeans_niters=2, index_path=str(index_path),
    )

    out = build_index(
        FakeModel(), collection, config,
        index_path=str(index_path), batch_size=8, num_gpus=1,
    )
    assert out == str(index_path)

    saver = IndexSaver(index_path)
    centroid_ids, packed = saver.load_compressed_embeddings()
    doclens = saver.load_doclens()
    pids = saver.load_pids()
    ivl = saver.load_inverted_lists()
    meta = saver.load_metadata()
    codec = saver.load_codec()

    # Per-doc metadata is aligned and complete.
    assert len(pids) == 64
    assert len(doclens) == 64
    total = int(doclens.sum())
    assert centroid_ids.shape == (total,)
    assert packed.shape == (total, codec.bytes_per_vector - 4)
    assert meta["num_embeddings"] == total
    assert meta["num_passages"] == 64

    # Inverted lists reference every embedding exactly once.
    all_ids = sorted(i for lst in ivl.values() for i in lst)
    assert all_ids == list(range(total))

    # Doc-order invariant: token rows for doc k live at offsets [sum(doclens[:k]),
    # sum(doclens[:k]+doclen[k])). Decode one doc and confirm it round-trips.
    offsets = np.zeros(len(doclens) + 1, dtype=np.int64)
    np.cumsum(doclens, out=offsets[1:])
    k = 5
    s, e = int(offsets[k]), int(offsets[k + 1])
    decoded = codec.decode(centroid_ids[s:e], packed[s:e])
    assert decoded.shape == (doclens[k], DIM)

    # The decoded vectors should be close to the fake model's originals for that pid.
    fake = FakeModel()
    D, m = fake.encode_docs([collection[pids[k]]], maxlen=config.doc_maxlen)
    original = D[0][m[0]]
    cos = torch.nn.functional.cosine_similarity(original, decoded, dim=1)
    assert cos.mean() > 0.5
